"""
Production Clinical Endpoints: Auth, Firestore, Storage, Predictor, and SSE Streaming.
====================================================================================
Wires up the complete end-to-end clinical workflow:
- POST /api/auth/profile: Saves patient onboarding profile to user-isolated Firestore.
- GET /api/auth/profile: Retrieves patient onboarding profile.
- POST /api/predict: Direct symptom prediction using ClinicalPredictionPipeline.
- POST /api/sessions: Creates a consultation session.
- GET /api/sessions: Lists user's consultation sessions.
- GET /api/sessions/{session_id}: 1-Read Session Context lookup.
- POST /api/upload: Uploads patient lab PDFs and rash photos to Firebase Storage.
- GET /api/files/{uid}/{filename}: Serves uploaded documents and photos.
- POST /api/chat/stream: SSE real-time streaming with Bearer JWT auth and single-write persistence.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from email.parser import BytesParser
from email.policy import default as email_default_policy
from typing import Any, List, Optional

import hashlib
import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from apps.context_synthesizer import context_synthesizer
from apps.firebase_service import firebase_service
from apps.vision_service import vision_service
from core.config import settings
from core.mailer import mailer_service
from model_building.pipeline import (
    ClinicalPredictionPipeline,
    DiagnosticReportSchema,
    SymptomInputRequest,
)
from retrieval.medical_evidence_engine import medical_evidence_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["HealthTrack Clinical Engine"])
security = HTTPBearer(auto_error=False)

# Shared singleton pipeline instance
clinical_pipeline = ClinicalPredictionPipeline()


# ── Security Dependency ─────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    authorization: Optional[str] = Header(None),
) -> dict[str, Any]:
    """
    Validates Firebase Bearer JWT Token.
    Supports Authorization header directly or via HTTPBearer.
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = firebase_service.verify_id_token(token)
        return claims
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authentication token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Schemas ────────────────────────────────────────────────────────────────────

class UserProfileSchema(BaseModel):
    name: Optional[str] = Field("Self", description="Given name or display label")
    age: Optional[float] = Field(None, ge=0, le=125, description="Patient age in years")
    biological_sex: Optional[str] = Field(None, description="'MALE', 'FEMALE', or 'INTERSEX'")
    is_pregnant: Optional[bool] = Field(False, description="Whether patient is currently pregnant")
    chronic_conditions: List[str] = Field(default_factory=list, description="Known chronic diagnoses")
    current_medications: List[str] = Field(default_factory=list, description="Active prescription & OTC drugs")
    drug_allergies: List[str] = Field(default_factory=list, description="Known drug allergies")
    baseline_symptoms: List[str] = Field(default_factory=list, description="Checklist symptoms from onboarding form")


class MemberProfileSchema(BaseModel):
    name: str = Field(..., min_length=1, description="Given name or display label (e.g. 'Dad', 'Mom', 'Self')")
    relationship: str = Field(default="OTHER", description="'SELF', 'FATHER', 'MOTHER', 'SPOUSE', 'CHILD', 'OTHER'")
    age: Optional[float] = Field(None, ge=0, le=125, description="Patient age in years")
    biological_sex: Optional[str] = Field(None, description="'MALE', 'FEMALE', or 'INTERSEX'")
    is_pregnant: Optional[bool] = Field(False, description="Whether patient is currently pregnant")
    chronic_conditions: List[str] = Field(default_factory=list, description="Known chronic diagnoses")
    current_medications: List[str] = Field(default_factory=list, description="Active prescription & OTC drugs")
    drug_allergies: List[str] = Field(default_factory=list, description="Known drug allergies")
    baseline_symptoms: List[str] = Field(default_factory=list, description="Checklist symptoms from onboarding form")
    longitudinal_summary: List[dict[str, Any]] = Field(default_factory=list, description="Past episodes & diagnoses")


class MemberProfileUpdateSchema(BaseModel):
    name: Optional[str] = None
    relationship: Optional[str] = None
    age: Optional[float] = None
    biological_sex: Optional[str] = None
    is_pregnant: Optional[bool] = None
    chronic_conditions: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    drug_allergies: Optional[List[str]] = None
    baseline_symptoms: Optional[List[str]] = None


class ConditionRecordSchema(BaseModel):
    condition: str = Field(..., min_length=1, description="Diagnosed condition or syndrome name")
    icd10: Optional[str] = Field(None, description="Optional ICD-10 diagnosis code")
    diagnosed_date: Optional[str] = Field(None, description="Date of diagnosis (YYYY-MM-DD)")
    status: str = Field(default="CHRONIC_ACTIVE", description="'CHRONIC_ACTIVE', 'RESOLVED', 'EPISODIC'")
    notes: Optional[str] = Field(None, description="Clinical notes or medication context")
    session_id: Optional[str] = Field(None, description="Originating consultation session ID")


class CreateSessionRequest(BaseModel):
    member_id: Optional[str] = Field(default="mem_primary", description="Target family member ID (e.g. 'mem_dad', 'mem_mom')")
    title: Optional[str] = Field(None, description="Optional custom session title")
    initial_symptoms: Optional[List[str]] = Field(None, description="Initial checklist symptoms")


class ChatStreamRequest(BaseModel):
    session_id: str = Field(..., description="ID of the consultation session")
    query_text: str = Field(..., min_length=1, description="Patient narrative query or symptom description")
    selected_symptoms: Optional[List[str]] = Field(None, description="Checklist symptoms")
    document_id: Optional[str] = Field(None, description="Optional uploaded lab report or rash photo ID")


class SendOtpRequest(BaseModel):
    email: str = Field(..., description="Recipient email address")
    name: Optional[str] = Field("Patient", description="Patient given name")


class VerifyOtpRequest(BaseModel):
    email: str = Field(..., description="Recipient email address")
    otp_code: str = Field(..., min_length=6, max_length=6, description="6-digit verification code")
    name: Optional[str] = Field(None, description="Optional patient given name for account creation")


# ── User Authentication & OTP Endpoints (Diagram 3) ──────────────────────────

@router.post("/auth/otp/send")
async def send_otp_endpoint(req: SendOtpRequest) -> dict[str, Any]:
    """
    SMTP OTP Dispatch Endpoint (Diagram 3 Architecture).
    Generates cryptographically random 6-digit verification code with 10-minute validity.
    """
    code = mailer_service.generate_and_store_otp(req.email)
    res = mailer_service.send_otp_email(req.email, code, req.name or "Patient")
    return {
        "status": "SUCCESS",
        "message": f"Verification code sent to {req.email}",
        "mode": res.get("mode"),
        "debug_code": res.get("debug_code"),
    }


@router.post("/auth/otp/verify")
async def verify_otp_endpoint(req: VerifyOtpRequest) -> dict[str, Any]:
    """
    SMTP OTP Verification Endpoint (Diagram 3 Architecture).
    Validates code, rate-limits brute force attempts, and issues authenticated patient session.
    """
    ok, msg = mailer_service.verify_otp(req.email, req.otp_code)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    # Derive unique deterministic tenant UID from verified email
    clean_email = req.email.strip().lower()
    uid = f"user_{hashlib.md5(clean_email.encode()).hexdigest()[:10]}"

    # Initialize or fetch profile
    profile = firebase_service.get_user_profile(uid)
    if not profile or not profile.get("name") or profile.get("name") == "Self":
        name = req.name or clean_email.split("@")[0].capitalize()
        profile = firebase_service.save_user_profile(uid, {
            "name": name,
            "email": clean_email,
            "verified": True,
        })

    token = f"dev_{uid}"
    return {
        "status": "SUCCESS",
        "message": "Account email verified successfully.",
        "token": token,
        "uid": uid,
        "profile": profile,
    }


# ── User Onboarding & Profile Endpoints ────────────────────────────────────────

@router.post("/auth/profile")
async def save_profile(
    profile: UserProfileSchema,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Saves or updates patient onboarding profile in user-isolated Firestore path."""
    uid = user["uid"]
    saved = firebase_service.save_user_profile(uid, profile.model_dump())
    return {"status": "SUCCESS", "profile": saved}


@router.get("/auth/profile")
async def get_profile(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieves patient onboarding profile from user-isolated Firestore path."""
    uid = user["uid"]
    profile = firebase_service.get_user_profile(uid)
    return {"status": "SUCCESS", "profile": profile}


# ── Multi-Patient Family Members & Longitudinal History ────────────────────

@router.post("/members")
async def create_member_endpoint(
    member: MemberProfileSchema,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Registers an independent family member profile (e.g. Dad, Mom, Child)."""
    uid = user["uid"]
    record = firebase_service.create_member(uid, member.model_dump())
    return {"status": "SUCCESS", "member": record}


@router.get("/members")
async def list_members_endpoint(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Lists all family members registered under the authenticated user account."""
    uid = user["uid"]
    members = firebase_service.list_members(uid)
    return {"status": "SUCCESS", "members": members, "count": len(members)}


@router.get("/members/{member_id}")
async def get_member_endpoint(
    member_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieves full profile, chronic conditions, and past episodes for a specific member."""
    uid = user["uid"]
    member = firebase_service.get_member(uid, member_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Member {member_id} not found.")
    return {"status": "SUCCESS", "member": member}


@router.put("/members/{member_id}")
async def update_member_endpoint(
    member_id: str,
    updates: MemberProfileUpdateSchema,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Updates demographic or clinical baseline for a specific family member."""
    uid = user["uid"]
    try:
        updated = firebase_service.update_member(uid, member_id, updates.model_dump(exclude_unset=True))
        return {"status": "SUCCESS", "member": updated}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Member {member_id} not found.")


@router.post("/members/{member_id}/conditions")
async def add_condition_endpoint(
    member_id: str,
    condition: ConditionRecordSchema,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Appends a diagnosed medical condition or clinical milestone to a member's
    longitudinal health record (e.g. Coronary Artery Disease diagnosed 1 month ago).
    """
    uid = user["uid"]
    try:
        updated = firebase_service.add_member_condition(uid, member_id, condition.model_dump())
        return {"status": "SUCCESS", "member": updated}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Member {member_id} not found.")


@router.get("/members/{member_id}/timeline")
async def get_timeline_endpoint(
    member_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Returns chronological disease history and past clinical episodes for a member."""
    uid = user["uid"]
    timeline = firebase_service.get_member_timeline(uid, member_id)
    return {"status": "SUCCESS", "timeline": timeline, "count": len(timeline)}


# ── Standalone Disease Prediction Endpoint ─────────────────────────────────────

@router.post("/predict", response_model=DiagnosticReportSchema)
async def predict_disease(
    req: SymptomInputRequest,
    user: Optional[dict[str, Any]] = Depends(get_current_user),
) -> DiagnosticReportSchema:
    """
    Executes ClinicalPredictionPipeline directly.
    If authenticated, automatically defaults patient age and sex from profile if omitted.
    """
    if user:
        profile = firebase_service.get_user_profile(user["uid"])
        if req.age is None and profile.get("age"):
            req.age = float(profile["age"])
        if req.biological_sex is None and profile.get("biological_sex"):
            req.biological_sex = str(profile["biological_sex"])

    report = clinical_pipeline.predict(req)
    return report


@router.get("/classes")
async def list_clinical_classes(category: Optional[str] = None, tier: Optional[str] = None) -> dict[str, Any]:
    """
    Returns the complete structured catalog of all 580 medical condition classes,
    properly arranged into clinical organ systems with typical symptom hallmarks.
    Supports filtering by category or model tier ('primary_41', 'general_50', 'all_580').
    """
    catalog_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "clinical_classes_catalog.json"
    if not catalog_path.exists():
        return {"status": "SUCCESS", "total_classes": 0, "classes": []}

    with open(catalog_path, encoding="utf-8") as f:
        classes = json.load(f)

    if category and category.lower() != "all systems":
        classes = [c for c in classes if c.get("category", "").lower() == category.lower()]
    if tier and tier.lower() != "all":
        classes = [c for c in classes if c.get("tier", "").lower() == tier.lower()]

    return {
        "status": "SUCCESS",
        "total_classes": len(classes),
        "classes": classes,
    }


# ── Consultation Session Endpoints ─────────────────────────────────────────────

@router.post("/sessions")
async def create_session_endpoint(
    req: CreateSessionRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Creates a new consultation session bound to a specific family member (Person A vs B).
    Embeds a denormalized member_snapshot to guarantee single-read access on streaming.
    """
    uid = user["uid"]
    session_record = firebase_service.create_session(
        uid=uid,
        title=req.title,
        member_id=req.member_id or "mem_primary",
        initial_data={"initial_symptoms": req.initial_symptoms or []},
    )
    return {"status": "SUCCESS", "session": session_record}


@router.get("/sessions")
async def list_sessions_endpoint(
    member_id: Optional[str] = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Lists consultation sessions belonging to the patient, optionally filtered by member role."""
    uid = user["uid"]
    sessions = firebase_service.list_user_sessions(uid, member_id=member_id)
    return {"status": "SUCCESS", "sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}")
async def get_session_endpoint(
    session_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    SINGLE-READ LOOKUP: Retrieves full session context and patient profile
    costing exactly 1 Firestore document read.
    """
    uid = user["uid"]
    context = firebase_service.get_session_context(uid=uid, session_id=session_id)
    return {"status": "SUCCESS", "context": context}


# ── Document & Image Upload (Firebase Storage) ─────────────────────────────────

@router.post("/upload")
async def upload_document_or_photo(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Uploads patient lab report (PDF) or rash photo to Firebase Cloud Storage.
    Supports:
    - multipart/form-data (native stdlib parsing, zero external dependency)
    - application/json with {"filename": str, "content_type": str, "file_base64": str, "document_type": str}
    - raw binary with headers (Content-Type, X-Filename, X-Document-Type)
    Tenant-Isolated Path: users/{uid}/documents/{file_id}_{filename}
    """
    uid = user["uid"]
    req_content_type = request.headers.get("content-type", "")

    file_bytes: bytes = b""
    filename: str = "upload"
    file_content_type: str = "application/octet-stream"
    document_type: str = "MEDICAL_DOCUMENT"

    if req_content_type.startswith("application/json"):
        body = await request.json()
        b64 = body.get("file_base64") or body.get("data")
        if not b64:
            raise HTTPException(status_code=400, detail="Missing 'file_base64' in JSON upload payload.")
        file_bytes = base64.b64decode(b64)
        filename = body.get("filename", "upload")
        file_content_type = body.get("content_type", "application/octet-stream")
        document_type = body.get("document_type", "MEDICAL_DOCUMENT")

    elif "multipart/form-data" in req_content_type:
        body_bytes = await request.body()
        ct_header = f"Content-Type: {req_content_type}\r\n\r\n".encode("utf-8")
        msg = BytesParser(policy=email_default_policy).parsebytes(ct_header + body_bytes)
        for part in msg.iter_parts():
            part_name = part.get_param("name", header="content-disposition")
            if part_name == "file" or part.get_filename():
                file_bytes = part.get_payload(decode=True) or b""
                filename = part.get_filename() or "upload"
                file_content_type = part.get_content_type() or "application/octet-stream"
            elif part_name == "document_type":
                payload = part.get_payload(decode=True)
                if payload:
                    document_type = payload.decode("utf-8").strip()

    else:
        file_bytes = await request.body()
        filename = request.headers.get("x-filename", "upload")
        file_content_type = req_content_type or "application/octet-stream"
        document_type = request.headers.get("x-document-type", "MEDICAL_DOCUMENT")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file content received.")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed 10MB limit.",
        )

    upload_record = firebase_service.upload_file(
        uid=uid,
        file_bytes=file_bytes,
        filename=filename,
        content_type=file_content_type,
    )

    # Process via Clinical Vision and Document Extractor
    obs = await vision_service.process_artifact(
        file_bytes=file_bytes,
        filename=filename,
        content_type=file_content_type,
        document_type=document_type,
    )

    prefix = "Visual findings: " if obs.artifact_type == "DERMATOLOGY_IMAGE" else "Document findings: "
    clinical_summary = f"{prefix}{obs.extracted_clinical_summary}"
    upload_record["extracted_clinical_summary"] = clinical_summary
    upload_record["artifact_type"] = obs.artifact_type
    upload_record["anatomical_or_panel_type"] = obs.anatomical_or_panel_type
    upload_record["abnormal_values"] = obs.abnormal_values
    upload_record["red_flag_alert"] = obs.red_flag_alert

    firebase_service.update_file_clinical_findings(
        uid=uid,
        file_id=upload_record["file_id"],
        extracted_clinical_summary=clinical_summary,
        artifact_type=obs.artifact_type,
        anatomical_or_panel_type=obs.anatomical_or_panel_type,
        abnormal_values=obs.abnormal_values,
        red_flag_alert=obs.red_flag_alert,
    )
    return {"status": "SUCCESS", "file": upload_record}


@router.get("/files/{uid}/{filename}")
async def get_file_endpoint(
    uid: str,
    filename: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Securely downloads uploaded patient document/image with tenant isolation check."""
    if user["uid"] != uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to requested file.")

    res = firebase_service.get_uploaded_file(uid, filename)
    if not res:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    file_bytes, content_type = res
    return Response(content=file_bytes, media_type=content_type)


# ── Server-Sent Events (SSE) Real-Time Chat Stream ─────────────────────────────

@router.post("/chat/stream")
async def chat_sse_stream(
    req: ChatStreamRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) streaming consultation endpoint:
    1. Authenticates Bearer JWT token.
    2. Single Read: Fetches session context and profile (1 read).
    3. Runs ClinicalPredictionPipeline (0 DB calls).
    4. Synthesizes compact clinical context prompt (< 350 tokens).
    5. Streams tokens in real time (text/event-stream).
    6. Single Write: On completion, persists turn and context (1 write).
    """
    uid = user["uid"]

    # 1. SINGLE READ from Firestore (profile + recent context)
    session_data = firebase_service.get_session_context(uid=uid, session_id=req.session_id)
    profile = session_data.get("profile", {})
    recent_turns = session_data.get("recent_context_window", [])

    # Accumulate affirmed symptoms across multi-turn consultation to solve symptom amnesia
    prior_symptoms: set[str] = set(session_data.get("cumulative_symptoms", []))
    for turn in recent_turns:
        for s in turn.get("extracted_present_symptoms", []):
            if s:
                prior_symptoms.add(s)

    combined_symptoms = list(set(req.selected_symptoms or []) | prior_symptoms)

    # 2. Resolve Vision & Document Notes if document_id provided
    vision_notes = None
    vision_findings_text = ""
    if req.document_id:
        file_rec = firebase_service.get_uploaded_file_record(uid, req.document_id)
        if file_rec:
            summary = file_rec.get("extracted_clinical_summary") or ""
            abnormals = file_rec.get("abnormal_values") or []
            if isinstance(abnormals, str):
                try:
                    abnormals = json.loads(abnormals)
                except Exception:
                    abnormals = [abnormals]
            red_flag = file_rec.get("red_flag_alert")

            parts = [f"Uploaded File: {file_rec.get('filename')} ({file_rec.get('artifact_type', 'MEDICAL_ARTIFACT')})"]
            if summary:
                parts.append(f"Clinical Findings: {summary}")
            if abnormals:
                parts.append(f"Abnormal Values: {', '.join(abnormals)}")
            if red_flag:
                parts.append(f"Alert: {red_flag}")
            vision_notes = "\n".join(parts)
            if summary or abnormals:
                vision_findings_text = f" Uploaded lab/image findings: {summary}. Abnormal values: {', '.join(abnormals)}."

    # 3. RUN UPSTREAM PREDICTOR (grounded on multi-turn symptoms and vision findings)
    query_with_context = f"{req.query_text}{vision_findings_text}"
    prediction_req = SymptomInputRequest(
        query_text=query_with_context,
        selected_symptoms=combined_symptoms if combined_symptoms else None,
        age=profile.get("age"),
        biological_sex=profile.get("biological_sex"),
        top_k=3,
    )
    diagnostic_report = clinical_pipeline.predict(prediction_req)

    # 4. Retrieve Medical Evidence from NIH MedlinePlus / PubMed
    top_conds = [c.condition for c in diagnostic_report.differential_diagnosis[:2]]
    if not top_conds and diagnostic_report.top_condition:
        top_conds = [diagnostic_report.top_condition]
    evidence_citations = medical_evidence_engine.get_multi_condition_evidence(
        top_conds, present_symptoms=diagnostic_report.extracted_present_symptoms
    )

    # 5. CONTEXT ASSEMBLER (< 385 tokens) with Doctor Persona, Evidence & Acute-on-Chronic Intelligence
    chat_messages = context_synthesizer.build_chat_messages(
        profile=profile,
        diagnostic_report=diagnostic_report,
        recent_turns=recent_turns,
        user_query=req.query_text,
        vision_notes=vision_notes,
    )
    compiled_prompt = context_synthesizer.build_prompt(
        profile=profile,
        diagnostic_report=diagnostic_report,
        recent_turns=recent_turns,
        user_query=req.query_text,
        vision_notes=vision_notes,
    )
    acute_alert = context_synthesizer.detect_acute_on_chronic(profile, diagnostic_report, req.query_text)

    # 6. SSE Stream Generator with Single-Write Hook
    async def sse_event_generator():
        accumulated_chunks: list[str] = []

        # Send initial diagnostic report metadata as first SSE event
        meta_payload = {
            "type": "diagnostic_report",
            "member_id": session_data.get("member_id", "mem_primary"),
            "member_name": session_data.get("member_name", "Patient"),
            "member_relationship": session_data.get("member_relationship", "SELF"),
            "top_condition": diagnostic_report.top_condition,
            "top_confidence": diagnostic_report.top_confidence,
            "triage_level": diagnostic_report.triage_level,
            "triage_recommendation": diagnostic_report.triage_recommendation,
            "conformal_set": diagnostic_report.conformal_differential_set,
            "is_emergency": bool(diagnostic_report.emergency_alert and diagnostic_report.emergency_alert.is_emergency),
            "is_acute_on_chronic": bool(acute_alert),
            "acute_on_chronic_alert": acute_alert,
            "retrieved_evidence": evidence_citations,
            "model_name": settings.DEFAULT_MODEL,
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        # Check if Ollama is available; primary via /api/chat with fallback to /api/generate
        ollama_stream_success = False
        try:
            stream_timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
            async with httpx.AsyncClient(timeout=stream_timeout) as client:
                ollama_chat_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
                async with client.stream(
                    "POST",
                    ollama_chat_url,
                    json={
                        "model": settings.DEFAULT_MODEL,
                        "messages": chat_messages,
                        "stream": True,
                        "options": {"temperature": 0.2},
                    },
                ) as stream_resp:
                    if stream_resp.status_code == 200:
                        async for line in stream_resp.aiter_lines():
                            if line.strip():
                                chunk_json = json.loads(line)
                                token = chunk_json.get("message", {}).get("content", "")
                                thinking = chunk_json.get("thinking", "")
                                if thinking:
                                    yield f"data: {json.dumps({'type': 'thinking', 'token': thinking})}\n\n"
                                if token:
                                    accumulated_chunks.append(token)
                                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                                if chunk_json.get("done", False):
                                    break
                        if accumulated_chunks:
                            ollama_stream_success = True
                    else:
                        # Fallback to /api/generate if /api/chat is unsupported
                        ollama_gen_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
                        async with client.stream(
                            "POST",
                            ollama_gen_url,
                            json={
                                "model": settings.DEFAULT_MODEL,
                                "prompt": compiled_prompt,
                                "stream": True,
                                "options": {"temperature": 0.2},
                            },
                        ) as gen_resp:
                            if gen_resp.status_code == 200:
                                async for line in gen_resp.aiter_lines():
                                    if line.strip():
                                        chunk_json = json.loads(line)
                                        token = chunk_json.get("response", "")
                                        if token:
                                            accumulated_chunks.append(token)
                                            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                                        if chunk_json.get("done", False):
                                            break
                                if accumulated_chunks:
                                    ollama_stream_success = True
        except Exception as e:
            logger.info(f"Ollama stream bypass ({e}); using calibrated generative stream.")

        # Fallback calibrated doctor response stream if Ollama is offline or sandboxed
        if not ollama_stream_success:
            paragraphs = []
            if diagnostic_report.emergency_alert and diagnostic_report.emergency_alert.is_emergency:
                paragraphs.append(
                    f"⚠️ **[CRITICAL EMERGENCY WARNING]**: {diagnostic_report.emergency_alert.message}\n"
                    f"**Action Required**: {diagnostic_report.emergency_alert.action}. Please call 911 or go to the nearest emergency department immediately."
                )

            if acute_alert:
                paragraphs.append(
                    f"⚠️ **[LONGITUDINAL CLINICAL CAUTION]**: {acute_alert}"
                )

            is_non_symptom_or_undetermined = (
                not diagnostic_report.extracted_present_symptoms
                or diagnostic_report.top_confidence == 0.0
                or "undetermined" in diagnostic_report.top_condition.lower()
            )

            patient_name = profile.get("name", "Patient")
            relation = profile.get("relationship", "SELF")

            if is_non_symptom_or_undetermined:
                address_target = f"for **{patient_name}**" if relation != "SELF" else f"**{patient_name}**"
                paragraphs.append(
                    f"Hello, I am Dr. Morgan, your clinical telehealth physician. I am here to help evaluate symptoms and health concerns {address_target}."
                )
                paragraphs.append(
                    "To help me provide an accurate and safe medical evaluation, please share what is going on:\n"
                    "• **Main Symptoms**: What specific discomfort or pain are you experiencing?\n"
                    "• **Timeline**: When did this begin, and has it gotten steadily worse or come and gone?\n"
                    "• **Medical Attachments**: If you have recent blood test results (PDF) or photos of a rash, you can attach them using the paperclip button."
                )
                paragraphs.append(
                    "Take your time to describe what you are feeling, and we will review it carefully together."
                )
            else:
                paragraphs.append(
                    f"Thank you for sharing your symptoms, {patient_name}. Based on our clinical diagnostic analysis, "
                    f"the primary suspected condition is **{diagnostic_report.top_condition}** "
                    f"({diagnostic_report.top_confidence:.1f}% confidence)."
                )

                if diagnostic_report.differential_diagnosis:
                    top_cand = diagnostic_report.differential_diagnosis[0]
                    paragraphs.append(f"**Clinical Assessment**: {top_cand.explanation_text}")

                if len(diagnostic_report.differential_diagnosis) > 1:
                    other_conds = [
                        f"{c.condition} ({c.confidence_percentage}%)"
                        for c in diagnostic_report.differential_diagnosis[1:3]
                    ]
                    paragraphs.append(f"Other clinical considerations we are monitoring include: {', '.join(other_conds)}.")

                if diagnostic_report.active_inquiry and diagnostic_report.active_inquiry.is_active_inquiry_needed:
                    qs = "\n".join(f"• {q}" for q in diagnostic_report.active_inquiry.high_yield_questions[:2])
                    paragraphs.append(f"**To clarify your clinical picture, please let me know**:\n{qs}")

                paragraphs.append(
                    "**Supportive Care Guidance**: Please prioritize rest and hydration. If you develop new red flags (such as sudden shortness of breath, severe chest pain, or high unrelenting fever), please seek emergency care right away."
                )

            paragraphs.append(f"\n*{diagnostic_report.disclaimer}*")

            full_synthetic_text = "\n\n".join(paragraphs)
            words = full_synthetic_text.split(" ")
            for w in words:
                token = w + " "
                accumulated_chunks.append(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        # 5. SINGLE WRITE on Completion: Persist turn in exactly 1 Firestore document write
        complete_assistant_response = "".join(accumulated_chunks).strip()
        firebase_service.append_session_turn(
            uid=uid,
            session_id=req.session_id,
            user_query=req.query_text,
            assistant_response=complete_assistant_response,
            diagnostic_report=diagnostic_report.model_dump(),
        )

        # 6. CONTINUOUS RLHF REPLAY BUFFER: Log interaction for offline/online alignment
        try:
            buffer_file = settings.DATA_DIR / "clinical_interaction_buffer.jsonl"
            buffer_file.parent.mkdir(parents=True, exist_ok=True)
            log_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": req.session_id,
                "user_id": uid,
                "user_query": req.query_text,
                "selected_symptoms": req.selected_symptoms or [],
                "extracted_symptoms": diagnostic_report.extracted_present_symptoms,
                "denied_symptoms": diagnostic_report.extracted_denied_symptoms,
                "top_condition": diagnostic_report.top_condition,
                "top_confidence": diagnostic_report.top_confidence,
                "triage_level": diagnostic_report.triage_level,
                "prompt": compiled_prompt,
                "assistant_response": complete_assistant_response,
                "retrieved_evidence": evidence_citations,
                "model": settings.DEFAULT_MODEL,
                "allergies": profile.get("drug_allergies", []),
                "chronic_conditions": profile.get("chronic_conditions", []),
            }
            with open(buffer_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\n")
        except Exception as buffer_err:
            logger.warning(f"Failed to record clinical interaction to replay buffer: {buffer_err}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
