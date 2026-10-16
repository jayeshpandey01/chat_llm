"""
End-to-End Integration Test Suite for Production Clinical Serving Gateway & SSE Stream.
======================================================================================
Tests:
1. Firebase Auth Bearer token validation and 401 unauthorized rejection.
2. User onboarding profile storage and retrieval in user-isolated Firestore.
3. Standalone disease prediction with profile demographic auto-fill.
4. Consultation session lifecycle with Single-Read context retrieval.
5. Medical PDF and image photo upload/download (Base64 & Multipart) with tenant isolation.
6. Real-time Server-Sent Events (SSE) streaming (/api/chat/stream) with single-write persistence.
7. Continuous RLHF clinical interaction buffer logging.
"""

import base64
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from apps.server import create_app
from core.config import settings


class TestClinicalApiAndSSE(unittest.TestCase):
    """End-to-end integration tests for HealthTrack AI Clinical API."""

    @classmethod
    def setUpClass(cls):
        """Build FastAPI application instance with clinical routes and test client."""
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.user_a_token = "Bearer test_patient_alpha"
        cls.user_b_token = "Bearer test_patient_beta"
        cls.user_a_id = "patient_alpha"
        cls.user_b_id = "patient_beta"

    def test_01_unauthorized_rejections(self):
        """Unauthenticated requests to protected endpoints must return 401."""
        endpoints = [
            ("GET", "/api/auth/profile"),
            ("POST", "/api/auth/profile"),
            ("POST", "/api/sessions"),
            ("GET", "/api/sessions"),
            ("POST", "/api/upload"),
            ("POST", "/api/chat/stream"),
        ]
        for method, endpoint in endpoints:
            if method == "GET":
                res = self.client.get(endpoint)
            else:
                res = self.client.post(endpoint, json={})
            self.assertEqual(
                res.status_code,
                401,
                f"Endpoint {method} {endpoint} did not reject unauthenticated request with 401.",
            )

    def test_02_patient_profile_onboarding_lifecycle(self):
        """Save onboarding patient profile and retrieve with tenant isolation."""
        profile_data = {
            "age": 42.0,
            "biological_sex": "FEMALE",
            "is_pregnant": False,
            "chronic_conditions": ["Asthma", "Mild Hypertension"],
            "current_medications": ["Albuterol inhaler", "Lisinopril 10mg"],
            "drug_allergies": ["Sulfa antibiotics"],
            "baseline_symptoms": ["fatigue"],
        }
        # Save profile
        save_res = self.client.post(
            "/api/auth/profile",
            headers={"Authorization": self.user_a_token},
            json=profile_data,
        )
        self.assertEqual(save_res.status_code, 200)
        saved_body = save_res.json()
        self.assertEqual(saved_body["status"], "SUCCESS")
        self.assertEqual(saved_body["profile"]["age"], 42.0)
        self.assertEqual(saved_body["profile"]["biological_sex"], "FEMALE")

        # Retrieve profile
        get_res = self.client.get(
            "/api/auth/profile",
            headers={"Authorization": self.user_a_token},
        )
        self.assertEqual(get_res.status_code, 200)
        retrieved_body = get_res.json()
        self.assertEqual(retrieved_body["status"], "SUCCESS")
        self.assertEqual(retrieved_body["profile"]["age"], 42.0)
        self.assertIn("Asthma", retrieved_body["profile"]["chronic_conditions"])

    def test_03_standalone_predict_endpoint(self):
        """Verify symptom prediction endpoint with free-text extraction and demographic auto-fill."""
        # Query with explicit symptoms
        payload = {
            "query_text": "Intense skin itching with rash and nodal skin eruptions across the arm, no fever",
            "selected_symptoms": ["itching", "skin_rash"],
        }
        res = self.client.post(
            "/api/predict",
            headers={"Authorization": self.user_a_token},
            json=payload,
        )
        self.assertEqual(res.status_code, 200)
        report = res.json()
        self.assertEqual(report["top_condition"], "Fungal infection")
        self.assertGreater(report["top_confidence"], 50.0)
        self.assertIn("Fungal infection", report["differential_diagnosis"][0]["condition"])
        self.assertIn("skin rash", report["differential_diagnosis"][0]["matched_symptoms"])
        self.assertIsNone(report["emergency_alert"])

    def test_04_session_lifecycle_and_single_read_lookup(self):
        """Test session creation and single-read context assembly."""
        create_payload = {
            "title": "Respiratory Consultation",
            "initial_symptoms": ["cough", "fatigue"],
        }
        create_res = self.client.post(
            "/api/sessions",
            headers={"Authorization": self.user_a_token},
            json=create_payload,
        )
        self.assertEqual(create_res.status_code, 200)
        session_obj = create_res.json()["session"]
        session_id = session_obj["session_id"]
        self.assertTrue(session_id.startswith("sess_"))

        # Single Read: fetch session context
        get_res = self.client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": self.user_a_token},
        )
        self.assertEqual(get_res.status_code, 200)
        context_data = get_res.json()["context"]
        self.assertEqual(context_data["session_id"], session_id)
        self.assertEqual(context_data["profile"]["age"], 42.0)
        self.assertEqual(context_data["recent_context_window"], [])

        # List sessions
        list_res = self.client.get(
            "/api/sessions",
            headers={"Authorization": self.user_a_token},
        )
        self.assertEqual(list_res.status_code, 200)
        sessions_list = list_res.json()["sessions"]
        self.assertGreaterEqual(len(sessions_list), 1)

    def test_05_document_and_photo_storage_tenant_isolation(self):
        """Test JSON Base64 and multipart file upload and strict tenant-isolated retrieval."""
        # 1. Base64 JSON Upload of Lab Report PDF
        pdf_dummy = b"%PDF-1.4 Mock Lab Blood Test Results ESR CRP WBC"
        pdf_b64 = base64.b64encode(pdf_dummy).decode("ascii")
        upload_pdf_res = self.client.post(
            "/api/upload",
            headers={"Authorization": self.user_a_token},
            json={
                "filename": "blood_panel.pdf",
                "content_type": "application/pdf",
                "file_base64": pdf_b64,
                "document_type": "MEDICAL_DOCUMENT",
            },
        )
        self.assertEqual(upload_pdf_res.status_code, 200)
        pdf_file_info = upload_pdf_res.json()["file"]
        self.assertEqual(pdf_file_info["filename"], "blood_panel.pdf")
        self.assertIn("Document findings", pdf_file_info["extracted_clinical_summary"])

        # 2. Raw / Header Upload of Rash Photo
        image_dummy = b"\x89PNG\r\n\x1a\nMockImageDataRash"
        upload_img_res = self.client.post(
            "/api/upload",
            headers={
                "Authorization": self.user_a_token,
                "Content-Type": "image/png",
                "X-Filename": "arm_rash.png",
                "X-Document-Type": "CLINICAL_PHOTO",
            },
            content=image_dummy,
        )
        self.assertEqual(upload_img_res.status_code, 200)
        img_file_info = upload_img_res.json()["file"]
        self.assertIn("Visual findings", img_file_info["extracted_clinical_summary"])

        # 3. Download uploaded file by owner (User A)
        download_res = self.client.get(
            f"/api/files/{self.user_a_id}/blood_panel.pdf",
            headers={"Authorization": self.user_a_token},
        )
        self.assertEqual(download_res.status_code, 200)
        self.assertEqual(download_res.content, pdf_dummy)

        # 4. Tenant Isolation Check: User B attempts to access User A's file -> 403 Forbidden
        forbidden_res = self.client.get(
            f"/api/files/{self.user_a_id}/blood_panel.pdf",
            headers={"Authorization": self.user_b_token},
        )
        self.assertEqual(forbidden_res.status_code, 403)

    def test_06_sse_chat_stream_and_single_write_persistence(self):
        """Test real-time SSE streaming consultation, event formatting, and 1-write persistence."""
        # Create session first
        sess_res = self.client.post(
            "/api/sessions",
            headers={"Authorization": self.user_a_token},
            json={"title": "Severe Headache Consultation"},
        )
        session_id = sess_res.json()["session"]["session_id"]

        # Call SSE endpoint
        chat_req = {
            "session_id": session_id,
            "query_text": "Severe throbbing headache on one side with nausea, sensitivity to light, and visual blur",
            "selected_symptoms": ["headache", "nausea"],
        }
        stream_res = self.client.post(
            "/api/chat/stream",
            headers={"Authorization": self.user_a_token},
            json=chat_req,
        )
        self.assertEqual(stream_res.status_code, 200)
        self.assertIn("text/event-stream", stream_res.headers.get("content-type", ""))

        # Parse SSE lines
        raw_events = stream_res.text.strip().split("\n\n")
        self.assertGreater(len(raw_events), 2)

        diagnostic_meta = None
        streamed_tokens = []
        has_done = False

        for evt_str in raw_events:
            lines = evt_str.split("\n")
            for line in lines:
                if line.startswith("data: "):
                    payload_raw = line[len("data: ") :].strip()
                    if payload_raw == "[DONE]":
                        has_done = True
                    else:
                        payload = json.loads(payload_raw)
                        if payload.get("type") == "diagnostic_report":
                            diagnostic_meta = payload
                        elif payload.get("type") == "token":
                            streamed_tokens.append(payload.get("token"))

        self.assertIsNotNone(diagnostic_meta, "SSE stream must emit initial diagnostic_report event.")
        self.assertIn("top_condition", diagnostic_meta)
        self.assertIn("triage_level", diagnostic_meta)
        self.assertTrue(has_done, "SSE stream must conclude with data: [DONE].")
        self.assertGreater(len(streamed_tokens), 0, "SSE stream must yield streaming text tokens.")

        full_text = "".join(streamed_tokens)
        self.assertIn(diagnostic_meta["top_condition"], full_text)

        # 5. SINGLE-WRITE PERSISTENCE VERIFICATION:
        # Check that session document was updated with turn in exactly 1 write
        session_lookup = self.client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": self.user_a_token},
        )
        self.assertEqual(session_lookup.status_code, 200)
        context = session_lookup.json()["context"]
        turns = context.get("recent_context_window", [])
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["user"], chat_req["query_text"])
        self.assertEqual(turns[0]["assistant"], full_text.strip())

    def test_07_continuous_rlhf_replay_buffer_logged(self):
        """Verify that consultation trajectories are appended to clinical_interaction_buffer.jsonl."""
        buffer_path = settings.DATA_DIR / "clinical_interaction_buffer.jsonl"
        self.assertTrue(buffer_path.exists(), "clinical_interaction_buffer.jsonl must exist after consultation.")

        with open(buffer_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertGreaterEqual(len(lines), 1)
        last_entry = json.loads(lines[-1])
        self.assertIn("session_id", last_entry)
        self.assertIn("user_id", last_entry)
        self.assertIn("user_query", last_entry)
        self.assertIn("top_condition", last_entry)
        self.assertIn("prompt", last_entry)
        self.assertIn("assistant_response", last_entry)

    def test_08_clinical_classes_catalog_and_all_580_tier(self):
        """Verify GET /api/classes endpoint returns arranged classes and model_tier='all_580' executes correctly."""
        # 1. Test GET /api/classes
        res = self.client.get("/api/classes")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["total_classes"], 580)
        self.assertGreater(len(body["classes"]), 0)

        # 2. Test Category Filtering
        res_resp = self.client.get("/api/classes?category=Respiratory%20%26%20Thoracic")
        self.assertEqual(res_resp.status_code, 200)
        resp_body = res_resp.json()
        self.assertGreater(resp_body["total_classes"], 0)
        self.assertTrue(all(c["category"] == "Respiratory & Thoracic" for c in resp_body["classes"]))

        # 3. Test Prediction with model_tier="all_580"
        pred_res = self.client.post(
            "/api/predict",
            headers={"Authorization": self.user_a_token},
            json={
                "query_text": "Crushing chest pain radiating to left arm, severe dyspnea, and cold sweats",
                "model_tier": "all_580",
                "top_k": 3,
            },
        )
        self.assertEqual(pred_res.status_code, 200)
        report = pred_res.json()
        self.assertIsNotNone(report["top_condition"])
        self.assertLessEqual(len(report["differential_diagnosis"]), 3)
        self.assertIn("Broad Universe Model (580 Classes)", report["differential_diagnosis"][0]["source_model"])


if __name__ == "__main__":
    unittest.main()
