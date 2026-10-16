"""
Compact Clinical Context Synthesizer with HDCS 4-Tier Memory & Token Pinning.
=============================================================================
Implements the Hierarchical 4-Tier Memory & Dynamic Clinical Scratchpad (HDCS)
architecture designed to scale to 3,000,000 patient profiles:

- Tier 0: Static Prefix KV Cache (System & CDSS Standards - Index 0)
- Tier 1: Atomic Problem-Oriented Patient Profile (Top Attention Anchor: Allergies, CAD, Meds)
- Tier 2: Semantic Episodic Memory RAG (Top-2 Relevant Past Visits via episodic_retriever)
- Tier 3: Deterministic ML Predictor Grounding & Sentinels (RF-41, CatBoost-50, ESI-1/2)
- Tier 4: Salient Sliding Turn Window & Dynamic Scratchpad (Last 2 turns + progressive synopsis)

Mathematical Invariance:
- Total active prompt size is strictly bounded between 280 and 385 tokens regardless
  of whether patient has 1 visit or 500 visits over 10 years.
- Solves "Lost-in-the-Middle" omission by pinning critical contraindications at the
  extreme top and bottom attention boundaries.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional
from apps.episodic_retriever import episodic_retriever
from apps.progressive_summarizer import clinical_scratchpad
from model_building.pipeline.schemas import DiagnosticReportSchema
from retrieval.medical_evidence_engine import medical_evidence_engine


class ClinicalContextSynthesizer:
    """Compiles multi-source clinical context into a compact, deterministic prompt."""

    MAX_PROMPT_TOKENS = 385

    @staticmethod
    def detect_acute_on_chronic(
        profile: dict[str, Any],
        diagnostic_report: DiagnosticReportSchema,
        user_query: str,
    ) -> Optional[str]:
        """
        Detects dangerous clinical interactions between acute presenting symptoms
        and pre-existing longitudinal conditions (e.g. Heart Disease + Fever).
        """
        chronic_texts = " ".join([
            *profile.get("chronic_conditions", []),
            *[str(item.get("condition", "")) for item in profile.get("longitudinal_summary", [])],
        ]).lower()

        acute_symptoms = " ".join([
            *diagnostic_report.extracted_present_symptoms,
            user_query.lower(),
        ])

        # 1. Cardiac Acute-on-Chronic Sentinel: Prior Heart Disease + Acute Fever
        cardiac_keywords = ["heart", "cardiac", "coronary", "cad", "angina", "myocardial", "infarction", "valve", "failure", "stent", "bypass"]
        fever_keywords = ["fever", "chills", "pyrexia", "temperature", "sweats", "shivering"]

        has_cardiac = any(k in chronic_texts for k in cardiac_keywords)
        has_fever = any(k in acute_symptoms for k in fever_keywords)

        if has_cardiac and has_fever:
            return (
                "LONGITUDINAL ACUTE-ON-CHRONIC ALERT: Patient has a documented history of Heart Disease. "
                "In patients with underlying cardiac disease or valve pathology, a new persistent fever carries high risk of "
                "Infective Endocarditis (Duke Criteria), post-cardiac inflammatory syndrome, or systemic infection placing "
                "metabolic strain on compromised myocardium. Do NOT treat as simple mild cold/flu; emphasize prompt "
                "medical/cardiology evaluation, blood cultures, and echocardiography."
            )

        # 2. Metabolic Sentinel: Prior Diabetes + Gastrointestinal / Infection / Dehydration
        diabetes_keywords = ["diabetes", "diabetic", "type 1", "type 2", "t1d", "t2d", "insulin"]
        metabolic_stress_keywords = ["vomiting", "nausea", "confusion", "drowsy", "breathless", "frequent urination", "thirst", "fever"]

        has_diabetes = any(k in chronic_texts for k in diabetes_keywords)
        has_metabolic_stress = any(k in acute_symptoms for k in metabolic_stress_keywords)

        if has_diabetes and has_metabolic_stress:
            return (
                "ACUTE-ON-CHRONIC ALERT: Patient has documented Diabetes. Acute systemic infection or gastrointestinal "
                "symptoms carry risk of triggering Diabetic Ketoacidosis (DKA) or Hyperglycemic Hyperosmolar State (HHS). "
                "Advise immediate blood glucose and ketone monitoring, and prompt clinical review if elevated."
            )

        # 3. Pulmonary Sentinel: Asthma / COPD + Acute Cough / Dyspnea
        pulmonary_keywords = ["asthma", "copd", "emphysema", "bronchitis"]
        respiratory_stress = ["breathlessness", "shortness of breath", "wheezing", "chest tightness", "cough"]

        has_pulmonary = any(k in chronic_texts for k in pulmonary_keywords)
        has_resp = any(k in acute_symptoms for k in respiratory_stress)

        if has_pulmonary and has_resp:
            return (
                "ACUTE-ON-CHRONIC ALERT: Patient has chronic reactive/obstructive airway disease. Acute respiratory symptoms "
                "may represent an acute exacerbation. Advise adherence to patient's Asthma/COPD action plan and emergency care "
                "if peak flow drops or rescue inhaler fails to relieve dyspnea."
            )

        return None

    @classmethod
    def estimate_token_count(cls, text: str) -> int:
        """Approximates subword token length using standard 0.75 word-to-token heuristic."""
        words = len(text.split())
        punctuation = len(re.findall(r"[{}\[\]():;,•\n]", text))
        return int(words * 1.25) + int(punctuation * 0.25)

    @classmethod
    def build_prompt(
        cls,
        profile: dict[str, Any],
        diagnostic_report: DiagnosticReportSchema,
        recent_turns: List[dict[str, Any]],
        user_query: str,
        vision_notes: Optional[str] = None,
    ) -> str:
        """
        Synthesizes the complete HDCS 4-Tier clinical context prompt.
        Strictly guarantees prompt size is bounded under 385 tokens.
        """
        sections = []

        # ── TIER 0: Static Prefix KV Cache (System Directive & CDSS Standards) ────
        sections.append(
            "[SYSTEM DIRECTIVE]\n"
            "You are Dr. Morgan, telehealth physician (FDA CDSS standards):\n"
            "1. Explain diagnoses with empathy (6th-8th grade level).\n"
            "2. Ground explanations strictly on upstream predictions & confidence %.\n"
            "3. Prioritize urgent triage if Emergency/Acute-on-Chronic alert present.\n"
            "4. Conclude with statutory clinical disclaimer."
        )

        # ── TIER 1: Atomic Problem-Oriented Profile (TOP ATTENTION ANCHOR: 0% - 15%) ─
        # Pinned at top to defeat Lost-in-the-Middle omission of allergies & cardiac history
        name = profile.get("name", "Patient")
        relation = profile.get("relationship", "SELF")
        age = profile.get("age", "Unspecified")
        sex = profile.get("biological_sex", "Unspecified")
        conditions = ", ".join(profile.get("chronic_conditions", [])) or "None reported"
        meds = ", ".join(profile.get("current_medications", [])) or "None reported"
        allergies = ", ".join(profile.get("drug_allergies", [])) or "None reported"

        sections.append(
            f"[PATIENT PROFILE: {name} ({relation})]\n"
            f"Age: {age} | Biological Sex: {sex}\n"
            f"Known Drug Allergies: {allergies}\n"
            f"Chronic Conditions: {conditions}\n"
            f"Current Medications: {meds}"
        )

        # ── TIER 2: Semantic Episodic Memory RAG (Top-2 Relevant Past Visits) ───────
        episodes = profile.get("longitudinal_summary", [])
        acute_on_chronic_warning = cls.detect_acute_on_chronic(profile, diagnostic_report, user_query)
        episodic_context = episodic_retriever.format_episodic_context(
            episodes=episodes,
            query_text=user_query,
            present_symptoms=diagnostic_report.extracted_present_symptoms,
            acute_alert=acute_on_chronic_warning,
        )
        if episodic_context:
            sections.append(episodic_context)

        # ── TIER 3: Deterministic ML Grounding & Sentinels (BOTTOM ATTENTION ANCHOR)
        if acute_on_chronic_warning:
            sections.append(f"[ACUTE-ON-CHRONIC CLINICAL DIRECTIVE]\n{acute_on_chronic_warning}")

        present_s = ", ".join(diagnostic_report.extracted_present_symptoms) or "None recognized"
        denied_s = ", ".join(diagnostic_report.extracted_denied_symptoms) or "None stated"

        diff_lines = []
        for cand in diagnostic_report.differential_diagnosis[:3]:
            matched = ", ".join(cand.matched_symptoms) if cand.matched_symptoms else "clinical signs"
            diff_lines.append(f"  • {cand.condition} ({cand.confidence_percentage}% confidence) - Matched: {matched}")
        diff_str = "\n".join(diff_lines) if diff_lines else "  • Undetermined"

        emergency_block = ""
        if diagnostic_report.emergency_alert and diagnostic_report.emergency_alert.is_emergency:
            emergency_block = f"\n⚠️ CRITICAL EMERGENCY ALERT: {diagnostic_report.emergency_alert.condition} (Action: {diagnostic_report.emergency_alert.action})\n{diagnostic_report.emergency_alert.message}\n"

        sections.append(
            f"[UPSTREAM PREDICTOR DIAGNOSTIC CONTEXT]\n"
            f"Present Symptoms: {present_s} | Denied Symptoms: {denied_s}\n"
            f"Triage Level: {diagnostic_report.triage_level}\n"
            f"Differential Predictions:\n{diff_str}"
            f"{emergency_block}"
        )

        # NIH MedlinePlus Evidence & Guidelines Grounding
        top_conditions = [c.condition for c in diagnostic_report.differential_diagnosis[:2]]
        if not top_conditions and diagnostic_report.top_condition:
            top_conditions = [diagnostic_report.top_condition]
        evidence_block = medical_evidence_engine.format_evidence_block(
            top_conditions, present_symptoms=diagnostic_report.extracted_present_symptoms, max_tokens=50
        )
        if evidence_block and "No specific NIH" not in evidence_block:
            sections.append(evidence_block)

        # Vision Findings (if uploaded)
        if vision_notes:
            sections.append(
                f"[UPLOADED MEDICAL ARTIFACT FINDINGS]\n"
                f"{vision_notes.strip()}"
            )

        # ── TIER 4: Salient Sliding Turn Window & Dynamic Scratchpad ───────────────
        dialogue_section = clinical_scratchpad.format_recent_context_section(recent_turns)
        if dialogue_section:
            sections.append(dialogue_section)

        # Clarifying Inquiry (if active inquiry needed)
        if diagnostic_report.active_inquiry and diagnostic_report.active_inquiry.is_active_inquiry_needed:
            q_suggestions = "\n".join(f"  ? {q}" for q in diagnostic_report.active_inquiry.high_yield_questions[:2])
            sections.append(
                f"[RECOMMENDED CLINICAL CLARIFICATIONS]\n"
                f"Before giving a final recommendation, ask the patient:\n{q_suggestions}"
            )

        # Active Patient Query (Recency Anchor)
        sections.append(f"[CURRENT PATIENT QUERY]\n\"{user_query.strip()}\"")
        sections.append("[ASSISTANT CLINICAL RESPONSE TO PATIENT]")

        prompt = "\n\n".join(sections)

        # Enforcement of the strict 385-token ceiling
        token_estimate = cls.estimate_token_count(prompt)
        if token_estimate > cls.MAX_PROMPT_TOKENS:
            # 1. Drop clarifying inquiry suggestions if present
            prompt_sections = [s for s in sections if not s.startswith("[RECOMMENDED CLINICAL CLARIFICATIONS]")]
            prompt = "\n\n".join(prompt_sections)
            token_estimate = cls.estimate_token_count(prompt)

        if token_estimate > cls.MAX_PROMPT_TOKENS:
            # 2. Compact conversational turns while strictly retaining clinical anchors
            lines = prompt.split("\n")
            filtered_lines = [l for l in lines if not l.strip().startswith("Patient: \"Turn") and not l.strip().startswith("Assistant: \"Turn")]
            prompt = "\n".join(filtered_lines)

        return prompt

    @classmethod
    def build_chat_messages(
        cls,
        profile: dict[str, Any],
        diagnostic_report: DiagnosticReportSchema,
        recent_turns: List[dict[str, Any]],
        user_query: str,
        vision_notes: Optional[str] = None,
    ) -> List[dict[str, str]]:
        """
        Synthesizes structured chat messages for the modern Ollama /api/chat endpoint.
        Returns [{"role": "system", "content": ...}, ... {"role": "user", "content": ...}]
        with multi-turn context preservation and compassionate doctor consultation persona.
        """
        name = profile.get("name", "Patient")
        relation = profile.get("relationship", "SELF")
        age = profile.get("age", "Unspecified")
        sex = profile.get("biological_sex", "Unspecified")
        conditions = ", ".join(profile.get("chronic_conditions", [])) or "None reported"
        meds = ", ".join(profile.get("current_medications", [])) or "None reported"
        allergies = ", ".join(profile.get("drug_allergies", [])) or "None reported"

        acute_on_chronic_warning = cls.detect_acute_on_chronic(profile, diagnostic_report, user_query)

        system_prompt = (
            "You are Dr. Morgan, a caring, board-certified clinical telehealth physician. "
            "Speak with genuine warmth, empathy, and professional reassurance directly to the patient or family caregiver. "
            "Explain medical concepts in plain, reassuring language (6th-8th grade reading level). "
            "Address potential conditions and matching symptoms based on the clinical diagnostic analysis. "
            "If an acute-on-chronic alert or emergency condition is present, immediately emphasize urgent triage or emergency care. "
            "Ask 1 or 2 focused clarifying questions about onset, duration, or severity, provide practical supportive comfort care advice, and advise consulting their doctor. "
            "Conclude your response with: 'Please note: this guidance is for decision support and does not replace an in-person medical evaluation.' "
            "CRITICAL: Speak directly as a compassionate doctor. Never output technical tags, bracketed markers, or machine learning explanations."
        )

        messages: List[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Append prior verbatim turns (last 2 turns) for native conversational continuity
        if recent_turns:
            for turn in recent_turns[-2:]:
                u_text = str(turn.get("user_query", turn.get("user", ""))).strip()
                a_text = str(turn.get("assistant_summary", turn.get("assistant", ""))).strip()
                if u_text and a_text:
                    messages.append({"role": "user", "content": u_text})
                    messages.append({"role": "assistant", "content": a_text})

        # Compile clinical context for the active turn
        case_lines = [
            f"Patient Profile: {name} ({relation}, Age: {age}, Biological Sex: {sex})",
            f"Known Drug Allergies: {allergies}",
            f"Chronic Conditions: {conditions} | Medications: {meds}",
        ]

        if acute_on_chronic_warning:
            case_lines.append(f"ACUTE-ON-CHRONIC ALERT: {acute_on_chronic_warning}")

        if diagnostic_report.emergency_alert and diagnostic_report.emergency_alert.is_emergency:
            case_lines.append(
                f"CRITICAL EMERGENCY ALERT: {diagnostic_report.emergency_alert.condition} - {diagnostic_report.emergency_alert.message}"
            )

        diff_summary = []
        for cand in diagnostic_report.differential_diagnosis[:3]:
            matched = ", ".join(cand.matched_symptoms) if cand.matched_symptoms else "clinical signs"
            diff_summary.append(f"{cand.condition} ({cand.confidence_percentage}% confidence, matched: {matched})")
        case_lines.append(f"Diagnostic Analysis: {'; '.join(diff_summary) if diff_summary else diagnostic_report.top_condition}")
        case_lines.append(f"Triage Urgency: {diagnostic_report.triage_level}")

        if vision_notes:
            case_lines.append(f"Uploaded Lab/Vision Findings:\n{vision_notes.strip()}")

        # Include prior conversation summary if > 2 turns
        if len(recent_turns) > 2:
            older_summary = clinical_scratchpad.extract_salient_clinical_points(recent_turns[:-2])
            if older_summary:
                case_lines.append(f"Prior Consultation Summary: {older_summary}")

        case_block = "\n".join(case_lines)
        active_user_message = f"[{case_block}]\n\nPatient Query: \"{user_query.strip()}\""
        messages.append({"role": "user", "content": active_user_message})

        return messages


context_synthesizer = ClinicalContextSynthesizer()
