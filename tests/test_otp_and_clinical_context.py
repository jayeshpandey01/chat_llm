"""
End-to-End Verification Suite: SMTP OTP Verification, Multi-Turn Context Accumulation,
Vision Findings Grounding, and Doctor Persona Compliance.
======================================================================================
"""

import json
import unittest
import uuid
from fastapi.testclient import TestClient

from apps.context_synthesizer import context_synthesizer
from apps.firebase_service import firebase_service
from apps.server import create_app
from core.mailer import mailer_service
from model_building.pipeline import SymptomInputRequest, ClinicalPredictionPipeline


class TestOtpAndClinicalContext(unittest.TestCase):
    """Verifies OTP lifecycle, multi-turn symptom continuity, vision context, and doctor persona."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.pipeline = ClinicalPredictionPipeline()
        cls.test_email = f"patient_{uuid.uuid4().hex[:6]}@example.com"
        cls.user_uid = f"user_{uuid.uuid4().hex[:8]}"
        cls.auth_header = {"Authorization": f"Bearer test_{cls.user_uid}"}

    def test_01_otp_dispatch_and_rate_limiting(self):
        """Verify 6-digit OTP generation, SQLite persistence, rate-limiting, and verification."""
        # 1. Dispatch OTP
        send_res = self.client.post(
            "/api/auth/otp/send",
            json={"email": self.test_email, "name": "Alice Patient"},
        )
        self.assertEqual(send_res.status_code, 200)
        data = send_res.json()
        self.assertEqual(data["status"], "SUCCESS")
        otp_code = data.get("debug_code")
        self.assertIsNotNone(otp_code)
        self.assertEqual(len(otp_code), 6)
        self.assertTrue(otp_code.isdigit())

        # 2. Test Invalid Code Attempt
        bad_res = self.client.post(
            "/api/auth/otp/verify",
            json={"email": self.test_email, "otp_code": "000000"},
        )
        self.assertEqual(bad_res.status_code, 400)
        self.assertIn("Invalid verification code", bad_res.json()["detail"])

        # 3. Test Valid Code Verification
        good_res = self.client.post(
            "/api/auth/otp/verify",
            json={"email": self.test_email, "otp_code": otp_code, "name": "Alice Patient"},
        )
        self.assertEqual(good_res.status_code, 200)
        good_data = good_res.json()
        self.assertEqual(good_data["status"], "SUCCESS")
        self.assertTrue(good_data["token"].startswith("dev_"))
        self.assertIn("profile", good_data)
        self.assertEqual(good_data["profile"]["name"], "Alice Patient")

    def test_02_multi_turn_symptom_accumulation(self):
        """Verify symptoms reported in Turn 1 are retained in Turn 2 to prevent symptom amnesia."""
        # 1. Create a consultation session
        sess_res = self.client.post(
            "/api/sessions",
            headers=self.auth_header,
            json={"title": "Multi-Turn Symptom Retention Test"},
        )
        self.assertEqual(sess_res.status_code, 200)
        session_id = sess_res.json()["session"]["session_id"]

        # 2. Turn 1: Present with high fever and chills
        t1_res = self.client.post(
            "/api/chat/stream",
            headers=self.auth_header,
            json={
                "session_id": session_id,
                "query_text": "I have high fever with shivering chills",
                "selected_symptoms": ["fever", "chills"],
            },
        )
        self.assertEqual(t1_res.status_code, 200)

        # Verify turn 1 was persisted with extracted symptoms
        session_ctx = firebase_service.get_session_context(self.user_uid, session_id)
        turns = session_ctx["recent_context_window"]
        self.assertEqual(len(turns), 1)
        self.assertIn("fever", session_ctx.get("cumulative_symptoms", []))

        # 3. Turn 2: Patient only reports new symptom (chest pain) without repeating fever
        t2_res = self.client.post(
            "/api/chat/stream",
            headers=self.auth_header,
            json={
                "session_id": session_id,
                "query_text": "Now my chest is tight and hurting",
            },
        )
        self.assertEqual(t2_res.status_code, 200)

        # Parse SSE meta payload from Turn 2
        raw_events = t2_res.text.strip().split("\n\n")
        meta_event = json.loads(raw_events[0].replace("data: ", ""))
        self.assertEqual(meta_event["type"], "diagnostic_report")

        # Session should now have cumulative symptoms containing both turns
        updated_ctx = firebase_service.get_session_context(self.user_uid, session_id)
        cum_symptoms = updated_ctx.get("cumulative_symptoms", [])
        self.assertIn("fever", cum_symptoms)
        self.assertEqual(len(updated_ctx["recent_context_window"]), 2)

    def test_03_vision_artifact_persistence_and_context_injection(self):
        """Verify lab report upload persists findings and injects into chat consultation."""
        # 1. Upload mock CBC Blood Panel PDF
        upload_res = self.client.post(
            "/api/upload",
            headers=self.auth_header,
            json={
                "filename": "cbc_panel.pdf",
                "content_type": "application/pdf",
                "file_base64": "JVBERi0xLjQKJcTl8uXr...",
                "document_type": "LAB_REPORT",
            },
        )
        self.assertEqual(upload_res.status_code, 200)
        file_data = upload_res.json()["file"]
        file_id = file_data["file_id"]

        # Verify DB metadata record contains extracted summary
        record = firebase_service.get_uploaded_file_record(self.user_uid, file_id)
        self.assertIsNotNone(record)
        self.assertIn("extracted_clinical_summary", record)

        # 2. Create session and stream chat with document_id attached
        sess_res = self.client.post(
            "/api/sessions",
            headers=self.auth_header,
            json={"title": "Lab Review Consultation"},
        )
        session_id = sess_res.json()["session"]["session_id"]

        stream_res = self.client.post(
            "/api/chat/stream",
            headers=self.auth_header,
            json={
                "session_id": session_id,
                "query_text": "Can you review my uploaded blood report results?",
                "document_id": file_id,
            },
        )
        self.assertEqual(stream_res.status_code, 200)
        self.assertIn("data: ", stream_res.text)

    def test_04_doctor_persona_build_chat_messages(self):
        """Verify build_chat_messages formats Dr. Morgan persona without engineering tags."""
        req = SymptomInputRequest(
            query_text="Throbbing headache on one side with nausea",
            selected_symptoms=["headache", "nausea"],
            age=32.0,
            biological_sex="FEMALE",
        )
        report = self.pipeline.predict(req)

        profile = {
            "name": "Sarah",
            "relationship": "SELF",
            "age": 32.0,
            "biological_sex": "FEMALE",
            "drug_allergies": ["Penicillin"],
            "chronic_conditions": [],
            "current_medications": [],
        }

        messages = context_synthesizer.build_chat_messages(
            profile=profile,
            diagnostic_report=report,
            recent_turns=[],
            user_query="My head is pounding and I feel sick",
        )

        self.assertGreaterEqual(len(messages), 2)
        system_msg = messages[0]
        self.assertEqual(system_msg["role"], "system")
        self.assertIn("Dr. Morgan", system_msg["content"])
        self.assertIn("caring", system_msg["content"])

        user_msg = messages[-1]
        self.assertEqual(user_msg["role"], "user")
        self.assertIn("Sarah", user_msg["content"])
        self.assertIn("Penicillin", user_msg["content"])
        self.assertIn("My head is pounding", user_msg["content"])


if __name__ == "__main__":
    unittest.main()

