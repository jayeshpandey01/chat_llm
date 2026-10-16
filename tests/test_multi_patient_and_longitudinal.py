"""
End-to-End Test Suite: Multi-Patient Family Profiles, Strict Role-Wise Isolation,
Longitudinal Disease Memory, and Acute-on-Chronic Clinical Reasoning.
================================================================================
Verifies:
1. Registration and listing of multiple family members (Dad, Mom, Self).
2. Longitudinal condition timeline persistence (Heart Disease in Month 0).
3. Strict role-wise context isolation: Mom's session has ZERO leakage of Dad's cardiac history or meds.
4. Acute-on-chronic clinical reasoning: Dad (Heart Disease) + 1 Month Later "Fever"
   -> Automatically flags Infective Endocarditis / cardiac strain in SSE stream.
5. Single-Read / Single-Write Firestore Spark tier compliance via denormalized session snapshots.
6. Persona-filtered session retrieval (GET /api/sessions?member_id=...).
"""

import json
import unittest
import uuid
from fastapi.testclient import TestClient

from apps.server import create_app


class TestMultiPatientAndLongitudinal(unittest.TestCase):
    """Test suite for role-wise multi-patient management and longitudinal clinical intelligence."""

    dad_id: str = ""
    mom_id: str = ""

    @classmethod
    def setUpClass(cls):
        """Initialize FastAPI app with test client and isolated test user tenant."""
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.user_uid = f"guardian_{uuid.uuid4().hex[:8]}"
        cls.auth_header = {"Authorization": f"Bearer test_{cls.user_uid}"}

    def test_01_create_multiple_family_members(self):
        """Account holder registers Member A (Dad) and Member B (Mom)."""
        # Register Dad (68M, Cardiac History)
        dad_payload = {
            "name": "Dad",
            "relationship": "FATHER",
            "age": 68.0,
            "biological_sex": "MALE",
            "chronic_conditions": ["Hypertension", "Coronary Artery Disease"],
            "current_medications": ["Metoprolol 50mg", "Aspirin 81mg"],
            "drug_allergies": ["Penicillin"],
        }
        res_dad = self.client.post("/api/members", headers=self.auth_header, json=dad_payload)
        self.assertEqual(res_dad.status_code, 200)
        TestMultiPatientAndLongitudinal.dad_id = res_dad.json()["member"]["member_id"]
        self.assertTrue(self.dad_id.startswith("mem_"))
        self.assertEqual(res_dad.json()["member"]["name"], "Dad")

        # Register Mom (64F, Orthopedic History)
        mom_payload = {
            "name": "Mom",
            "relationship": "MOTHER",
            "age": 64.0,
            "biological_sex": "FEMALE",
            "chronic_conditions": ["Osteoarthritis"],
            "current_medications": ["Acetaminophen 500mg"],
            "drug_allergies": ["Sulfa"],
        }
        res_mom = self.client.post("/api/members", headers=self.auth_header, json=mom_payload)
        self.assertEqual(res_mom.status_code, 200)
        TestMultiPatientAndLongitudinal.mom_id = res_mom.json()["member"]["member_id"]
        self.assertTrue(self.mom_id.startswith("mem_"))
        self.assertEqual(res_mom.json()["member"]["name"], "Mom")

        # List members and verify both appear
        list_res = self.client.get("/api/members", headers=self.auth_header)
        self.assertEqual(list_res.status_code, 200)
        member_names = [m["name"] for m in list_res.json()["members"]]
        self.assertIn("Dad", member_names)
        self.assertIn("Mom", member_names)

    def test_02_longitudinal_condition_logging(self):
        """Log a historical clinical milestone (Heart Disease episode in Month 0) for Dad."""
        condition_payload = {
            "condition": "Coronary Artery Disease / Angina",
            "icd10": "I25.1",
            "diagnosed_date": "2026-09-15",
            "status": "CHRONIC_ACTIVE",
            "notes": "Diagnosed in prior cardiology consultation. Prescribed Metoprolol & Aspirin.",
        }
        cond_res = self.client.post(
            f"/api/members/{self.dad_id}/conditions",
            headers=self.auth_header,
            json=condition_payload,
        )
        self.assertEqual(cond_res.status_code, 200)
        updated_dad = cond_res.json()["member"]
        self.assertIn("Coronary Artery Disease / Angina", updated_dad["chronic_conditions"])

        # Check timeline endpoint
        timeline_res = self.client.get(
            f"/api/members/{self.dad_id}/timeline",
            headers=self.auth_header,
        )
        self.assertEqual(timeline_res.status_code, 200)
        timeline = timeline_res.json()["timeline"]
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0]["condition"], "Coronary Artery Disease / Angina")
        self.assertEqual(timeline[0]["diagnosed_date"], "2026-09-15")

    def test_03_strict_role_wise_isolation_mom(self):
        """Consultation for Mom (Person B) must have ZERO leakage of Dad's cardiac history or meds."""
        # 1. Create session for Mom
        create_res = self.client.post(
            "/api/sessions",
            headers=self.auth_header,
            json={"member_id": self.mom_id, "title": "Mom Knee Joint Consultation"},
        )
        self.assertEqual(create_res.status_code, 200)
        mom_session_id = create_res.json()["session"]["session_id"]

        # 2. Single-read check on session context
        session_res = self.client.get(
            f"/api/sessions/{mom_session_id}",
            headers=self.auth_header,
        )
        self.assertEqual(session_res.status_code, 200)
        context = session_res.json()["context"]
        mom_snapshot = context["member_snapshot"]

        self.assertEqual(mom_snapshot["name"], "Mom")
        self.assertEqual(mom_snapshot["relationship"], "MOTHER")
        self.assertEqual(mom_snapshot["biological_sex"], "FEMALE")
        self.assertEqual(mom_snapshot["age"], 64.0)

        # PROOF OF ZERO LEAKAGE: Dad's CAD, nitrates, beta-blockers, and Penicillin allergy must NOT exist
        snapshot_str = json.dumps(mom_snapshot).lower()
        self.assertNotIn("coronary", snapshot_str)
        self.assertNotIn("metoprolol", snapshot_str)
        self.assertNotIn("penicillin", snapshot_str)

        # 3. Stream a chat for Mom
        stream_res = self.client.post(
            "/api/chat/stream",
            headers=self.auth_header,
            json={
                "session_id": mom_session_id,
                "query_text": "Severe joint pain and stiffness in right knee, difficulty walking stairs, denies fever",
                "selected_symptoms": ["joint_pain", "knee_pain"],
            },
        )
        self.assertEqual(stream_res.status_code, 200)

        # Parse SSE
        events = stream_res.text.strip().split("\n\n")
        first_event = json.loads(events[0].replace("data: ", ""))
        self.assertEqual(first_event["type"], "diagnostic_report")
        self.assertEqual(first_event["member_name"], "Mom")
        self.assertFalse(first_event["is_acute_on_chronic"], "Mom does not have cardiac acute-on-chronic alert")
        self.assertIsNone(first_event["acute_on_chronic_alert"])

        full_streamed = stream_res.text
        self.assertNotIn("endocarditis", full_streamed.lower())
        self.assertNotIn("coronary", full_streamed.lower())

    def test_04_acute_on_chronic_reasoning_dad_1_month_later(self):
        """
        THE 1-MONTH-LATER SCENARIO:
        Dad with pre-existing Heart Disease (from Month 0) presents in Month 1 with 'fever'.
        AI must recall Dad's cardiac comorbidity and flag Infective Endocarditis risk!
        """
        # 1. Create session for Dad
        create_res = self.client.post(
            "/api/sessions",
            headers=self.auth_header,
            json={"member_id": self.dad_id, "title": "Dad Month 1 Follow-up: Fever"},
        )
        self.assertEqual(create_res.status_code, 200)
        dad_session_id = create_res.json()["session"]["session_id"]

        # 2. Verify Dad's session snapshot includes the longitudinal Heart Disease
        session_res = self.client.get(
            f"/api/sessions/{dad_session_id}",
            headers=self.auth_header,
        )
        self.assertEqual(session_res.status_code, 200)
        dad_snapshot = session_res.json()["context"]["member_snapshot"]
        self.assertEqual(dad_snapshot["name"], "Dad")
        self.assertEqual(dad_snapshot["relationship"], "FATHER")
        self.assertIn("Coronary Artery Disease / Angina", dad_snapshot["chronic_conditions"])

        # 3. Stream Dad's acute complaint: "i also getting fever now"
        stream_res = self.client.post(
            "/api/chat/stream",
            headers=self.auth_header,
            json={
                "session_id": dad_session_id,
                "query_text": "I also getting fever now with sweating and chills",
                "selected_symptoms": ["fever", "chills"],
            },
        )
        self.assertEqual(stream_res.status_code, 200)

        # 4. Parse SSE events
        raw_events = stream_res.text.strip().split("\n\n")
        meta_event = json.loads(raw_events[0].replace("data: ", ""))

        self.assertEqual(meta_event["type"], "diagnostic_report")
        self.assertEqual(meta_event["member_name"], "Dad")
        self.assertEqual(meta_event["member_relationship"], "FATHER")

        # CRITICAL TEST: Verify Acute-on-Chronic flag is TRUE and contains Endocarditis warning
        self.assertTrue(
            meta_event["is_acute_on_chronic"],
            "AI must detect acute-on-chronic cardiac complication when Dad with Heart Disease presents with fever!",
        )
        self.assertIn("Infective Endocarditis", meta_event["acute_on_chronic_alert"])
        self.assertIn("Heart Disease", meta_event["acute_on_chronic_alert"])

        # Verify streamed response includes the acute-on-chronic alert
        streamed_tokens = []
        for line in stream_res.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                payload = json.loads(line[len("data: "):])
                if payload.get("type") == "token":
                    streamed_tokens.append(payload.get("token"))
        assembled_text = "".join(streamed_tokens)

        self.assertIn("LONGITUDINAL ACUTE-ON-CHRONIC ALERT", assembled_text)
        self.assertIn("Infective Endocarditis", assembled_text)

        # 5. Verify Single-Write persistence on Dad's session
        lookup_res = self.client.get(
            f"/api/sessions/{dad_session_id}",
            headers=self.auth_header,
        )
        self.assertEqual(lookup_res.status_code, 200)
        context = lookup_res.json()["context"]
        turns = context["recent_context_window"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["user_query"], "I also getting fever now with sweating and chills")

    def test_05_persona_filtered_sessions(self):
        """Verify GET /api/sessions?member_id=... returns only sessions for the chosen persona."""
        # Query Dad's sessions
        dad_sessions_res = self.client.get(
            f"/api/sessions?member_id={self.dad_id}",
            headers=self.auth_header,
        )
        self.assertEqual(dad_sessions_res.status_code, 200)
        dad_sessions = dad_sessions_res.json()["sessions"]
        self.assertGreaterEqual(len(dad_sessions), 1)
        for s in dad_sessions:
            self.assertEqual(s["member_id"], self.dad_id)
            self.assertEqual(s["member_name"], "Dad")

        # Query Mom's sessions
        mom_sessions_res = self.client.get(
            f"/api/sessions?member_id={self.mom_id}",
            headers=self.auth_header,
        )
        self.assertEqual(mom_sessions_res.status_code, 200)
        mom_sessions = mom_sessions_res.json()["sessions"]
        self.assertGreaterEqual(len(mom_sessions), 1)
        for s in mom_sessions:
            self.assertEqual(s["member_id"], self.mom_id)
            self.assertEqual(s["member_name"], "Mom")


if __name__ == "__main__":
    unittest.main()
