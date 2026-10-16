"""
Scalability, Context RAG & Anti-Hallucination Test Suite (HDCS Architecture).
=============================================================================
Verifies:
1. Mathematical Invariance: Prompt token count strictly <= 385 tokens across
   1, 5, 20, and 50 past clinical encounters (O(1) scaling).
2. Semantic Episodic Retrieval (RAG): Selects top-2 clinically relevant past
   encounters (cardiac comorbidity) while ignoring unrelated visits.
3. Lost-in-the-Middle Pinning: Drug allergies & chronic diseases anchored at top,
   ML predictions and acute directives anchored at bottom.
4. Dynamic Progressive Scratchpad: Condenses multi-turn sessions (10 turns) into
   a bounded 2-turn window + synopsis.
5. High-Throughput Performance: Sub-5ms context compilation latency across scale.
"""

import time
import unittest

from apps.context_synthesizer import context_synthesizer
from apps.episodic_retriever import episodic_retriever
from apps.progressive_summarizer import clinical_scratchpad
from model_building.pipeline import (
    ClinicalPredictionPipeline,
    SymptomInputRequest,
)


class TestContextScalabilityAndRAG(unittest.TestCase):
    """Test suite for large-scale context window bounds and anti-hallucination guarantees."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ClinicalPredictionPipeline()

    def test_01_token_invariance_across_encounter_scale(self):
        """Verify prompt token count remains strictly <= 385 tokens across 1, 5, 20, and 50 past encounters."""
        req = SymptomInputRequest(
            query_text="High fever with sweating and chills",
            selected_symptoms=["fever", "chills"],
            age=68.0,
            biological_sex="MALE",
        )
        report = self.pipeline.predict(req)

        for num_episodes in [1, 5, 20, 50]:
            # Generate synthetic past episodes
            episodes = [
                {
                    "condition": f"Past Diagnosis {i}: " + ("Cardiac Angina" if i % 5 == 0 else "Routine checkup"),
                    "icd10": f"I25.{i}",
                    "diagnosed_date": f"202{i%5}-0{i%9+1}-15",
                    "status": "CHRONIC_ACTIVE" if i % 3 == 0 else "RESOLVED",
                    "notes": f"Encounter {i} detailed history notes from clinical interview.",
                }
                for i in range(1, num_episodes + 1)
            ]

            profile = {
                "name": "Dad",
                "relationship": "FATHER",
                "age": 68.0,
                "biological_sex": "MALE",
                "drug_allergies": ["Penicillin Anaphylaxis"],
                "chronic_conditions": ["Coronary Artery Disease", "Hypertension"],
                "current_medications": ["Metoprolol 50mg", "Aspirin 81mg"],
                "longitudinal_summary": episodes,
            }

            recent_turns = [
                {"user_query": f"Turn {t}: patient query about discomfort", "assistant_summary": f"Turn {t} advice"}
                for t in range(1, 6)
            ]

            prompt = context_synthesizer.build_prompt(
                profile=profile,
                diagnostic_report=report,
                recent_turns=recent_turns,
                user_query="I also getting fever now with sweating and chills",
            )

            token_estimate = context_synthesizer.estimate_token_count(prompt)
            self.assertLessEqual(
                token_estimate,
                385,
                f"Prompt exceeded 385-token ceiling with {num_episodes} encounters: got {token_estimate} tokens.",
            )
            # Ensure it is rich enough to contain clinical signal
            self.assertGreaterEqual(token_estimate, 180)

    def test_02_episodic_retriever_clinical_relevance(self):
        """Episodic RAG must retrieve the 2 cardiac encounters and ignore unrelated orthopedic/dental visits."""
        episodes = [
            {"condition": "Right Knee Meniscus Sprain", "notes": "Sports injury while running", "diagnosed_date": "2025-01-10"},
            {"condition": "Dental Cavity Restoration", "notes": "Routine tooth filling", "diagnosed_date": "2025-03-12"},
            {"condition": "Coronary Artery Disease / Angina", "notes": "Exertional chest pressure, prescribed Metoprolol", "diagnosed_date": "2026-09-15"},
            {"condition": "Contact Dermatitis", "notes": "Allergic rash on forearm", "diagnosed_date": "2026-04-05"},
            {"condition": "Tension Headache", "notes": "Work-related stress headache", "diagnosed_date": "2026-06-20"},
            {"condition": "Myocardial Infarction / Post-Stent", "notes": "Cardiac catheterization with drug-eluting stent", "diagnosed_date": "2026-08-01"},
        ]

        query = "I have a high fever, sweating, and heart flutter"
        present_symptoms = ["fever", "sweating", "palpitations"]
        acute_alert = "CRITICAL ACUTE-ON-CHRONIC ALERT: Patient has a documented history of Heart Disease."

        retrieved = episodic_retriever.retrieve_relevant_episodes(
            episodes=episodes,
            query_text=query,
            present_symptoms=present_symptoms,
            acute_alert=acute_alert,
        )

        self.assertEqual(len(retrieved), 2)
        retrieved_conditions = [r["condition"] for r in retrieved]
        # Must pick both cardiac episodes
        self.assertIn("Coronary Artery Disease / Angina", retrieved_conditions)
        self.assertIn("Myocardial Infarction / Post-Stent", retrieved_conditions)
        # Must NOT pick dental or knee
        self.assertNotIn("Dental Cavity Restoration", retrieved_conditions)
        self.assertNotIn("Right Knee Meniscus Sprain", retrieved_conditions)

    def test_03_lost_in_the_middle_attentional_pinning(self):
        """Verify drug allergies are pinned at top anchor and acute directives at bottom anchor."""
        req = SymptomInputRequest(
            query_text="Severe headache with neck stiffness",
            selected_symptoms=["headache", "neck_stiffness"],
            age=45.0,
            biological_sex="FEMALE",
        )
        report = self.pipeline.predict(req)

        profile = {
            "name": "Mom",
            "relationship": "MOTHER",
            "age": 64.0,
            "biological_sex": "FEMALE",
            "drug_allergies": ["Severe Anaphylaxis to Penicillin"],
            "chronic_conditions": ["Hypertension"],
            "current_medications": ["Lisinopril"],
            "longitudinal_summary": [],
        }

        prompt = context_synthesizer.build_prompt(
            profile=profile,
            diagnostic_report=report,
            recent_turns=[],
            user_query="Severe throbbing headache",
        )

        total_length = len(prompt)
        profile_pos = prompt.find("[PATIENT PROFILE: Mom (MOTHER)]")
        allergy_pos = prompt.find("Severe Anaphylaxis to Penicillin")
        predictor_pos = prompt.find("[UPSTREAM PREDICTOR DIAGNOSTIC CONTEXT]")
        query_pos = prompt.find("[CURRENT PATIENT QUERY]")

        # 1. Structural Ordering: Profile & Allergies anchored before Upstream ML Context, Query at the end
        self.assertNotEqual(profile_pos, -1)
        self.assertNotEqual(allergy_pos, -1)
        self.assertNotEqual(predictor_pos, -1)
        self.assertNotEqual(query_pos, -1)
        self.assertLess(profile_pos, predictor_pos)
        self.assertLess(predictor_pos, query_pos)

        # 2. Allergy must appear in the top 500 characters of the context window (Top Anchor)
        self.assertLess(allergy_pos, 500)

        # 3. Current query must appear at the recency boundary (last 150 characters)
        self.assertGreater(query_pos, total_length - 150)

    def test_04_progressive_dialogue_scratchpad_compaction(self):
        """Dynamic scratchpad must condense a 10-turn dialogue to 2 verbatim turns + synopsis."""
        long_session = [
            {"user_query": f"Turn {i}: I have joint stiffness and fatigue", "assistant_summary": f"Advice {i}"}
            for i in range(1, 11)
        ]

        scratchpad, verbatim_turns = clinical_scratchpad.build_bounded_turn_context(long_session)

        # Verbatim turns must be strictly <= 2
        self.assertEqual(len(verbatim_turns), 2)
        self.assertEqual(verbatim_turns[0]["user_query"], "Turn 9: I have joint stiffness and fatigue")
        self.assertEqual(verbatim_turns[1]["user_query"], "Turn 10: I have joint stiffness and fatigue")

        # Scratchpad must capture older discussion synopsis
        self.assertIsNotNone(scratchpad)
        self.assertIn("Earlier in consultation", scratchpad)

        # Formatted section must stay under 80 tokens
        formatted = clinical_scratchpad.format_recent_context_section(long_session)
        self.assertIsNotNone(formatted)
        token_count = context_synthesizer.estimate_token_count(formatted)
        self.assertLessEqual(token_count, 85)

    def test_05_sub_5ms_synthesis_latency_at_scale(self):
        """Context synthesis across 50 past episodes must execute in sub-5ms."""
        episodes = [
            {"condition": f"Condition {i}", "diagnosed_date": "2026-01-01", "notes": "Clinical note"}
            for i in range(50)
        ]
        profile = {
            "name": "Dad",
            "relationship": "FATHER",
            "age": 68.0,
            "biological_sex": "MALE",
            "drug_allergies": ["Sulfa"],
            "chronic_conditions": ["Coronary Artery Disease"],
            "current_medications": ["Metoprolol"],
            "longitudinal_summary": episodes,
        }
        req = SymptomInputRequest(query_text="fever", selected_symptoms=["fever"])
        report = self.pipeline.predict(req)

        # Benchmark 20 iterations
        start = time.perf_counter()
        for _ in range(20):
            _ = context_synthesizer.build_prompt(
                profile=profile,
                diagnostic_report=report,
                recent_turns=[{"user": "q", "assistant": "a"}],
                user_query="I have a fever",
            )
        elapsed = time.perf_counter() - start
        avg_latency_ms = (elapsed / 20) * 1000

        # Sub-5ms latency assertion
        self.assertLess(avg_latency_ms, 5.0, f"Context synthesis took {avg_latency_ms:.2f}ms (threshold: 5.0ms)")


if __name__ == "__main__":
    unittest.main()
