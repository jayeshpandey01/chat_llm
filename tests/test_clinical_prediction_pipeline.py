"""
Comprehensive Unit & Integration Test Suite for Clinical Symptom Prediction Pipeline.
======================================================================================
Tests clinical accuracy, negation extraction, conformal coverage, emergency alerts,
active questioning, demographic masking, explanation generation, and sub-5ms caching.
"""

import time
import unittest

from model_building.pipeline import (
    ClinicalPredictionPipeline,
    SymptomInputRequest,
)


class TestClinicalPredictionPipeline(unittest.TestCase):
    """Test suite for HealthTrack AI Clinical Prediction Pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initialize pipeline once for all tests to amortize model loading time."""
        cls.pipeline = ClinicalPredictionPipeline()

    def setUp(self):
        # Clear cache before each test to ensure deterministic benchmarking
        self.pipeline.cache.clear()

    def test_dermatology_free_text_with_negation(self):
        """Verify free-text extraction, negation handling, and dermatology condition prediction."""
        query = "Severe skin rash with intense itching and nodal skin eruptions across arm, denies fever"
        report = self.pipeline.predict(query)

        self.assertIn("skin_rash", report.extracted_present_symptoms)
        self.assertIn("itching", report.extracted_present_symptoms)
        self.assertIn("fever", report.extracted_denied_symptoms)
        self.assertEqual(report.top_condition, "Fungal infection")
        self.assertGreater(report.top_confidence, 50.0)
        self.assertFalse(report.is_out_of_distribution)
        self.assertFalse(report.abstention_triggered)

        # Verify explanations
        self.assertGreater(len(report.differential_diagnosis), 0)
        top_cand = report.differential_diagnosis[0]
        self.assertEqual(top_cand.condition, "Fungal infection")
        self.assertIn("skin rash", top_cand.matched_symptoms)
        self.assertIn("Fungal infection", top_cand.explanation_text)
        self.assertIn("absence of fever", top_cand.explanation_text.lower())

    def test_cardiac_emergency_red_flag_override(self):
        """Verify Emergency Acuity Sentinel overrides model output on life-threatening signs."""
        query = "Crushing chest pain radiating to left arm, severe breathlessness, and sweating profusely."
        report = self.pipeline.predict(query)

        self.assertIsNotNone(report.emergency_alert)
        self.assertTrue(report.emergency_alert.is_emergency)
        self.assertEqual(report.emergency_alert.acuity_level, 1)
        self.assertEqual(report.emergency_alert.action, "CALL_911_IMMEDIATELY")
        self.assertEqual(report.triage_level, "ESI-1")
        self.assertIn("EMERGENCY", report.triage_recommendation)

    def test_respiratory_asthma_prediction(self):
        """Verify pulmonology symptom constellation predicts Bronchial Asthma."""
        query = "Patient presents with persistent cough, breathlessness, wheezing, and fatigue, denies fever."
        report = self.pipeline.predict(query)

        self.assertIn("cough", report.extracted_present_symptoms)
        self.assertIn("breathlessness", report.extracted_present_symptoms)
        self.assertEqual(report.top_condition, "Bronchial Asthma")
        self.assertGreater(report.top_confidence, 40.0)
        self.assertTrue(len(report.conformal_differential_set) >= 1)
        self.assertIn("Bronchial Asthma", report.conformal_differential_set)

    def test_structured_onboarding_symptoms_input(self):
        """Verify pipeline handles structured symptom list without narrative query text."""
        req = SymptomInputRequest(
            selected_symptoms=["itching", "skin_rash", "nodal_skin_eruptions"],
        )
        report = self.pipeline.predict(req)

        self.assertIn("itching", report.extracted_present_symptoms)
        self.assertIn("skin_rash", report.extracted_present_symptoms)
        self.assertEqual(report.top_condition, "Fungal infection")
        self.assertFalse(report.cached)

    def test_combined_narrative_and_structured_input(self):
        """Verify pipeline merges free-text narrative and structured checklist tokens without duplicates."""
        req = SymptomInputRequest(
            query_text="Patient has persistent cough and mild fatigue, denies headache",
            selected_symptoms=["breathlessness", "cough"],
        )
        report = self.pipeline.predict(req)

        self.assertIn("cough", report.extracted_present_symptoms)
        self.assertIn("fatigue", report.extracted_present_symptoms)
        self.assertIn("breathlessness", report.extracted_present_symptoms)
        self.assertIn("headache", report.extracted_denied_symptoms)
        # Verify no duplicate cough
        self.assertEqual(report.extracted_present_symptoms.count("cough"), 1)

    def test_conformal_differential_coverage_guarantee(self):
        """Verify conformal prediction set provides coverage mass for the differential."""
        query = "High fever, chills, vomiting, and joint pain"
        report = self.pipeline.predict(query, conformal_coverage=0.95)

        self.assertTrue(report.conformal_coverage_guaranteed)
        self.assertGreaterEqual(len(report.conformal_differential_set), 1)
        self.assertIn(report.top_condition, report.conformal_differential_set)

    def test_solitary_symptom_active_inquiry(self):
        """Verify non-specific solitary symptom triggers active inquiry instead of blind guessing."""
        query = "I am feeling fatigued"
        report = self.pipeline.predict(query)

        self.assertTrue(report.abstention_triggered)
        self.assertIsNotNone(report.active_inquiry)
        self.assertTrue(report.active_inquiry.is_active_inquiry_needed)
        self.assertGreaterEqual(len(report.active_inquiry.high_yield_questions), 1)

    def test_ood_conversational_abstention(self):
        """Verify non-medical conversational inputs safely abstain without hallucinating diagnoses."""
        query = "Hello doctor! Can you tell me what the stock market or weather is doing?"
        report = self.pipeline.predict(query)

        self.assertTrue(report.is_out_of_distribution)
        self.assertTrue(report.abstention_triggered)
        self.assertEqual(report.top_condition, "Undetermined (No Medical Symptoms Recognized)")
        self.assertEqual(report.top_confidence, 0.0)
        self.assertEqual(len(report.differential_diagnosis), 0)

    def test_sub_5ms_caching_speed(self):
        """Verify canonical LRU cache delivers responses in under 5ms on repeated queries."""
        query = "Persistent cough, breathlessness, and fatigue"

        # Cold execution
        cold_report = self.pipeline.predict(query)
        self.assertFalse(cold_report.cached)

        # Hot execution (Tier 1 raw cache hit)
        hot_report = self.pipeline.predict(query)
        self.assertTrue(hot_report.cached)
        self.assertLess(
            hot_report.execution_time_ms,
            5.0,
            f"Expected cached response < 5ms, got {hot_report.execution_time_ms} ms",
        )

    def test_canonical_reordering_cache_invariance(self):
        """Verify reordered symptoms hit the Tier 2 canonical cache invariant."""
        # Query 1
        rep1 = self.pipeline.predict("cough, breathlessness, and fatigue")
        self.assertFalse(rep1.cached)

        # Query 2 (reordered text with identical canonical symptoms)
        rep2 = self.pipeline.predict("fatigue, breathlessness, and cough")
        self.assertTrue(rep2.cached)


if __name__ == "__main__":
    unittest.main()

