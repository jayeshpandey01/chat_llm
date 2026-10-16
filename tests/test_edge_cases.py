"""
Comprehensive Clinical ML Edge-Case Test Suite
==============================================
Validates the 10 critical clinical edge-case safeguards:
- EC-01: Red-flag triage overrides & asymmetric risk handling (ESI Levels 1 & 2)
- EC-02: Solitary non-specific symptom active questioning & Bayesian inquiry
- EC-03: Conformal differential prediction sets (guaranteed statistical coverage)
- EC-07: Biological and demographic constraint post-masking (age & biological sex)
- EC-09: Out-of-distribution (OOD) and non-medical query rejection
- E2E: Unified Multi-Model Ensemble Router & Clinical Diagnostic Engine integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_building.data.demographic_rules import DemographicFilter
from model_building.models.active_questioning import ActiveQuestioningEngine
from model_building.models.acuity_sentinel import ClinicalAcuitySentinel
from model_building.models.conformal_predictor import ConformalDifferentialPredictor
from model_building.models.diagnostic_engine import ClinicalDiagnosticEngine
from model_building.models.ensemble_router import MultiModelEnsembleRouter


@pytest.fixture(scope="session")
def project_paths():
    base = Path(__file__).resolve().parent.parent
    return {
        "b_model": base / "models" / "model_benchmark_41.joblib",
        "g_model": base / "models" / "model_general_clinical.joblib",
        "resmlp_model": base / "models" / "pytorch_resmlp.pt",
        "vocab": base / "data" / "processed" / "vocabularies.json",
        "db": base / "data" / "processed" / "clinical_evidence.db",
    }


# =========================================================================
# EC-01: Red-Flag Emergency Acuity Sentinel Tests
# =========================================================================

def test_ec01_acute_cardiac_triad():
    sentinel = ClinicalAcuitySentinel()
    # Chest pain + breathlessness + sweating
    alert = sentinel.evaluate_triage(
        differential=[("Gastroesophageal Reflux Disease (Gerd)", 0.85), ("Heart attack", 0.05)],
        observed_symptoms=["chest_pain", "breathlessness", "sweating"],
    )
    assert alert.is_emergency is True
    assert alert.acuity_level == 1
    assert "Acute Cardiac Triad" in alert.condition_name
    assert alert.recommended_action == "CALL_911_IMMEDIATELY"
    assert "RED-FLAG EMERGENCY" in alert.triage_message


def test_ec01_asymmetric_risk_override():
    sentinel = ClinicalAcuitySentinel()
    # Heart attack with 9% probability (below GERD, but above 8% safety trigger)
    alert = sentinel.evaluate_triage(
        differential=[("GERD", 0.70), ("Heart attack", 0.09)],
        observed_symptoms=["chest_tightness"],
    )
    assert alert.is_emergency is True
    assert alert.acuity_level == 1
    assert alert.condition_name == "Heart attack"
    assert alert.recommended_action == "CALL_911_IMMEDIATELY"


def test_ec01_routine_non_emergency():
    sentinel = ClinicalAcuitySentinel()
    alert = sentinel.evaluate_triage(
        differential=[("Common Cold", 0.80), ("Allergy", 0.15)],
        observed_symptoms=["runny_nose", "cough"],
    )
    assert alert.is_emergency is False
    assert alert.acuity_level == 3
    assert alert.recommended_action == "PRIMARY_CARE_FOLLOW_UP"


# =========================================================================
# EC-02: Solitary Symptom Active Questioning Tests
# =========================================================================

def test_ec02_solitary_symptom_inquiry(project_paths):
    engine = ActiveQuestioningEngine(project_paths["db"])
    res = engine.evaluate_inquiry(["fatigue"])

    assert res.is_active_inquiry_needed is True
    assert "fatigue" in res.primary_symptoms
    assert res.candidate_diseases_count > 0
    assert len(res.discriminative_questions) >= 1

    # Check question quality
    top_q = res.discriminative_questions[0]
    assert len(top_q.question_text) > 10
    assert len(top_q.clinical_rationale) > 10
    assert top_q.information_gain_score > 0.0


def test_ec02_multi_symptom_no_inquiry_needed(project_paths):
    engine = ActiveQuestioningEngine(project_paths["db"])
    # 4 distinct pathognomonic symptoms
    res = engine.evaluate_inquiry(
        ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
        top_confidence=0.88,
        entropy_percentage=0.20,
    )
    assert res.is_active_inquiry_needed is False


# =========================================================================
# EC-03: Conformal Differential Prediction Tests
# =========================================================================

def test_ec03_conformal_singleton_on_high_certainty():
    predictor = ConformalDifferentialPredictor()
    candidates = [
        ("Fungal infection", 0.96),
        ("Allergy", 0.02),
        ("Drug Reaction", 0.02),
    ]
    res = predictor.predict_conformal_set(candidates, desired_coverage=0.95)
    assert res.is_singleton is True
    assert res.set_size == 1
    assert res.candidates[0].condition == "Fungal infection"
    assert res.candidates[0].cumulative_mass >= 0.81


def test_ec03_conformal_adaptive_set_on_ambiguity():
    predictor = ConformalDifferentialPredictor()
    candidates = [
        ("Influenza", 0.30),
        ("COVID-19", 0.28),
        ("Acute Bronchitis", 0.25),
        ("Common Cold", 0.12),
        ("Pneumonia", 0.05),
    ]
    res = predictor.predict_conformal_set(candidates, desired_coverage=0.95)
    assert res.is_singleton is False
    assert res.set_size >= 3
    assert res.total_coverage_mass >= 0.81
    # Check that top 3 candidates are in set
    conditions_in_set = [c.condition for c in res.candidates]
    assert "Influenza" in conditions_in_set
    assert "COVID-19" in conditions_in_set
    assert "Acute Bronchitis" in conditions_in_set


# =========================================================================
# EC-07: Biological and Demographic Constraint Tests
# =========================================================================

def test_ec07_female_prostate_rejection():
    f = DemographicFilter()
    ok, reason = f.is_biologically_plausible("Prostate Cancer", sex="female")
    assert ok is False
    assert "exclusive to biological male" in reason


def test_ec07_male_ovarian_rejection():
    f = DemographicFilter()
    ok, reason = f.is_biologically_plausible("Ovarian Cyst", sex="male")
    assert ok is False
    assert "exclusive to biological female" in reason


def test_ec07_age_restrictions():
    f = DemographicFilter()
    # Elderly patient with Croup (pediatric)
    ok_croup, reason_croup = f.is_biologically_plausible("Croup", age=70)
    assert ok_croup is False
    assert "restricted to pediatric" in reason_croup

    # Child with Alzheimer's (adult/geriatric)
    ok_alz, reason_alz = f.is_biologically_plausible("Alzheimer Disease", age=6)
    assert ok_alz is False
    assert "typically occurs at or above age" in reason_alz


def test_ec07_differential_filtering_and_renormalization():
    f = DemographicFilter()
    raw = [
        ("Prostate Cancer", 0.60),
        ("Urinary Tract Infection", 0.30),
        ("Bladder Infection", 0.10),
    ]
    filtered, exclusions = f.filter_differential(raw, sex="female")
    assert len(filtered) == 2
    assert "Prostate Cancer" not in [c[0] if isinstance(c, tuple) else c.condition for c in filtered]
    assert len(exclusions) == 1


# =========================================================================
# EC-09: Out-of-Distribution & Non-Medical Input Tests
# =========================================================================

def test_ec09_non_medical_rejection(project_paths):
    router = MultiModelEnsembleRouter(
        project_paths["b_model"],
        project_paths["g_model"],
        project_paths["vocab"],
        project_paths["db"],
    )
    report = router.predict("Write a computer program in Python to sort numbers.")
    assert report.is_out_of_distribution is True
    assert report.differential_diagnosis == []
    assert report.top_confidence == 0.0
    assert report.abstention_triggered is True


# =========================================================================
# End-to-End Ensemble & Diagnostic Engine Integration Tests
# =========================================================================

def test_e2e_ensemble_router_full_cycle(project_paths):
    router = MultiModelEnsembleRouter(
        project_paths["b_model"],
        project_paths["g_model"],
        project_paths["vocab"],
        project_paths["db"],
    )
    # Query with symptoms + demographic metadata
    report = router.predict(
        "Patient has high fever, persistent cough, chills, and fatigue. Denies chest pain.",
        age=35,
        sex="female",
    )
    assert report.is_out_of_distribution is False
    assert len(report.extracted_symptoms) >= 3
    assert "chest_pain" not in report.extracted_symptoms  # NegEx negation verified
    assert report.top_confidence > 0.0
    assert report.conformal_differential is not None
    assert report.conformal_differential.set_size >= 1


def test_e2e_diagnostic_engine_full_cycle(project_paths):
    engine = ClinicalDiagnosticEngine(
        project_paths["resmlp_model"],
        project_paths["vocab"],
        project_paths["db"],
    )
    report = engine.predict(
        "Severe joint pain, knee pain, and stiffness. Denies fever.",
        age=62,
        sex="male",
    )
    assert report.is_out_of_distribution is False
    assert len(report.extracted_symptoms) >= 2
    assert "fever" not in report.extracted_symptoms
    assert report.free_energy != 0.0
    assert report.conformal_differential is not None
    assert len(report.differential_diagnosis) > 0

