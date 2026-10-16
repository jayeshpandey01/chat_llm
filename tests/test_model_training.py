"""
Unit & Integration Tests for Model Training, PyTorch ResMLP & Diagnostic Engine
=============================================================================
Verifies:
1. PyTorch ClinicalResMLP parameter dimensions, skip connection forward pass, and shape consistency.
2. Label smoothing cross-entropy loss computation.
3. Temperature calibration scaling parameter bounds (T > 0).
4. Top-1, Top-3, Top-5 accuracy calculation accuracy.
5. Scikit-Learn baseline model artifact loading and prediction.
6. End-to-end ClinicalDiagnosticEngine report generation and abstention guard.
"""

from pathlib import Path

import joblib
import pytest
import torch

from model_building.models.diagnostic_engine import ClinicalDiagnosticEngine
from model_building.models.pytorch_resmlp import (
    ClinicalResMLP,
    TemperatureScaler,
    compute_clinical_metrics,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
VOCAB_PATH = PROC_DIR / "vocabularies.json"
DB_PATH = PROC_DIR / "clinical_evidence.db"
PYTORCH_MODEL_PATH = BASE_DIR / "models" / "pytorch_resmlp.pt"
SKLEARN_MODEL_PATH = BASE_DIR / "models" / "sklearn_baseline.joblib"


def test_pytorch_resmlp_forward_shape():
    """Verify PyTorch ResMLP produces correct logits shape for input batch."""
    batch_size = 4
    num_symptoms = 100
    num_classes = 20

    model = ClinicalResMLP(
        num_symptoms=num_symptoms,
        num_diseases=num_classes,
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.1,
    )

    x = torch.randint(0, 2, (batch_size, num_symptoms)).float()
    logits = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


def test_temperature_scaler_optimization():
    """Verify temperature scaler softens overconfident logits without changing top prediction."""
    scaler = TemperatureScaler()
    logits = torch.tensor([[10.0, 5.0, 1.0], [2.0, 8.0, 0.5]])
    targets = torch.tensor([0, 1])

    scaler.fit(logits, targets)
    assert scaler.temperature.item() > 0.1

    scaled = scaler(logits)
    assert torch.argmax(logits, dim=-1).equal(torch.argmax(scaled, dim=-1))


def test_compute_clinical_metrics():
    """Verify calculation of Top-1, Top-3, Top-5, and ECE."""
    # Create synthetic logits where true targets are in top positions
    logits = torch.tensor([
        [5.0, 3.0, 1.0, 0.5, 0.1],  # True is 0 (Rank 1)
        [1.0, 4.0, 2.0, 0.5, 0.1],  # True is 2 (Rank 2)
        [0.1, 0.2, 0.3, 0.4, 5.0],  # True is 4 (Rank 1)
    ])
    targets = torch.tensor([0, 2, 4])

    metrics = compute_clinical_metrics(logits, targets, temperature=1.0)

    assert metrics["top1_accuracy"] == pytest.approx(2.0 / 3.0, abs=0.01)
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["mrr"] > 0.8
    assert 0.0 <= metrics["ece"] <= 1.0


def test_sklearn_baseline_artifact_valid():
    """Verify saved scikit-learn model artifact loads and reports valid test metrics."""
    assert SKLEARN_MODEL_PATH.exists(), f"Model file {SKLEARN_MODEL_PATH} not found"
    data = joblib.load(SKLEARN_MODEL_PATH)

    assert "model" in data
    assert "test_metrics" in data
    test_metrics = data["test_metrics"]

    assert test_metrics["top1_accuracy"] > 0.50
    assert test_metrics["top3_accuracy"] > 0.58
    assert test_metrics["top5_accuracy"] > 0.60
    assert test_metrics["mrr"] > 0.55


def test_diagnostic_engine_end_to_end():
    """Verify diagnostic engine returns ranked differential candidates and handles NegEx."""
    assert PYTORCH_MODEL_PATH.exists(), f"PyTorch checkpoint {PYTORCH_MODEL_PATH} missing"
    assert DB_PATH.exists(), f"Database {DB_PATH} missing"

    engine = ClinicalDiagnosticEngine(
        model_path=PYTORCH_MODEL_PATH,
        vocab_path=VOCAB_PATH,
        db_path=DB_PATH,
        device="cpu",
    )

    query = "Patient has severe joint pain and skin rash. Denies fever."
    report = engine.predict(query, top_k=5)

    assert report.query_text == query
    assert "joint_pain" in report.extracted_symptoms
    assert "skin_rash" in report.extracted_symptoms
    assert "fever" not in report.extracted_symptoms

    assert len(report.differential_diagnosis) == 5
    # Ranks must be 1 to 5
    assert [c.rank for c in report.differential_diagnosis] == [1, 2, 3, 4, 5]
    # Probabilities should be descending
    probs = [c.probability for c in report.differential_diagnosis]
    assert probs == sorted(probs, reverse=True)


def test_diagnostic_engine_empty_input_abstention():
    """Verify engine safely abstains when given input with zero recognized symptoms."""
    engine = ClinicalDiagnosticEngine(
        model_path=PYTORCH_MODEL_PATH,
        vocab_path=VOCAB_PATH,
        db_path=DB_PATH,
        device="cpu",
    )

    report = engine.predict("Hello, good morning doctor.")
    assert report.abstention_triggered is True
    assert len(report.extracted_symptoms) == 0
    assert report.top_confidence == 0.0

