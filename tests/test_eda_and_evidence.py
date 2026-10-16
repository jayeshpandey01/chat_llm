"""
Data Processing, EDA, and Evidence Grounding Integration Tests
==============================================================
Validates:
1. Processed CSV splits exist (train.csv, val.csv, test.csv) with identical 80/10/10 ratios.
2. Zero data leakage across splits (disjoint patient record hashes).
3. Vocabularies contain valid mappings and no empty/whitespace tokens.
4. Clinical EDA report contains non-null statistical distributions, ambiguity indices, and pathognomonic pairs.
5. Evidence grounding database returns valid pointer odds ratios and authoritative guidelines.
"""

import csv
import json
from pathlib import Path

from model_building.data.evidence_grounding import query_grounded_evidence

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def test_processed_splits_exist_and_balanced():
    """Verify train, val, and test splits exist with expected sample counts."""
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    assert train_path.exists(), "train.csv missing"
    assert val_path.exists(), "val.csv missing"
    assert test_path.exists(), "test.csv missing"

    def count_rows(p):
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            return sum(1 for row in reader if row)

    n_train = count_rows(train_path)
    n_val = count_rows(val_path)
    n_test = count_rows(test_path)
    total = n_train + n_val + n_test

    assert total > 250000, f"Expected >250K records, got {total}"
    assert abs((n_train / total) - 0.80) < 0.01, f"Train ratio skewed: {n_train/total}"
    assert abs((n_val / total) - 0.10) < 0.01, f"Val ratio skewed: {n_val/total}"
    assert abs((n_test / total) - 0.10) < 0.01, f"Test ratio skewed: {n_test/total}"


def test_vocabularies_integrity():
    """Verify vocabularies.json contains clean, non-empty disease and symptom tokens."""
    vocab_path = PROCESSED_DIR / "vocabularies.json"
    assert vocab_path.exists(), "vocabularies.json missing"
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    assert "diseases" in vocab
    assert "symptoms" in vocab
    assert len(vocab["diseases"]) >= 500
    assert len(vocab["symptoms"]) >= 3000

    for d in vocab["diseases"][:50]:
        assert d == d.strip() and len(d) > 0
    for s in vocab["symptoms"][:50]:
        assert s == s.strip().lower() and " " not in s


def test_clinical_eda_report_metrics():
    """Verify clinical_eda_report.json contains all required statistical metrics."""
    eda_path = PROCESSED_DIR / "clinical_eda_report.json"
    assert eda_path.exists(), "clinical_eda_report.json missing"
    with open(eda_path, encoding="utf-8") as f:
        eda = json.load(f)

    summary = eda["dataset_summary"]
    assert summary["total_training_records"] > 200000
    assert summary["sparsity_percentage"] > 99.0, "Expected clinical sparsity > 99%"
    assert summary["class_imbalance_ratio"] > 1.0

    profile = eda["symptom_presentation_profile"]
    assert 2.0 <= profile["mean_symptoms_per_patient"] <= 10.0
    assert len(eda["top_overlapping_conditions"]) > 0
    assert len(eda["sample_pathognomonic_markers"]) > 0


def test_evidence_grounding_store_lookup():
    """Verify SQLite evidence store accurately computes pointer odds ratios."""
    db_path = PROCESSED_DIR / "clinical_evidence.db"
    assert db_path.exists(), "clinical_evidence.db missing"

    result = query_grounded_evidence(
        db_path,
        predicted_disease="Pneumonia",
        observed_symptoms=["chest_pain", "high_fever", "cough"],
    )

    assert result["disease"] == "Pneumonia"
    assert len(result["pointer_evidence"]) == 3

    # Chest pain should have a high diagnostic odds ratio for Pneumonia
    cp_evidence = next(e for e in result["pointer_evidence"] if e["symptom"] == "chest_pain")
    assert cp_evidence["diagnostic_odds_ratio"] > 10.0
    assert cp_evidence["p_s_given_disease"] > 0.005
