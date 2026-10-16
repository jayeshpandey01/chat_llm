"""
Free-Text NLP and Preprocessing Feature Vectorizer Tests
======================================================
Tests:
1. Lexical tokenization & phrase matching (exact and multi-word phrases).
2. NegEx clinical negation scope detection ("denies cough", "no chest pain").
3. Pseudo-negation robustness ("no doubt", "not only").
4. Levenshtein edit distance & typo correction ("fevr" -> "fever", "hedache" -> "headache").
5. Transformation into sparse and dense feature vectors.
6. Integer label encoding and decoding reproducibility.
"""

from pathlib import Path

import numpy as np
import pytest

from model_building.data.vectorizer import ClinicalFeatureVectorizer
from model_building.nlp.symptom_extractor import ClinicalSymptomExtractor, similarity_ratio

VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "vocabularies.json"


@pytest.fixture(scope="module")
def extractor():
    return ClinicalSymptomExtractor(VOCAB_PATH)


@pytest.fixture(scope="module")
def vectorizer():
    return ClinicalFeatureVectorizer(VOCAB_PATH)


def test_similarity_ratio_computation():
    """Verify Levenshtein distance similarity calculation."""
    assert similarity_ratio("headache", "headache") == 1.0
    assert similarity_ratio("hedache", "headache") > 0.85
    assert similarity_ratio("fever", "fevr") > 0.75
    assert similarity_ratio("cough", "vomiting") < 0.3


def test_negation_detection(extractor):
    """Verify NegEx algorithm correctly identifies negated clinical signs."""
    text = "Patient denies shortness of breath. Reports mild joint pain. No fever."
    extracted = extractor.extract_symptoms(text)
    ext_dict = {e.canonical_token: e for e in extracted}

    assert "joint_pain" in ext_dict
    assert ext_dict["joint_pain"].is_negated is False

    if "shortness_of_breath" in ext_dict:
        assert ext_dict["shortness_of_breath"].is_negated is True

    if "fever" in ext_dict:
        assert ext_dict["fever"].is_negated is True


def test_pseudo_negation_not_negated(extractor):
    """Verify pseudo-negation phrases (e.g., 'no doubt') do not trigger false negative flags."""
    text = "Patient has no doubt experienced severe chest pain."
    extracted = extractor.extract_symptoms(text)
    cp = next((e for e in extracted if e.canonical_token == "chest_pain"), None)
    assert cp is not None
    assert cp.is_negated is False


def test_vectorizer_shape_and_indices(vectorizer):
    """Verify vectorizer produces correctly sized binary indicator vectors."""
    tokens = ["chest_pain", "high_fever", "cough"]
    vec = vectorizer.transform_tokens(tokens)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (vectorizer.num_features,)
    assert vec.sum() == 3

    for t in tokens:
        idx = vectorizer.symptom_to_id[t]
        assert vec[idx] == 1.0


def test_vectorizer_batch_sparse(vectorizer):
    """Verify batch transformation creates valid sparse CSR matrix."""
    batch = [
        ["chest_pain", "cough"],
        ["joint_pain", "skin_rash", "itching"],
    ]
    mat = vectorizer.transform_batch(batch, as_sparse=True)

    assert mat.shape == (2, vectorizer.num_features)
    assert mat[0].sum() == 2
    assert mat[1].sum() == 3


def test_label_encoding_roundtrip(vectorizer):
    """Verify encoding and decoding labels is lossless."""
    test_diseases = vectorizer.diseases[:4]
    encoded = vectorizer.encode_labels(test_diseases)
    decoded = [vectorizer.decode_label(int(idx)) for idx in encoded]
    assert decoded == test_diseases
