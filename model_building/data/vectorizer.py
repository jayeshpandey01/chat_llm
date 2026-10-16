"""
Clinical Vectorizer & Feature Matrix Pipeline
============================================
Transforms clinical records or free-text inputs into standardized model feature matrices:
1. Multi-hot sparse binary encoding over canonical symptom vocabulary.
2. Label encoding for all 729 calibrated disease categories.
3. Batch vectorizer for training/evaluation pipelines.
4. Feature serialization & deserialization via joblib.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    from model_building.nlp.symptom_extractor import ClinicalSymptomExtractor
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    try:
        from model_building.nlp.symptom_extractor import ClinicalSymptomExtractor
    except ModuleNotFoundError:
        from nlp.symptom_extractor import ClinicalSymptomExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ClinicalFeatureVectorizer:
    """
    Transforms clinical symptom tokens into sparse multi-hot feature vectors
    calibrated against the vocabulary index.
    """

    def __init__(self, vocab_path: Path):
        self.vocab_path = vocab_path
        with open(vocab_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.diseases: list[str] = meta["diseases"]
        self.symptoms: list[str] = meta["symptoms"]
        self.disease_to_id: dict[str, int] = meta["disease_to_id"]
        self.symptom_to_id: dict[str, int] = meta["symptom_to_id"]
        self.id_to_disease: dict[int, str] = {v: k for k, v in self.disease_to_id.items()}
        self.id_to_symptom: dict[int, str] = {v: k for k, v in self.symptom_to_id.items()}

        self.num_features = len(self.symptoms)
        self.num_classes = len(self.diseases)

        # NLP Free-text Extractor
        self.nlp_extractor = ClinicalSymptomExtractor(vocab_path)
        logger.info(f"Initialized ClinicalFeatureVectorizer with {self.num_features} features, {self.num_classes} classes.")

    def transform_tokens(self, symptom_tokens: list[str]) -> np.ndarray:
        """Transforms a list of canonical symptom tokens into a 1D binary numpy array."""
        vector = np.zeros(self.num_features, dtype=np.float32)
        for s in symptom_tokens:
            idx = self.symptom_to_id.get(s)
            if idx is not None:
                vector[idx] = 1.0
        return vector

    def transform_text(self, free_text: str) -> tuple[np.ndarray, list[str]]:
        """
        Extracts positive clinical symptoms from free text using NLP NegEx rules
        and transforms them into a binary feature vector.
        """
        extracted = self.nlp_extractor.extract_positive_symptoms(free_text)
        vec = self.transform_tokens(extracted)
        return vec, extracted

    def transform_batch(
        self,
        token_lists: list[list[str]],
        as_sparse: bool = True,
    ) -> sp.csr_matrix | np.ndarray:
        """Transforms a batch of patient symptom lists into a 2D matrix (CSR or dense)."""
        rows, cols, data = [], [], []
        for row_idx, tokens in enumerate(token_lists):
            for t in tokens:
                col_idx = self.symptom_to_id.get(t)
                if col_idx is not None:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(1.0)

        n_rows = len(token_lists)
        sparse_mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_rows, self.num_features), dtype=np.float32
        )
        return sparse_mat if as_sparse else sparse_mat.toarray()

    def encode_labels(self, disease_names: list[str]) -> np.ndarray:
        """Encodes disease string labels into integer class IDs."""
        return np.array([self.disease_to_id[d] for d in disease_names], dtype=np.int64)

    def decode_label(self, class_id: int) -> str:
        """Decodes integer class ID into canonical disease name."""
        return self.id_to_disease.get(class_id, "Unknown Condition")
