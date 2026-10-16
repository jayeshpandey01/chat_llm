"""
Symptom-Grounded Explanation Engine.
===================================
Generates transparent, evidence-grounded clinical explanations for each
candidate condition in the differential diagnosis. Correlates reported symptoms
with precomputed Diagnostic Odds Ratios (DOR) and verified NIH MedQuAD guidelines.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

from model_building.pipeline.schemas import (
    ConditionExplanation,
    MatchedSymptomEvidence,
)

logger = logging.getLogger(__name__)

# Clinical alias map linking diverse classifier labels to evidence DB concepts
CLINICAL_ALIAS_MAP: dict[str, list[str]] = {
    "Heart attack": ["Acute Coronary Syndrome", "Myocardial Infarction", "Angina Pectoris"],
    "Bronchial Asthma": ["Asthma", "Acute Bronchospasm", "Reactive Airway Disease"],
    "GERD": ["Gastroesophageal Reflux Disease", "Esophagitis"],
    "Diabetes": ["Diabetes Mellitus", "Type 2 Diabetes", "Hyperglycemia"],
    "Common Cold": ["Upper Respiratory Infection", "Viral Pharyngitis", "Nasopharyngitis"],
    "Dimorphic hemmorhoids(piles)": ["Hemorrhoids", "Hemorrhoids (Piles)"],
    "Peptic ulcer diseae": ["Peptic Ulcer", "Gastric Ulcer", "Duodenal Ulcer"],
    "Osteoarthristis": ["Osteoarthritis", "Degenerative Joint Disease"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Benign Paroxysmal Positional Vertigo", "Vertigo"],
    "Fungal infection": ["Tinea", "Dermatophytosis", "Candidiasis"],
    "Jaundice": ["Acute Hepatitis", "Cholestasis", "Bile Duct Obstruction"],
    "Hypertension": ["Essential Hypertension", "High Blood Pressure"],
    "urinary tract infection": ["Urinary Tract Infection", "Cystitis", "Pyelonephritis"],
    "Pneumonia": ["Community-Acquired Pneumonia", "Bacterial Pneumonia", "Aspiration Pneumonia"],
    "Migraine": ["Migraine with Aura", "Migraine without Aura", "Vascular Headache"],
    "Allergy": ["Allergic Rhinitis", "Atopic Reaction", "Anaphylaxis"],
    "Gastroenteritis": ["Infectious Gastroenteritis", "Viral Enteritis", "Food Poisoning"],
    "Chicken pox": ["Varicella Zoster Infection", "Varicella"],
    "Dengue": ["Dengue Fever", "Dengue Shock Syndrome"],
    "Typhoid": ["Enteric Fever (Typhoid)", "Salmonella Infection"],
    "Tuberculosis": ["Pulmonary Tuberculosis", "Mycobacterial Disease"],
    "Hypothyroidism": ["Primary Hypothyroidism", "Hashimoto Thyroiditis"],
    "Hyperthyroidism": ["Thyrotoxicosis", "Graves Disease"],
    "Hypoglycemia": ["Acute Hypoglycemia", "Low Blood Sugar"],
    "Arthritis": ["Rheumatoid Arthritis", "Inflammatory Arthritis"],
    "Acne": ["Acne Vulgaris"],
    "Malaria": ["Plasmodium Falciparum Malaria", "Malaria"],
}


class SymptomExplanationEngine:
    """Generates patient-friendly, evidence-backed clinical explanations for predicted conditions."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        if not self.db_path.exists():
            logger.warning(f"Clinical evidence DB not found at {db_path}; using fallback explanations.")

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        if not self.db_path.exists():
            return None
        return sqlite3.connect(str(self.db_path))

    def _find_matching_db_disease(self, cursor: sqlite3.Cursor, disease_name: str) -> Optional[str]:
        """Resolves classifier disease name to matching evidence DB concept."""
        # 1. Exact match
        cursor.execute("SELECT disease FROM evidence_odds WHERE disease = ? LIMIT 1", (disease_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        # 2. Case-insensitive exact match
        cursor.execute("SELECT disease FROM evidence_odds WHERE LOWER(disease) = LOWER(?) LIMIT 1", (disease_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        # 3. Known clinical aliases
        aliases = CLINICAL_ALIAS_MAP.get(disease_name, [])
        for alias in aliases:
            cursor.execute("SELECT disease FROM evidence_odds WHERE LOWER(disease) = LOWER(?) LIMIT 1", (alias,))
            row = cursor.fetchone()
            if row:
                return row[0]

        # 4. Partial substring match
        cursor.execute("SELECT disease FROM evidence_odds WHERE disease LIKE ? LIMIT 1", (f"%{disease_name}%",))
        row = cursor.fetchone()
        if row:
            return row[0]

        return None

    def _fetch_nih_guidelines(
        self, cursor: sqlite3.Cursor, disease_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Queries NIH MedQuAD guidelines for topic overview and summaries."""
        # Try direct match
        cursor.execute(
            "SELECT topic, answer FROM clinical_guidelines WHERE topic LIKE ? LIMIT 1",
            (f"%{disease_name}%",),
        )
        row = cursor.fetchone()
        if row:
            topic, ans = row[0], row[1]
            summary = ans[:280] + "..." if len(ans) > 280 else ans
            return topic, summary

        # Try aliases
        for alias in CLINICAL_ALIAS_MAP.get(disease_name, []):
            cursor.execute(
                "SELECT topic, answer FROM clinical_guidelines WHERE topic LIKE ? LIMIT 1",
                (f"%{alias}%",),
            )
            row = cursor.fetchone()
            if row:
                topic, ans = row[0], row[1]
                summary = ans[:280] + "..." if len(ans) > 280 else ans
                return topic, summary

        return None, None

    def explain_candidate(
        self,
        condition: str,
        probability: float,
        rank: int,
        source_model: str,
        observed_symptoms: List[str],
        denied_symptoms: Optional[List[str]] = None,
    ) -> ConditionExplanation:
        """Constructs an explainable ConditionExplanation record."""
        denied_symptoms = denied_symptoms or []
        conn = self._get_connection()
        matched_evidence: List[MatchedSymptomEvidence] = []
        matched_symptom_names: List[str] = []
        missing_symptoms: List[str] = []
        nih_topic: Optional[str] = None
        nih_summary: Optional[str] = None
        max_dor = 1.0

        if conn is not None:
            try:
                cursor = conn.cursor()
                resolved_db_disease = self._find_matching_db_disease(cursor, condition)

                if resolved_db_disease:
                    # Query all known symptoms for this disease in evidence DB
                    cursor.execute(
                        """
                        SELECT symptom, p_s_given_disease, odds_ratio, is_pathognomonic
                        FROM evidence_odds
                        WHERE disease = ?
                        ORDER BY odds_ratio DESC
                        """,
                        (resolved_db_disease,),
                    )
                    all_disease_symptoms = cursor.fetchall()
                    disease_symptom_set = {row[0]: row for row in all_disease_symptoms}

                    # Correlate observed symptoms with disease symptoms
                    for s in observed_symptoms:
                        s_norm = s.strip().lower().replace(" ", "_")
                        if s_norm in disease_symptom_set:
                            row = disease_symptom_set[s_norm]
                            dor = float(row[2])
                            if dor > max_dor:
                                max_dor = dor
                            matched_evidence.append(
                                MatchedSymptomEvidence(
                                    symptom=s_norm,
                                    diagnostic_odds_ratio=dor,
                                    p_s_given_disease=float(row[1]),
                                    is_pathognomonic=bool(row[3]),
                                )
                            )
                            matched_symptom_names.append(s_norm.replace("_", " "))

                    # Find top 2 classic symptoms not reported
                    observed_norm_set = {s.strip().lower().replace(" ", "_") for s in observed_symptoms}
                    for row in all_disease_symptoms:
                        s_name = row[0]
                        if s_name not in observed_norm_set and s_name not in denied_symptoms:
                            missing_symptoms.append(s_name.replace("_", " "))
                            if len(missing_symptoms) >= 2:
                                break

                    # Query NIH MedQuAD guidelines
                    nih_topic, nih_summary = self._fetch_nih_guidelines(cursor, resolved_db_disease)

            except Exception as e:
                logger.error(f"Error querying clinical evidence for {condition}: {e}")
            finally:
                conn.close()

        # Fallback if no database match: use observed symptoms directly
        if not matched_symptom_names:
            matched_symptom_names = [s.replace("_", " ") for s in observed_symptoms[:3]]

        # Synthesize patient-friendly clinical explanation
        matched_str = ", ".join(matched_symptom_names) if matched_symptom_names else "reported indicators"
        missing_str = f" Commonly associated signs such as {', '.join(missing_symptoms)} were not mentioned." if missing_symptoms else ""
        denied_str = f" The absence of {', '.join(denied_symptoms[:2])} helps differentiate this presentation." if denied_symptoms else ""

        if max_dor > 10.0:
            evidence_weight = "shows strong clinical correlation with"
        elif max_dor > 2.0:
            evidence_weight = "is moderately suggestive of"
        else:
            evidence_weight = "is compatible with"

        explanation_text = (
            f"Your reported symptom profile ({matched_str}) {evidence_weight} {condition}."
            f"{missing_str}{denied_str} "
            f"A healthcare provider should perform a physical assessment to confirm."
        )

        return ConditionExplanation(
            condition=condition,
            probability=round(probability, 4),
            confidence_percentage=round(probability * 100.0, 1),
            rank=rank,
            source_model=source_model,
            matched_symptoms=matched_symptom_names,
            missing_classic_symptoms=missing_symptoms,
            evidence_symptoms=matched_evidence,
            explanation_text=explanation_text,
            nih_guideline_topic=nih_topic,
            nih_guideline_summary=nih_summary,
        )

