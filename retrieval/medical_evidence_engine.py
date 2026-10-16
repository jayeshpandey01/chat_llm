"""Clinical Medical Evidence Engine.

Coordinates authoritative medical platforms (NIH MedlinePlus, PubMed, and clinical evidence DB)
to retrieve quantitative Diagnostic Odds Ratios (DOR), pathognomonic hallmarks,
and guideline-backed clinical summaries for candidate conditions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from ingestion.sources.medlineplus import MedlinePlusSource

logger = logging.getLogger(__name__)


class MedicalEvidenceEngine:
    """Production medical evidence retrieval and grounding engine."""

    def __init__(self, medline_source: Optional[MedlinePlusSource] = None):
        self.medline = medline_source or MedlinePlusSource()
        self._cache: Dict[str, dict[str, Any]] = {}

    def get_evidence_for_condition(
        self, condition: str, present_symptoms: Optional[List[str]] = None
    ) -> dict[str, Any]:
        """Fetch structured clinical evidence for a specific condition."""
        norm_key = condition.strip().lower()
        if norm_key in self._cache:
            return self._cache[norm_key]

        evidence = self.medline.get_disease_evidence(condition, present_symptoms=present_symptoms)
        self._cache[norm_key] = evidence
        return evidence

    def get_multi_condition_evidence(
        self, conditions: List[str], present_symptoms: Optional[List[str]] = None, max_conditions: int = 2
    ) -> List[dict[str, Any]]:
        """Retrieve and rank evidence across multiple differential conditions."""
        results = []
        for cond in conditions[:max_conditions]:
            if not cond or "undetermined" in cond.lower():
                continue
            ev = self.get_evidence_for_condition(cond, present_symptoms=present_symptoms)
            results.append(ev)
        return results

    def format_evidence_block(
        self, conditions: List[str], present_symptoms: Optional[List[str]] = None, max_tokens: int = 60
    ) -> str:
        """Assembles a compact, high-signal medical evidence prompt block for context synthesis."""
        evidence_list = self.get_multi_condition_evidence(conditions, present_symptoms=present_symptoms, max_conditions=2)
        if not evidence_list:
            return "No specific NIH clinical guidelines indexed for this presentation."

        lines = ["[RETRIEVED MEDICAL EVIDENCE & GUIDELINES]"]
        for ev in evidence_list:
            cond_name = ev["disease"]
            source = ev.get("source", "NIH / MedlinePlus")
            overview = ev.get("overview", "")
            # Truncate overview to 1 concise sentence
            first_sentence = overview.split(".")[0].strip() if overview else ""
            if first_sentence:
                lines.append(f"• {cond_name} ({source}): {first_sentence}.")

            # Add pathognomonic hallmarks or high DOR symptoms
            hallmarks = ev.get("pathognomonic_hallmarks", [])
            if hallmarks:
                lines.append(f"  Key hallmarks: {', '.join(hallmarks[:3])}")
            elif ev.get("diagnostic_odds_ratios"):
                top_dors = [
                    f"{d['symptom']} (DOR {d['diagnostic_odds_ratio']:.1f}x)"
                    for d in ev["diagnostic_odds_ratios"][:2]
                ]
                lines.append(f"  Diagnostic markers: {', '.join(top_dors)}")

        return "\n".join(lines)


# Shared singleton instance
medical_evidence_engine = MedicalEvidenceEngine()

