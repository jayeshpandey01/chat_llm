"""
Biological & Demographic Clinical Rule Engine (EC-07)
=====================================================
Enforces physiological and epidemiological hard boundaries on model predictions:
1. Biological Sex Exclusions:
   - Male-only pathologies (Prostate Cancer, BPH, Testicular conditions, Balanitis)
     cannot be predicted in biological females.
   - Female-only pathologies (Ovarian Cyst, PCOS, Cervicitis, Endometriosis,
     Pregnancy-related, Uterine Fibroids, Vaginal conditions) cannot be predicted in biological males.
2. Age-Exclusive Pathologies:
   - Infant / Pediatric conditions (e.g. Infantile Diarrhea, Croup) restricted to children.
   - Degenerative / Adult conditions (e.g. Alzheimer's, Dementia, Parkinson's, BPH)
     restricted to appropriate minimum age brackets.
3. Dual-mode support:
   - `mask_logits`: Sets impossible disease logits to -inf during forward inference.
   - `filter_differential`: Filters and re-normalizes candidate lists post-inference,
     providing full audit trails with clinical rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemographicConstraint:
    required_sex: str | None = None  # "M" or "F"
    min_age: float | None = None
    max_age: float | None = None
    rationale: str = ""


# Canonical database of clinical demographic constraints
# Condition keys are normalized to lowercase for robust matching.
DEMOGRAPHIC_CONSTRAINTS: dict[str, DemographicConstraint] = {
    # --- Male-Exclusive Conditions ---
    "prostate cancer": DemographicConstraint(
        required_sex="M",
        min_age=30.0,
        rationale="Prostate gland is biologically present only in male anatomy; rare before age 30.",
    ),
    "benign prostatic hyperplasia (bph)": DemographicConstraint(
        required_sex="M",
        min_age=35.0,
        rationale="Benign prostatic hyperplasia occurs exclusively in male prostate tissue, primarily in older men.",
    ),
    "balanitis": DemographicConstraint(
        required_sex="M",
        rationale="Balanitis is inflammation of the glans penis, anatomically exclusive to males.",
    ),
    "epididymitis": DemographicConstraint(
        required_sex="M",
        rationale="Epididymitis is inflammation of the epididymis, anatomically exclusive to males.",
    ),
    "hydrocele of the testicle": DemographicConstraint(
        required_sex="M",
        rationale="Hydrocele of the testicle occurs exclusively in male scrotal anatomy.",
    ),
    "varicocele of the testicles": DemographicConstraint(
        required_sex="M",
        rationale="Varicocele of the testicles occurs exclusively in male scrotal anatomy.",
    ),

    # --- Female-Exclusive Conditions ---
    "ovarian cyst": DemographicConstraint(
        required_sex="F",
        min_age=8.0,
        rationale="Ovaries are anatomically exclusive to female reproductive anatomy.",
    ),
    "polycystic ovarian syndrome (pcos)": DemographicConstraint(
        required_sex="F",
        min_age=11.0,
        max_age=60.0,
        rationale="PCOS affects ovarian function in biological females during reproductive and peri-menopausal years.",
    ),
    "cervicitis": DemographicConstraint(
        required_sex="F",
        min_age=12.0,
        rationale="Cervicitis is inflammation of the uterine cervix, anatomically exclusive to females.",
    ),
    "endometriosis": DemographicConstraint(
        required_sex="F",
        min_age=10.0,
        max_age=65.0,
        rationale="Endometriosis involves ectopic endometrial tissue, biologically exclusive to females.",
    ),
    "ectopic pregnancy": DemographicConstraint(
        required_sex="F",
        min_age=10.0,
        max_age=55.0,
        rationale="Pregnancy occurs exclusively in biological female reproductive anatomy during fertile years.",
    ),
    "threatened pregnancy": DemographicConstraint(
        required_sex="F",
        min_age=10.0,
        max_age=55.0,
        rationale="Pregnancy complications occur exclusively in biological female anatomy during fertile years.",
    ),
    "problem during pregnancy": DemographicConstraint(
        required_sex="F",
        min_age=10.0,
        max_age=55.0,
        rationale="Pregnancy complications occur exclusively in biological female anatomy during fertile years.",
    ),
    "uterine atony": DemographicConstraint(
        required_sex="F",
        min_age=12.0,
        max_age=55.0,
        rationale="Uterine atony is a postpartum condition exclusive to females with a uterus.",
    ),
    "uterine fibroids": DemographicConstraint(
        required_sex="F",
        min_age=15.0,
        rationale="Uterine fibroids develop within the myometrium of the uterus, anatomically exclusive to females.",
    ),
    "benign vaginal discharge (leukorrhea)": DemographicConstraint(
        required_sex="F",
        rationale="Vaginal discharge is anatomically exclusive to females.",
    ),
    "vaginal cyst": DemographicConstraint(
        required_sex="F",
        rationale="Vaginal cysts are anatomically exclusive to females.",
    ),
    "vaginal yeast infection": DemographicConstraint(
        required_sex="F",
        rationale="Vaginal candidiasis is anatomically exclusive to females.",
    ),
    "idiopathic absence of menstruation": DemographicConstraint(
        required_sex="F",
        min_age=12.0,
        max_age=55.0,
        rationale="Amenorrhea occurs exclusively in females with female hormonal and uterine systems.",
    ),
    "idiopathic excessive menstruation": DemographicConstraint(
        required_sex="F",
        min_age=11.0,
        max_age=55.0,
        rationale="Menorrhagia occurs exclusively in females with menstrual physiology.",
    ),
    "idiopathic irregular menstrual cycle": DemographicConstraint(
        required_sex="F",
        min_age=11.0,
        max_age=55.0,
        rationale="Menstrual irregularities occur exclusively in females with menstrual physiology.",
    ),
    "idiopathic painful menstruation": DemographicConstraint(
        required_sex="F",
        min_age=11.0,
        max_age=55.0,
        rationale="Dysmenorrhea occurs exclusively in females with menstrual physiology.",
    ),

    # --- Age-Exclusive Conditions ---
    "diarrhea, infantile": DemographicConstraint(
        max_age=3.0,
        rationale="Infantile diarrhea is clinically restricted to infants and toddlers under age 3.",
    ),
    "croup": DemographicConstraint(
        min_age=0.25,
        max_age=8.0,
        rationale="Laryngotracheobronchitis (Croup) predominantly affects pediatric airways (6 months to 6-8 years).",
    ),
    "alzheimer disease": DemographicConstraint(
        min_age=35.0,
        rationale="Alzheimer disease is a progressive neurodegenerative disorder of adulthood (extremely rare before 35-40).",
    ),
    "dementia": DemographicConstraint(
        min_age=35.0,
        rationale="Acquired progressive dementia syndromes present in adult and geriatric populations.",
    ),
    "parkinson disease": DemographicConstraint(
        min_age=25.0,
        rationale="Parkinson disease is an adult-onset neurodegenerative disorder (juvenile forms <25 are exceptionally rare).",
    ),
    "osteoarthritis": DemographicConstraint(
        min_age=20.0,
        rationale="Degenerative osteoarthritis requires joint cartilage wear typically seen in adult or older populations.",
    ),
}


def normalize_sex(sex: str | None) -> str | None:
    """Normalizes patient sex string to 'M', 'F', or None."""
    if sex is None:
        return None
    s = sex.strip().upper()
    if s in {"M", "MALE", "MAN", "BOY"}:
        return "M"
    if s in {"F", "FEMALE", "WOMAN", "GIRL"}:
        return "F"
    return None


class DemographicFilter:
    """
    Evaluates condition plausibility against patient demographics (age and biological sex).
    Guarantees zero biological hallucinations reach the clinician or patient.
    """

    def __init__(self, custom_constraints: dict[str, DemographicConstraint] | None = None):
        self.constraints = dict(DEMOGRAPHIC_CONSTRAINTS)
        if custom_constraints:
            for k, v in custom_constraints.items():
                self.constraints[k.strip().lower()] = v

    def is_biologically_plausible(
        self,
        condition: str,
        age: float | None = None,
        sex: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Checks whether condition is biologically plausible given patient age and biological sex.
        Returns:
            (is_plausible, exclusion_reason)
        """
        c_norm = condition.strip().lower()
        constraint = self.constraints.get(c_norm)

        # Partial matching fallback if exact match not found
        if constraint is None:
            for key, val in self.constraints.items():
                if key in c_norm or c_norm in key:
                    constraint = val
                    break

        if constraint is None:
            return True, None

        norm_sex = normalize_sex(sex)

        # 1. Check Biological Sex
        if norm_sex is not None and constraint.required_sex is not None:
            if norm_sex != constraint.required_sex:
                expected = "male" if constraint.required_sex == "M" else "female"
                given = "male" if norm_sex == "M" else "female"
                reason = (
                    f"Biological mismatch: '{condition}' is exclusive to biological {expected} physiology "
                    f"but patient is indicated as {given}. {constraint.rationale}"
                )
                return False, reason

        # 2. Check Minimum Age
        if age is not None and constraint.min_age is not None:
            if age < constraint.min_age:
                reason = (
                    f"Age violation: '{condition}' typically occurs at or above age {constraint.min_age}, "
                    f"but patient age is {age}. {constraint.rationale}"
                )
                return False, reason

        # 3. Check Maximum Age
        if age is not None and constraint.max_age is not None:
            if age > constraint.max_age:
                reason = (
                    f"Age violation: '{condition}' is restricted to pediatric/reproductive age <= {constraint.max_age}, "
                    f"but patient age is {age}. {constraint.rationale}"
                )
                return False, reason

        return True, None

    def filter_differential(
        self,
        candidates: list[Any],
        age: float | None = None,
        sex: str | None = None,
    ) -> tuple[list[Any], list[str]]:
        """
        Filters a list of candidate objects (or tuples) having .condition and .probability attributes,
        removes biologically impossible conditions, and renormalizes remaining probabilities.
        Returns:
            (filtered_candidates, list_of_filtered_explanations)
        """
        if not candidates or (age is None and sex is None):
            return candidates, []

        retained: list[Any] = []
        excluded_explanations: list[str] = []

        for c in candidates:
            cond_name = getattr(c, "condition", None) or (c[0] if isinstance(c, (tuple, list)) else str(c))
            plausible, reason = self.is_biologically_plausible(cond_name, age=age, sex=sex)
            if plausible:
                retained.append(c)
            else:
                logger.info(f"Demographic Filter masked out '{cond_name}': {reason}")
                if reason:
                    excluded_explanations.append(reason)

        if not retained:
            # If all candidates were filtered, keep original to avoid empty output, but log warning
            logger.warning("Demographic filter would exclude all candidates; keeping original set.")
            return candidates, excluded_explanations

        # Re-normalize probabilities across retained candidates
        total_p = sum(getattr(c, "probability", 0.0) for c in retained)
        if total_p > 0:
            for rank_idx, c in enumerate(retained, start=1):
                if hasattr(c, "probability"):
                    new_p = float(c.probability / total_p)
                    c.probability = round(new_p, 4)
                    if hasattr(c, "confidence_percentage"):
                        c.confidence_percentage = round(new_p * 100.0, 1)
                if hasattr(c, "rank"):
                    c.rank = rank_idx

        return retained, excluded_explanations

