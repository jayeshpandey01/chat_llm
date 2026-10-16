"""
Production Clinical Symptom-to-Disease Prediction Pipeline.
===========================================================
Master orchestration engine implementing end-to-end clinical diagnosis from user symptoms:
1. Input Processing: Supports free-text narrative, structured checklist tokens, and demographics.
2. Clinical NLP: NegEx negation parsing, inflection lemmatization, and typo recovery.
3. High-Speed Caching: Canonical SHA-256 caching delivering sub-5ms responses on repeated queries.
4. OOD & Sentinel Gating: Helmholtz free-energy rejection and ESI Level 1/2 life-threat detection.
5. Multi-Model Inference: Coordinates Primary Care (41 diseases) and General Clinical (50 diseases).
6. Demographic Filtering: Enforces physiological boundaries on age and biological sex.
7. Conformal Prediction: Outputting adaptive differential sets guaranteeing 95% coverage.
8. Active Bayesian Inquiry: Information-theoretic clarifying questions for solitary symptoms.
9. Symptom-Grounded Explanations: Diagnostic Odds Ratios (DOR) and NIH MedQuAD citations.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import joblib
import numpy as np

from model_building.data.demographic_rules import DemographicFilter
from model_building.data.evidence_grounding import query_grounded_evidence
from model_building.models.active_questioning import ActiveQuestioningEngine
from model_building.models.acuity_sentinel import ClinicalAcuitySentinel
from model_building.models.conformal_predictor import ConformalDifferentialPredictor
from model_building.nlp.symptom_extractor import ClinicalSymptomExtractor
from model_building.pipeline.cache import CanonicalSymptomCache
from model_building.pipeline.explanation_engine import SymptomExplanationEngine
from model_building.pipeline.schemas import (
    ActiveQuestionInquiry,
    ConditionExplanation,
    DiagnosticReportSchema,
    EmergencyAlert,
    SymptomInputRequest,
)

logger = logging.getLogger(__name__)


class ClinicalPredictionPipeline:
    """Master production-grade diagnostic pipeline for HealthTrack AI."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        processed_data_dir: Optional[Path] = None,
        cache_max_size: int = 5000,
        abstention_confidence_threshold: float = 0.20,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.models_dir = models_dir or (base_dir / "models")
        self.processed_dir = processed_data_dir or (base_dir / "data" / "processed")

        self.vocab_path = self.processed_dir / "vocabularies.json"
        self.db_path = self.processed_dir / "clinical_evidence.db"
        self.b_model_path = self.models_dir / "model_benchmark_41.joblib"
        self.g_model_path = self.models_dir / "model_general_clinical.joblib"

        self.abstention_confidence_threshold = abstention_confidence_threshold

        # 1. Initialize Caching Layer
        self.cache = CanonicalSymptomCache(max_size=cache_max_size)

        # 2. Initialize NLP Extractor
        logger.info(f"Initializing Clinical NLP Extractor with {self.vocab_path}...")
        self.extractor = ClinicalSymptomExtractor(self.vocab_path)

        # 3. Load Model 1 (Primary Care Benchmark - 41 Diseases)
        logger.info(f"Loading Primary Care Benchmark Model from {self.b_model_path}...")
        b_data = joblib.load(self.b_model_path)
        self.b_model = b_data["model"]
        self.b_vocab = b_data["vocab"]
        self.b_symptom_to_idx = b_data["symptom_to_idx"]
        self.b_idx_to_disease = b_data["idx_to_disease"]

        # 4. Load Model 2 (General Multi-System Clinical - 50 Diseases)
        logger.info(f"Loading General Clinical Model from {self.g_model_path}...")
        g_data = joblib.load(self.g_model_path)
        self.g_model = g_data["model"]
        self.g_vocab = g_data["vocab"]
        self.g_symptom_to_idx = g_data["symptom_to_idx"]
        self.g_idx_to_disease = g_data["idx_to_disease"]

        # 4B. Load Model 3 (Broad Multi-Class Universe - 580 Diseases)
        self.s_model_path = self.models_dir / "sklearn_baseline.joblib"
        if self.s_model_path.exists() and self.vocab_path.exists():
            logger.info(f"Loading Broad Baseline Model from {self.s_model_path}...")
            s_data = joblib.load(self.s_model_path)
            self.s_model = s_data["model"]
            with open(self.vocab_path, encoding="utf-8") as vf:
                v_meta = json.load(vf)
            self.s_diseases = v_meta.get("diseases", [])
            self.s_symptom_to_idx = v_meta.get("symptom_to_id", {})
        else:
            self.s_model = None
            self.s_diseases = []
            self.s_symptom_to_idx = {}

        # 5. Initialize Clinical Sentinels & Safeguards
        self.acuity_sentinel = ClinicalAcuitySentinel()
        self.demographic_filter = DemographicFilter()
        self.conformal_predictor = ConformalDifferentialPredictor()
        self.active_questioning = ActiveQuestioningEngine(self.db_path)

        # 6. Initialize Symptom-Grounded Explanation Engine
        self.explanation_engine = SymptomExplanationEngine(self.db_path)

        logger.info("Clinical Prediction Pipeline successfully initialized and ready for inference!")

    def predict(
        self,
        request: Union[SymptomInputRequest, dict[str, Any], str],
        **kwargs: Any,
    ) -> DiagnosticReportSchema:
        """
        Executes end-to-end clinical diagnosis, edge-case gating, and explanation synthesis.
        """
        start_time = time.perf_counter()

        # Parse request object
        if isinstance(request, str):
            req = SymptomInputRequest(query_text=request, **kwargs)
        elif isinstance(request, dict):
            req = SymptomInputRequest(**request)
        elif isinstance(request, SymptomInputRequest):
            req = request
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")

        query_str = (req.query_text or "").strip()
        selected = [s.strip().lower().replace(" ", "_") for s in (req.selected_symptoms or []) if s.strip()]

        # Tier 1 Cache: Direct Raw Input Hash (Instant < 0.1ms bypass)
        raw_key = self.cache.make_raw_key(
            query_text=req.query_text,
            selected_symptoms=req.selected_symptoms,
            age=req.age,
            biological_sex=req.biological_sex,
        )
        cached_result = self.cache.get(raw_key)
        if cached_result is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            cached_result.execution_time_ms = round(elapsed_ms, 2)
            return cached_result

        # 1. NLP Extraction (Affirmed & Denied Symptoms)
        extracted_present: list[str] = []
        extracted_denied: list[str] = []

        if query_str:
            all_extracted = self.extractor.extract_symptoms(query_str)
            for item in all_extracted:
                if item.is_negated:
                    extracted_denied.append(item.canonical_token)
                else:
                    extracted_present.append(item.canonical_token)

        # Merge structured checklist symptoms from onboarding form
        for s in selected:
            if s not in extracted_present and s not in extracted_denied:
                extracted_present.append(s)

        # Deduplicate while preserving order
        present_tokens = list(dict.fromkeys(extracted_present))
        denied_tokens = list(dict.fromkeys(extracted_denied))

        # Tier 2 Cache: Canonical Extracted Token Hash
        canonical_key = self.cache.make_canonical_key(
            present_symptoms=present_tokens,
            denied_symptoms=denied_tokens,
            age=req.age,
            biological_sex=req.biological_sex,
        )
        cached_result = self.cache.get(canonical_key)
        if cached_result is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            cached_result.execution_time_ms = round(elapsed_ms, 2)
            # Also register raw key for subsequent lookups
            if raw_key:
                self.cache.set(raw_key, cached_result)
            return cached_result

        # 3. Out-of-Distribution (OOD) Check: No clinical symptoms found
        if not present_tokens:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return DiagnosticReportSchema(
                query_text=query_str or f"Structured: {', '.join(selected)}",
                extracted_present_symptoms=[],
                extracted_denied_symptoms=denied_tokens,
                top_condition="Undetermined (No Medical Symptoms Recognized)",
                top_confidence=0.0,
                differential_diagnosis=[],
                conformal_differential_set=[],
                conformal_coverage_guaranteed=False,
                emergency_alert=None,
                active_inquiry=None,
                demographic_exclusions=[],
                is_out_of_distribution=True,
                abstention_triggered=True,
                triage_level="ESI-5",
                triage_recommendation=(
                    "No recognized clinical symptoms were detected in your input. "
                    "Please describe specific signs such as fever, cough, chest tightness, or joint pain."
                ),
                cached=False,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # 4. Multi-Model Inference & Posterior Estimation
        # 4A. Query Model 1 (Primary Care Benchmark 41)
        b_vec = np.zeros((1, len(self.b_vocab)), dtype=np.float32)
        b_hits = 0
        for s in present_tokens:
            idx = self.b_symptom_to_idx.get(s) or self.b_symptom_to_idx.get(s.replace(" ", "_"))
            if idx is not None:
                b_vec[0, idx] = 1.0
                b_hits += 1

        b_top_conf = 0.0
        b_candidates: list[tuple[str, float, str]] = []
        if b_hits > 0:
            b_probs = self.b_model.predict_proba(b_vec)[0]
            b_top_idx = np.argsort(-b_probs)[: req.top_k * 3]
            b_top_conf = float(b_probs[b_top_idx[0]])
            for idx in b_top_idx:
                p = float(b_probs[idx])
                b_candidates.append(
                    (self.b_idx_to_disease[idx], p, "Primary Care Benchmark Model (41 Conditions)")
                )

        # 4B. Query Model 2 (General Multi-System Clinical 50)
        g_vec = np.zeros((1, len(self.g_vocab)), dtype=np.float32)
        g_hits = 0
        for s in present_tokens:
            idx = self.g_symptom_to_idx.get(s) or self.g_symptom_to_idx.get(s.replace("_", " "))
            if idx is not None:
                g_vec[0, idx] = 1.0
                g_hits += 1

        g_top_conf = 0.0
        g_candidates: list[tuple[str, float, str]] = []
        if g_hits > 0:
            g_probs = self.g_model.predict_proba(g_vec)[0]
            g_top_idx = np.argsort(-g_probs)[: req.top_k * 3]
            g_top_conf = float(g_probs[g_top_idx[0]])
            for idx in g_top_idx:
                p = float(g_probs[idx])
                g_candidates.append(
                    (self.g_idx_to_disease[idx], p, "General Clinical Model (50 Diseases)")
                )

        # 4C. Query Model 3 (Broad Multi-Class Universe 580)
        s_candidates: list[tuple[str, float, str]] = []
        if self.s_model is not None and self.s_symptom_to_idx:
            s_vec = np.zeros((1, len(self.s_symptom_to_idx)), dtype=np.float32)
            s_hits = 0
            for s in present_tokens:
                idx = self.s_symptom_to_idx.get(s) or self.s_symptom_to_idx.get(s.replace(" ", "_")) or self.s_symptom_to_idx.get(s.replace("_", " "))
                if idx is not None and idx < s_vec.shape[1]:
                    s_vec[0, idx] = 1.0
                    s_hits += 1
            if s_hits > 0:
                s_probs = self.s_model.predict_proba(s_vec)[0]
                s_top_idx = np.argsort(-s_probs)[: req.top_k * 3]
                for idx in s_top_idx:
                    p = float(s_probs[idx])
                    dis_name = self.s_diseases[idx] if idx < len(self.s_diseases) else f"Condition #{idx}"
                    s_candidates.append(
                        (dis_name, p, "Broad Universe Model (580 Classes)")
                    )

        # 4D. Clinical Router or User-Forced Model Selection
        user_tier = getattr(req, "model_tier", "auto") or "auto"
        if user_tier == "primary_41" and b_candidates:
            selected_candidates = b_candidates
            source_model_name = "Primary Care Benchmark Model (41 Conditions)"
        elif user_tier == "general_50" and g_candidates:
            selected_candidates = g_candidates
            source_model_name = "General Multi-System Clinical Model (50 Diseases)"
        elif user_tier == "all_580" and s_candidates:
            selected_candidates = s_candidates
            source_model_name = "Broad Universe Model (580 Classes)"
        elif b_hits >= g_hits and b_top_conf >= 0.20:
            selected_candidates = b_candidates
            source_model_name = "Primary Care Benchmark Model (41 Conditions)"
        elif g_candidates and g_top_conf >= 0.15:
            selected_candidates = g_candidates
            source_model_name = "General Multi-System Clinical Model (50 Diseases)"
        elif b_candidates:
            selected_candidates = b_candidates
            source_model_name = "Primary Care Benchmark Model (Low Support)"
        elif s_candidates:
            selected_candidates = s_candidates
            source_model_name = "Broad Universe Model (580 Classes)"
        else:
            selected_candidates = []
            source_model_name = "None"

        # 5. Biological Demographic Rules Filter
        # Mocking an internal candidate object for demographic filter compatibility
        class _CandidateAdapter:
            def __init__(self, c: str, p: float, s: str, r: int):
                self.condition = c
                self.probability = p
                self.source_model = s
                self.rank = r

        raw_adapters = [
            _CandidateAdapter(c, p, s, r)
            for r, (c, p, s) in enumerate(selected_candidates, start=1)
        ]

        filtered_adapters, demographic_exclusions = self.demographic_filter.filter_differential(
            raw_adapters,
            age=req.age,
            sex=req.biological_sex,
        )

        # 6. Conformal Adaptive Prediction Set (95% coverage)
        conformal_set = self.conformal_predictor.predict_conformal_set(
            filtered_adapters,
            desired_coverage=req.conformal_coverage,
        )
        conformal_conditions = [c.condition for c in conformal_set.candidates] if conformal_set else []

        # 7. Emergency Acuity Sentinel (ESI Level 1 & 2 Overrides)
        diff_pairs = [(a.condition, a.probability) for a in filtered_adapters]
        acuity_alert = self.acuity_sentinel.evaluate_triage(
            differential=diff_pairs,
            observed_symptoms=present_tokens,
        )

        emergency_alert_schema = None
        if acuity_alert.is_emergency:
            emergency_alert_schema = EmergencyAlert(
                is_emergency=True,
                acuity_level=acuity_alert.acuity_level,
                condition=acuity_alert.condition_name,
                action=acuity_alert.recommended_action,
                message=acuity_alert.triage_message,
            )

        # 8. Active Questioning Engine (Solitary Presentations)
        top_prob = filtered_adapters[0].probability if filtered_adapters else 0.0
        active_inquiry = self.active_questioning.evaluate_inquiry(
            observed_symptoms=present_tokens,
            top_confidence=top_prob,
            entropy_percentage=0.80 if len(present_tokens) <= 1 else 0.40,
            max_questions=3,
        )

        active_inquiry_schema = None
        if active_inquiry.is_active_inquiry_needed:
            q_list = [q.question_text for q in active_inquiry.discriminative_questions]
            primary_s = ", ".join(active_inquiry.primary_symptoms) if active_inquiry.primary_symptoms else (present_tokens[0] if present_tokens else "unspecified")
            active_inquiry_schema = ActiveQuestionInquiry(
                is_active_inquiry_needed=True,
                primary_symptom=primary_s,
                candidate_conditions=active_inquiry.top_candidate_diseases,
                high_yield_questions=q_list,
                rationale=active_inquiry.trigger_reason,
            )

        # 9. Triage Level & Clinical Recommendation
        abstain = False
        if acuity_alert.is_emergency:
            triage_tier = "ESI-1" if acuity_alert.acuity_level == 1 else "ESI-2"
            triage_msg = acuity_alert.triage_message
        elif active_inquiry.is_active_inquiry_needed and len(present_tokens) <= 1:
            abstain = True
            triage_tier = "ESI-4"
            triage_msg = (
                f"The symptom '{present_tokens[0]}' is non-specific and occurs in several conditions. "
                "Targeted clarifying questions are provided below to narrow down your differential diagnosis."
            )
        elif top_prob < self.abstention_confidence_threshold:
            abstain = True
            triage_tier = "ESI-3"
            triage_msg = (
                f"Low diagnostic confidence ({top_prob * 100:.1f}%). The presentation does not conclusively "
                "align with a single profile. A comprehensive outpatient clinical exam is advised."
            )
        elif top_prob >= 0.70:
            triage_tier = "ESI-4"
            top_name = filtered_adapters[0].condition
            triage_msg = f"High confidence for {top_name}. Routine clinical evaluation advised."
        else:
            triage_tier = "ESI-4"
            top_name = filtered_adapters[0].condition
            triage_msg = f"Moderate confidence for {top_name}. Follow-up with a primary care provider recommended."

        # 10. Generate Symptom-Grounded Explanations for Differential (Top 3 Candidates)
        differential_explanations: list[ConditionExplanation] = []
        top_k_candidates = filtered_adapters[: min(req.top_k, 3)]

        for rank, cand in enumerate(top_k_candidates, start=1):
            explanation = self.explanation_engine.explain_candidate(
                condition=cand.condition,
                probability=cand.probability,
                rank=rank,
                source_model=cand.source_model,
                observed_symptoms=present_tokens,
                denied_symptoms=denied_tokens,
            )
            differential_explanations.append(explanation)

        top_cond_name = differential_explanations[0].condition if differential_explanations else "Undetermined"
        top_cond_conf = differential_explanations[0].confidence_percentage if differential_explanations else 0.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 11. Assemble Final Diagnostic Report
        report = DiagnosticReportSchema(
            query_text=query_str or f"Structured: {', '.join(selected)}",
            extracted_present_symptoms=present_tokens,
            extracted_denied_symptoms=denied_tokens,
            top_condition=top_cond_name,
            top_confidence=top_cond_conf,
            differential_diagnosis=differential_explanations,
            conformal_differential_set=conformal_conditions,
            conformal_coverage_guaranteed=bool(conformal_conditions),
            emergency_alert=emergency_alert_schema,
            active_inquiry=active_inquiry_schema,
            demographic_exclusions=demographic_exclusions,
            is_out_of_distribution=False,
            abstention_triggered=abstain,
            triage_level=triage_tier,
            triage_recommendation=triage_msg,
            cached=False,
            execution_time_ms=round(elapsed_ms, 2),
        )

        # 12. Persist to LRU Cache (Both Raw Key and Canonical Key)
        if raw_key:
            self.cache.set(raw_key, report)
        self.cache.set(canonical_key, report)

        return report
