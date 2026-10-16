"""Clinical Rejection Sampling and Preference Pair Synthesizer.

Implements RLHF Step 3 Rejection Sampling from the Llama 2 / DeepSeek architecture:
1. Samples or evaluates K candidate completions for a clinical context.
2. Scores candidates using the ClinicalDualRewardEngine (Safety + Helpfulness).
3. Selects top candidate y* (Rejection Sampling SFT target).
4. Generates high-margin (y_chosen, y_rejected) pairs for DPO / PPO fine-tuning.
5. Ingests user interactions from the clinical replay buffer into 70/30 stratified batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.schemas import PreferencePairSchema
from training.m5_preference.clinical_dual_reward import (
    ClinicalDualRewardEngine,
    RewardBreakdown,
    clinical_dual_reward,
)


class CandidateEvaluation(BaseModel):
    """Candidate completion with associated dual reward score."""

    candidate_text: str
    reward_breakdown: RewardBreakdown
    composite_reward: float


class RejectionSamplingResult(BaseModel):
    """Result of clinical rejection sampling over K candidates."""

    prompt: str
    best_candidate: str
    best_reward: float
    worst_candidate: str
    worst_reward: float
    margin: float
    all_evaluations: List[CandidateEvaluation]
    preference_pair: Optional[PreferencePairSchema] = None


class ClinicalRejectionSampler:
    """Orchestrates candidate scoring, ranking, and preference synthesis."""

    def __init__(self, reward_engine: Optional[ClinicalDualRewardEngine] = None):
        self.reward_engine = reward_engine or clinical_dual_reward

    def sample_and_rank(
        self,
        prompt_context: str,
        candidates: List[str],
        allergies: Optional[List[str]] = None,
        chronic_conditions: Optional[List[str]] = None,
        evidence_topics: Optional[List[str]] = None,
        min_acceptable_reward: float = 0.50,
        age: Optional[int] = None,
        is_pregnant: bool = False,
        demographics: Optional[Dict[str, Any]] = None,
    ) -> RejectionSamplingResult:
        """Evaluates K candidates, selects top y*, and constructs preference pair."""
        if not candidates:
            raise ValueError("Candidates list must contain at least 1 completion.")

        evaluations: List[CandidateEvaluation] = []
        for cand in candidates:
            breakdown = self.reward_engine.evaluate_completion(
                prompt_context=prompt_context,
                completion=cand,
                allergies=allergies,
                chronic_conditions=chronic_conditions,
                evidence_topics=evidence_topics,
                age=age,
                is_pregnant=is_pregnant,
                demographics=demographics,
            )
            evaluations.append(
                CandidateEvaluation(
                    candidate_text=cand,
                    reward_breakdown=breakdown,
                    composite_reward=breakdown.composite_reward,
                )
            )

        # Sort descending by composite reward
        evaluations.sort(key=lambda x: x.composite_reward, reverse=True)

        best = evaluations[0]
        worst = evaluations[-1]
        margin = round(best.composite_reward - worst.composite_reward, 4)

        pref_pair = None
        # Enforce statutory safety on chosen candidate (Theorem 1: R_safety(y*) >= 0.40)
        # and clinically meaningful contrastive margin (Delta R >= 0.15)
        if best.reward_breakdown.safety_score >= 0.40 and margin >= 0.15:
            pref_pair = PreferencePairSchema(
                prompt=prompt_context,
                chosen=best.candidate_text,
                rejected=worst.candidate_text,
                margin=margin,
                category="clinical_symptom_checker",
                metadata={
                    "reward_chosen": best.composite_reward,
                    "reward_rejected": worst.composite_reward,
                    "safety_chosen": best.reward_breakdown.safety_score,
                    "safety_rejected": worst.reward_breakdown.safety_score,
                },
            )

        return RejectionSamplingResult(
            prompt=prompt_context,
            best_candidate=best.candidate_text,
            best_reward=best.composite_reward,
            worst_candidate=worst.candidate_text,
            worst_reward=worst.composite_reward,
            margin=margin,
            all_evaluations=evaluations,
            preference_pair=pref_pair,
        )

    def sample_best_of_k_ollama(
        self,
        prompt_context: str,
        k: int = 3,
        temperatures: Optional[List[float]] = None,
        model: Optional[str] = None,
        allergies: Optional[List[str]] = None,
        chronic_conditions: Optional[List[str]] = None,
        evidence_topics: Optional[List[str]] = None,
        age: Optional[int] = None,
        is_pregnant: bool = False,
        demographics: Optional[Dict[str, Any]] = None,
    ) -> RejectionSamplingResult:
        """Samples K candidates live from Ollama daemon with varying temperatures and ranks them."""
        import httpx
        from core.config import settings

        temperatures = temperatures or [0.2, 0.6, 0.9]
        model_name = model or settings.DEFAULT_MODEL
        candidates: List[str] = []

        try:
            with httpx.Client(timeout=10.0) as client:
                chat_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are HealthTrack Clinical AI, an expert Telehealth Attending Physician delivering structured clinical decision support.\n"
                            "Communicate with clinical authority, diagnostic rigor, and bedside empathy.\n"
                            "MANDATORY: Structure as: ### Clinical Impression, ### Differential Diagnosis, ### Targeted Clinical Inquiries, ### Supportive Care, ### Red Flag Warning Sentinels.\n"
                            "Conclude with a statutory CDSS notice."
                        ),
                    },
                    {"role": "user", "content": prompt_context},
                ]

                for temp in temperatures[:k]:
                    # 1. Try native ChatML endpoint
                    try:
                        chat_resp = client.post(
                            f"{settings.OLLAMA_BASE_URL}/api/chat",
                            json={
                                "model": model_name,
                                "messages": chat_messages,
                                "stream": False,
                                "options": {"temperature": temp, "top_p": 0.9},
                            },
                        )
                        if chat_resp.status_code == 200:
                            txt = chat_resp.json().get("message", {}).get("content", "").strip()
                            if txt:
                                candidates.append(txt)
                                continue
                    except Exception:
                        pass

                    # 2. Fallback to /api/generate endpoint
                    try:
                        resp = client.post(
                            f"{settings.OLLAMA_BASE_URL}/api/generate",
                            json={
                                "model": model_name,
                                "prompt": prompt_context,
                                "stream": False,
                                "options": {"temperature": temp},
                            },
                        )
                        if resp.status_code == 200:
                            txt = resp.json().get("response", "").strip()
                            if txt:
                                candidates.append(txt)
                    except Exception:
                        pass
        except Exception:
            pass

        if len(candidates) < 2:
            candidates = self.synthesize_rejection_candidates(
                prompt=prompt_context,
                baseline_response=candidates[0] if candidates else "Please consult a healthcare provider.",
                top_condition="Clinical Condition",
                triage_level="ESI-3",
                allergies=allergies,
                age=age,
            )

        return self.sample_and_rank(
            prompt_context=prompt_context,
            candidates=candidates,
            allergies=allergies,
            chronic_conditions=chronic_conditions,
            evidence_topics=evidence_topics,
            age=age,
            is_pregnant=is_pregnant,
            demographics=demographics,
        )

    def synthesize_rejection_candidates(
        self,
        prompt: str,
        baseline_response: str,
        top_condition: str,
        triage_level: str,
        is_emergency: bool = False,
        allergies: Optional[List[str]] = None,
        age: Optional[int] = None,
    ) -> List[str]:
        """Generates synthetic contrastive candidates for an interaction trajectory.

        Creates:
        1. Fully calibrated, grounded response (chosen candidate).
        2. Uncalibrated response missing disclaimer or active inquiry.
        3. Sycophantic / premature diagnostic closure response (negative candidate).
        4. (If emergency) Non-urgent response missing 911 directive (critical failure).
        5. (If allergy) Contraindicated drug recommendation.
        6. (If pediatric) Pediatric aspirin toxicity recommendation.
        7. (If viral + antibiotic request) Inappropriate antibiotic compliance.
        """
        candidates = [baseline_response]

        # Candidate 2: Sycophantic premature closure
        sycophantic = (
            f"You definitely have {top_condition}. I can confirm this diagnosis 100%. "
            "There is no need to visit a doctor or check with a hospital."
        )
        candidates.append(sycophantic)

        # Candidate 3: Missing disclaimer and inquiry
        abrupt = (
            f"Your condition might be {top_condition}. Triage: {triage_level}."
        )
        candidates.append(abrupt)

        # Candidate 4: If emergency, a dangerous non-urgent candidate
        if is_emergency:
            dangerous = (
                f"Your symptoms could just be stress or a mild chest muscle strain. "
                "Drink some tea, take a nap, and call a clinic next week if it doesn't get better."
            )
            candidates.append(dangerous)

        # Candidate 5: Contraindicated allergy recommendation
        if allergies and any("penicillin" in a.lower() for a in allergies):
            contraindicated = (
                f"You probably have a bacterial infection like {top_condition}. "
                "You should start taking amoxicillin or augmentin tablets immediately."
            )
            candidates.append(contraindicated)

        # Candidate 6: Pediatric Aspirin / Reye's Hazard
        numeric_age = None
        if age is not None:
            try:
                numeric_age = float(age)
            except (ValueError, TypeError):
                numeric_age = None

        if numeric_age is not None and numeric_age < 19:
            pediatric_hazard = (
                f"For fever and body aches associated with {top_condition}, give aspirin tablets "
                "or Pepto-Bismol to reduce fever quickly."
            )
            candidates.append(pediatric_hazard)

        # Candidate 7: Antimicrobial stewardship failure
        if any(v in prompt.lower() for v in ["cold", "runny nose", "congestion", "viral"]) and "antibiotic" in prompt.lower():
            steward_failure = (
                f"I understand your urgency. I will validate your request for antibiotics: take azithromycin (Z-pack) "
                f"immediately to cure this infection."
            )
            candidates.append(steward_failure)

        return candidates


# Shared singleton instance
clinical_rejection_sampler = ClinicalRejectionSampler()
