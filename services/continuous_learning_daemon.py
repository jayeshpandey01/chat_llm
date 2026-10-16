"""Continuous Clinical Learning & Alignment Daemon.

An asynchronous, decoupled background queue consumer that:
1. Ingests raw interaction logs from data/clinical_interaction_buffer.jsonl 1-by-1.
2. Audits tokens using ModelCollapseGuard (entropy H >= 2.50, VDR >= 0.20, n-gram rep <= 0.25).
3. Verifies factual claims against retrieved medical evidence using ClinicalEvidenceVerifier.
4. Enforces the Clinical Dual Reward Engine (R_safety >= 0.40, Invariant Phi_6 Anti-Evasion).
5. Appends verified pairs to data/verified_trajectories_buffer.jsonl.
6. Automatically triggers DPO post-training & statistical release gating when threshold N >= 20 is met.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from core.config import settings
from curation.replay_buffer import ModelCollapseGuard
from training.m5_preference.clinical_dpo_adapter import clinical_dpo_adapter
from training.m5_preference.clinical_dpo_worker import clinical_dpo_worker
from training.m5_preference.clinical_dual_reward import clinical_dual_reward
from training.m5_preference.evidence_verifier import clinical_evidence_verifier
from training.m5_preference.rejection_sampler import clinical_rejection_sampler
from training.m5_preference.schemas import PreferencePairSchema


class ContinuousLearningDaemon:
    """Decoupled background worker for 1-by-1 queue processing and continuous alignment."""

    def __init__(
        self,
        raw_buffer_path: Optional[Path] = None,
        verified_buffer_path: Optional[Path] = None,
        state_file_path: Optional[Path] = None,
        retrain_threshold: int = 20,
        poll_interval_seconds: float = 2.0,
    ):
        self.raw_buffer_path = raw_buffer_path or (settings.DATA_DIR / "clinical_interaction_buffer.jsonl")
        self.verified_buffer_path = verified_buffer_path or (settings.DATA_DIR / "verified_trajectories_buffer.jsonl")
        self.state_file_path = state_file_path or (settings.DATA_DIR / "continuous_learning_daemon_state.json")
        self.retrain_threshold = retrain_threshold
        self.poll_interval_seconds = poll_interval_seconds

        self.collapse_guard = ModelCollapseGuard(
            min_shannon_entropy=2.50,
            min_vocab_diversity=0.20,
            max_ngram_repetition_rate=0.25,
            min_tokens=15,
        )

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Ensure directories
        self.raw_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        self.verified_buffer_path.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Loads persistent daemon state or initializes default."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read daemon state, initializing fresh: {e}")

        default_state = {
            "last_processed_line": 0,
            "total_ingested": 0,
            "total_verified": 0,
            "total_discarded": 0,
            "unprocessed_verified_count": 0,
            "last_dpo_run_timestamp": None,
            "last_dpo_decision": None,
            "active_model_tag": settings.DEFAULT_MODEL,
            "is_running": False,
        }
        self._save_state(default_state)
        return default_state

    def _save_state(self, state: Dict[str, Any]) -> None:
        """Persists daemon state to disk atomically."""
        try:
            temp_path = self.state_file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            temp_path.replace(self.state_file_path)
        except Exception as e:
            logger.error(f"Error persisting daemon state: {e}")

    def process_single_interaction(self, record: Dict[str, Any]) -> Optional[PreferencePairSchema]:
        """Processes one interaction record 1-by-1 through the full verification loop.

        Returns PreferencePairSchema if verified and high-margin, or None if discarded.
        """
        prompt = record.get("prompt", "")
        completion = record.get("assistant_response", "")
        retrieved_evidence = record.get("retrieved_evidence", [])
        top_condition = record.get("top_condition", "Clinical Syndrome")
        triage_level = record.get("triage_level", "ESI-3")
        allergies = record.get("allergies", [])
        chronic = record.get("chronic_conditions", [])
        raw_age = record.get("age")
        age = None
        if raw_age is not None:
            try:
                age = float(raw_age)
            except (ValueError, TypeError):
                age = None
        is_pregnant = bool(record.get("is_pregnant", False))
        demographics = {
            "age": age,
            "is_pregnant": is_pregnant,
            "biological_sex": record.get("biological_sex"),
        }
        is_emergency = "ESI-1" in triage_level or "ESI-2" in triage_level

        if not prompt or not completion:
            return None

        # 1. ModelCollapseGuard entropy & repetition check
        tokens = self.collapse_guard.tokenize_words(completion)
        if len(tokens) >= self.collapse_guard.min_tokens:
            entropy = self.collapse_guard.calculate_entropy(tokens)
            diversity = self.collapse_guard.calculate_vocab_diversity(tokens)
            repetition = self.collapse_guard.calculate_max_ngram_repetition(tokens)
            if (
                entropy < self.collapse_guard.min_shannon_entropy
                or diversity < self.collapse_guard.min_vocab_diversity
                or repetition > self.collapse_guard.max_ngram_repetition_rate
            ):
                logger.info(
                    f"Daemon discarded degenerate record: H={entropy:.2f}, VDR={diversity:.2f}, Rep={repetition:.2f}"
                )
                return None

        # 2. Decompose-then-Verify Evidence Grounding Check
        ev_report = clinical_evidence_verifier.verify_completion(
            response_text=completion,
            retrieved_evidence=retrieved_evidence,
            top_condition=top_condition,
        )
        if not ev_report.is_grounded:
            logger.info(
                f"Daemon discarded ungrounded record: FActScore={ev_report.fact_score:.2f} < 0.70"
            )
            return None

        # 3. Dual Reward Scoring & Anti-Evasion Invariant Phi_6
        breakdown = clinical_dual_reward.evaluate_completion(
            prompt_context=prompt,
            completion=completion,
            allergies=allergies,
            chronic_conditions=chronic,
            age=int(age) if age is not None else None,
            is_pregnant=is_pregnant,
            demographics=demographics,
        )
        if breakdown.safety_score < 0.40:
            logger.info(
                f"Daemon discarded unsafe record: R_safety={breakdown.safety_score:.2f} < 0.40"
            )
            return None

        # 4. Synthesize Contrastive Candidates & Select High-Margin Preference Pair
        candidates = clinical_rejection_sampler.synthesize_rejection_candidates(
            prompt=prompt,
            baseline_response=completion,
            top_condition=top_condition,
            triage_level=triage_level,
            is_emergency=is_emergency,
            allergies=allergies,
            age=age,
        )

        ranking_result = clinical_rejection_sampler.sample_and_rank(
            prompt_context=prompt,
            candidates=candidates,
            allergies=allergies,
            chronic_conditions=chronic,
            age=age,
            is_pregnant=is_pregnant,
            demographics=demographics,
        )

        if ranking_result.preference_pair and ranking_result.preference_pair.margin >= 0.15:
            return ranking_result.preference_pair

        return None

    def process_next_batch(self, max_items: int = 10) -> Dict[str, Any]:
        """Reads unprocessed interactions from the raw buffer 1-by-1.

        Args:
            max_items: Maximum records to consume in this step.

        Returns:
            Dictionary with batch processing statistics.
        """
        with self._lock:
            if not self.raw_buffer_path.exists():
                return {"processed": 0, "verified": 0, "discarded": 0, "status": "NO_RAW_BUFFER"}

            lines: List[str] = []
            with open(self.raw_buffer_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            total_lines = len(lines)
            last_idx = self.state.get("last_processed_line", 0)

            if last_idx >= total_lines:
                return {
                    "processed": 0,
                    "verified": 0,
                    "discarded": 0,
                    "unprocessed_verified_count": self.state.get("unprocessed_verified_count", 0),
                    "status": "UP_TO_DATE",
                }

            items_to_process = lines[last_idx : last_idx + max_items]
            verified_in_batch = 0
            discarded_in_batch = 0

            for line in items_to_process:
                try:
                    record = json.loads(line)
                    pair = self.process_single_interaction(record)
                    if pair:
                        # Append to verified buffer
                        with open(self.verified_buffer_path, "a", encoding="utf-8") as vf:
                            vf.write(json.dumps(pair.model_dump()) + "\n")
                        verified_in_batch += 1
                    else:
                        discarded_in_batch += 1
                except Exception as err:
                    logger.warning(f"Daemon failed parsing buffer line: {err}")
                    discarded_in_batch += 1

            # Update state
            new_last_idx = last_idx + len(items_to_process)
            self.state["last_processed_line"] = new_last_idx
            self.state["total_ingested"] = self.state.get("total_ingested", 0) + len(items_to_process)
            self.state["total_verified"] = self.state.get("total_verified", 0) + verified_in_batch
            self.state["total_discarded"] = self.state.get("total_discarded", 0) + discarded_in_batch
            self.state["unprocessed_verified_count"] = (
                self.state.get("unprocessed_verified_count", 0) + verified_in_batch
            )
            self._save_state(self.state)

            logger.info(
                f"ContinuousLearningDaemon processed {len(items_to_process)} records "
                f"(+{verified_in_batch} verified, +{discarded_in_batch} discarded). "
                f"Queue Verified Count: {self.state['unprocessed_verified_count']}/{self.retrain_threshold}"
            )

            # Check if threshold reached
            dpo_triggered = False
            dpo_result = None
            if self.state["unprocessed_verified_count"] >= self.retrain_threshold:
                logger.info(
                    f"Retraining threshold met ({self.state['unprocessed_verified_count']} >= {self.retrain_threshold}). "
                    f"Triggering DPO Post-Training & Statistical Release Gating..."
                )
                dpo_result = self.trigger_dpo_cycle()
                dpo_triggered = True

            return {
                "processed": len(items_to_process),
                "verified": verified_in_batch,
                "discarded": discarded_in_batch,
                "unprocessed_verified_count": self.state["unprocessed_verified_count"],
                "dpo_triggered": dpo_triggered,
                "dpo_result": dpo_result,
                "status": "PROCESSED",
            }

    def trigger_dpo_cycle(self, force_deploy: bool = False) -> Dict[str, Any]:
        """Exports 70/30 preference dataset and executes DPO training with statistical release gating."""
        try:
            # 1. Synthesize 70/30 stratified dataset (105 Anchors / 45 User Trajectories)
            dataset_file = clinical_dpo_adapter.export_preference_dataset(
                target_total_pairs=150,
            )

            # 2. Run DPO training & release gating
            worker = clinical_dpo_worker
            result = worker.train_and_gate_candidate(force_deploy=force_deploy)

            # 3. Update persistent state
            self.state["last_dpo_run_timestamp"] = datetime.now(timezone.utc).isoformat()
            self.state["last_dpo_decision"] = result.get("decision")
            if result.get("status") == "APPROVED" and result.get("ollama_model"):
                self.state["active_model_tag"] = result.get("ollama_model")
            self.state["unprocessed_verified_count"] = 0
            self._save_state(self.state)

            return result
        except Exception as e:
            logger.error(f"DPO retraining cycle failed in daemon: {e}")
            return {"status": "ERROR", "error": str(e)}

    def start(self) -> None:
        """Starts the background monitoring loop in a separate thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ContinuousLearningDaemon")
            self._thread.start()
            self.state["is_running"] = True
            self._save_state(self.state)
            logger.info("ContinuousLearningDaemon background thread started.")

    def stop(self) -> None:
        """Stops the background monitoring loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self.state["is_running"] = False
            self._save_state(self.state)
            logger.info("ContinuousLearningDaemon background thread stopped.")

    def _run_loop(self) -> None:
        """Internal daemon polling loop."""
        while not self._stop_event.is_set():
            try:
                self.process_next_batch(max_items=5)
            except Exception as err:
                logger.error(f"Error in continuous learning daemon loop: {err}")
            self._stop_event.wait(self.poll_interval_seconds)

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status telemetry."""
        with self._lock:
            raw_count = 0
            if self.raw_buffer_path.exists():
                try:
                    with open(self.raw_buffer_path, "r", encoding="utf-8") as f:
                        raw_count = sum(1 for line in f if line.strip())
                except Exception:
                    pass

            return {
                "is_running": (self._thread is not None and self._thread.is_alive()),
                "raw_buffer_total_lines": raw_count,
                "last_processed_line": self.state.get("last_processed_line", 0),
                "total_ingested": self.state.get("total_ingested", 0),
                "total_verified": self.state.get("total_verified", 0),
                "total_discarded": self.state.get("total_discarded", 0),
                "unprocessed_verified_count": self.state.get("unprocessed_verified_count", 0),
                "retrain_threshold": self.retrain_threshold,
                "active_model_tag": self.state.get("active_model_tag", settings.DEFAULT_MODEL),
                "last_dpo_run_timestamp": self.state.get("last_dpo_run_timestamp"),
                "last_dpo_decision": self.state.get("last_dpo_decision"),
            }


# Singleton daemon instance
continuous_learning_daemon = ContinuousLearningDaemon()
