#!/usr/bin/env python3
"""Run Continuous Clinical DPO Post-Training, Benchmark Gating, and Model Promotion.

Execution command:
    python scripts/run_continuous_dpo.py --epochs 2 --lr 5e-5 --beta 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from training.m5_preference.clinical_dpo_adapter import clinical_dpo_adapter
from training.m5_preference.clinical_dpo_worker import ClinicalDPOWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Continuous Clinical DPO Post-Training on the 70/30 Stratified Preference Dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of DPO training epochs (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="AdamW learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO loss beta KL penalty parameter (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro-batch size for preference training (default: 4)",
    )
    parser.add_argument(
        "--export-dataset",
        action="store_true",
        default=True,
        help="Synthesize and export data/clinical_preference_dataset.jsonl before training (default: True)",
    )
    parser.add_argument(
        "--force-deploy",
        action="store_true",
        default=False,
        help="Force promotion even if statistical release gate flags regression (default: False)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=settings.DATA_DIR / "clinical_preference_dataset.jsonl",
        help="Custom path to preference dataset JSONL",
    )
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()

    print("================================================================================")
    print(" [CDSS] HealthTrack AI: Continuous Clinical DPO Post-Training Pipeline")
    print("================================================================================")
    print(f" Dataset Path:      {args.dataset_path}")
    print(f" Epochs:            {args.epochs}")
    print(f" Learning Rate:     {args.lr}")
    print(f" DPO Beta (KL):     {args.beta}")
    print(f" Batch Size:        {args.batch_size}")
    print("================================================================================")

    # 1. Export or refresh dataset if requested
    if args.export_dataset or not args.dataset_path.exists():
        print("\n[1/3] Synthesizing 70/30 Stratified Preference Dataset (105 Anchors / 45 User Trajectories)...")
        dataset_file = clinical_dpo_adapter.export_preference_dataset(
            target_path=args.dataset_path,
            target_total_pairs=150,
        )
        print(f"      [OK] Exported 150 preference pairs to: {dataset_file}")
    else:
        print(f"\n[1/3] Using existing preference dataset at: {args.dataset_path}")

    # 2. Instantiate Clinical DPO Worker
    print("\n[2/3] Initializing Clinical DPO Worker & Detecting Hardware Strategy...")
    worker = ClinicalDPOWorker(
        dataset_path=args.dataset_path,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"      [OK] Hardware Engine: {worker.device_strategy.description}")

    # 3. Execute Post-Training & Statistical Release Gating
    print("\n[3/3] Executing DPO Policy Optimization & Milestone 4 Statistical Release Gate...")
    result = worker.train_and_gate_candidate(
        force_deploy=args.force_deploy,
    )

    print("\n================================================================================")
    print(" CONTINUOUS POST-TRAINING EXECUTION REPORT")
    print("================================================================================")
    print(f" Status:               {result.get('status')}")
    print(f" Gating Decision:      {result.get('decision')}")
    print(f" Candidate Run ID:     {result.get('run_id')}")
    print(f" Checkpoint Path:      {result.get('checkpoint')}")
    print(f" Gate Audit Report:    {result.get('gate_report')}")
    print(f" Active Serving Tag:   {result.get('ollama_model')}")
    print(f" Bootstrap 95% CI:     {result.get('bootstrap_ci')}")
    print(f" Mean Delta vs Base:   {result.get('mean_delta')}")
    print(f" Final DPO Loss:       {result.get('final_loss')}")
    print(f" Execution Duration:   {result.get('execution_seconds')} seconds")
    print("================================================================================")

    if result.get("status") == "APPROVED":
        print("\n[PASS] SUCCESS: Candidate policy cleared all statistical gates and was promoted to serving!")
        return 0
    else:
        print(f"\n[BLOCKED] Candidate policy blocked from deployment ({result.get('reason')}). Active model retained.")
        return 1 if not args.force_deploy else 0


if __name__ == "__main__":
    sys.exit(main())
