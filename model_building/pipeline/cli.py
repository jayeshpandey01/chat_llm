"""
Interactive CLI for HealthTrack AI Clinical Prediction Pipeline.
================================================================
Usage:
    python -m model_building.pipeline.cli "chest pain and shortness of breath"
    python -m model_building.pipeline.cli --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_building.pipeline.clinical_pipeline import ClinicalPredictionPipeline
from model_building.pipeline.schemas import SymptomInputRequest


def format_report_for_terminal(report) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("HEALTH TRACK AI - CLINICAL DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append(f"Query: {report.query_text}")
    lines.append(f"Present Symptoms Extracted: {', '.join(report.extracted_present_symptoms) if report.extracted_present_symptoms else 'None'}")
    lines.append(f"Denied Symptoms (NegEx):     {', '.join(report.extracted_denied_symptoms) if report.extracted_denied_symptoms else 'None'}")
    lines.append("-" * 70)

    if report.emergency_alert and report.emergency_alert.is_emergency:
        lines.append("🚨 EMERGENCY ACUITY ALERT:")
        lines.append(f"  Level:  ESI-{report.emergency_alert.acuity_level}")
        lines.append(f"  Alert:  {report.emergency_alert.condition}")
        lines.append(f"  Action: {report.emergency_alert.action}")
        lines.append(f"  Message: {report.emergency_alert.message}")
        lines.append("-" * 70)

    lines.append(f"Primary Condition: {report.top_condition} ({report.top_confidence:.1f}% confidence)")
    lines.append(f"Triage Urgency:    {report.triage_level}")
    lines.append(f"Recommendation:    {report.triage_recommendation}")
    lines.append(f"Conformal Set:     {', '.join(report.conformal_differential_set) if report.conformal_differential_set else 'None'}")
    lines.append("-" * 70)

    lines.append("TOP DIFFERENTIAL DIAGNOSES & CLINICAL EXPLANATIONS:")
    for cand in report.differential_diagnosis:
        lines.append(f"\n  #{cand.rank} {cand.condition.upper()} ({cand.confidence_percentage}% confidence | {cand.source_model})")
        lines.append(f"     Matched Symptoms: {', '.join(cand.matched_symptoms) if cand.matched_symptoms else 'None'}")
        if cand.missing_classic_symptoms:
            lines.append(f"     Unreported Signs: {', '.join(cand.missing_classic_symptoms)}")
        lines.append(f"     Explanation:      {cand.explanation_text}")
        if cand.nih_guideline_summary:
            lines.append(f"     NIH Guideline:    {cand.nih_guideline_summary}")

    if report.active_inquiry and report.active_inquiry.is_active_inquiry_needed:
        lines.append("\n" + "-" * 70)
        lines.append("HIGH-YIELD CLINICAL CLARIFYING QUESTIONS:")
        for q in report.active_inquiry.high_yield_questions:
            lines.append(f"  • {q}")

    lines.append("-" * 70)
    lines.append(f"Latency: {report.execution_time_ms} ms | Cached: {report.cached}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="HealthTrack AI Clinical Prediction CLI")
    parser.add_argument("query", nargs="?", type=str, help="Free-text patient symptom query")
    parser.add_argument("--age", type=float, default=None, help="Patient age in years")
    parser.add_argument("--sex", type=str, default=None, choices=["MALE", "FEMALE", "INTERSEX"], help="Patient biological sex")
    parser.add_argument("--symptoms", nargs="*", default=None, help="Structured symptom tokens from onboarding")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive consultation loop")

    args = parser.parse_args()

    pipeline = ClinicalPredictionPipeline()

    if args.interactive:
        print("\nWelcome to HealthTrack AI Clinical Diagnostic Engine (Interactive Mode)")
        print("Type 'exit' or 'quit' to quit.\n")
        while True:
            try:
                user_q = input("\nEnter patient symptoms: ").strip()
                if not user_q or user_q.lower() in ("exit", "quit"):
                    break
                age_input = input("Patient age (optional, press Enter to skip): ").strip()
                age_val = float(age_input) if age_input else None
                sex_input = input("Biological sex (MALE/FEMALE, press Enter to skip): ").strip().upper()
                sex_val = sex_input if sex_input in ("MALE", "FEMALE") else None

                rep = pipeline.predict(SymptomInputRequest(
                    query_text=user_q,
                    age=age_val,
                    biological_sex=sex_val,
                ))
                print(format_report_for_terminal(rep))
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")
        return

    if not args.query and not args.symptoms:
        parser.print_help()
        sys.exit(1)

    req = SymptomInputRequest(
        query_text=args.query,
        selected_symptoms=args.symptoms,
        age=args.age,
        biological_sex=args.sex,
    )

    report = pipeline.predict(req)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_report_for_terminal(report))


if __name__ == "__main__":
    main()

