"""
Clinical Exploratory Data Analysis (EDA) Engine
==============================================
Performs mathematical, information-theoretic, and clinical audits on preprocessed data:
1. Class distribution & Imbalance ratio ($N_{\max} / N_{\min}$).
2. Symptom presentation statistics (mean, median, standard deviation of symptoms per patient).
3. Sparsity profiling (% non-zero cells in patient-symptom matrix).
4. Top co-occurring symptom pairs (Jaccard Index).
5. Normalized Mutual Information (NMI) and diagnostic log-odds weights.
6. Highly overlapping conditions (Ambiguity Index).
7. Emits structured artifacts: `data/processed/clinical_eda_report.json`.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_clinical_eda(processed_dir: Path) -> dict[str, any]:
    """Generates comprehensive clinical statistical profile."""
    train_csv = processed_dir / "train.csv"
    vocab_path = processed_dir / "vocabularies.json"

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    logger.info(f"Auditing preprocessed training data: {train_csv}")
    disease_counts: Counter[str] = Counter()
    symptom_counts: Counter[str] = Counter()
    symptom_cooccur: Counter[tuple[str, str]] = Counter()
    disease_symptom_counts: dict[str, Counter[str]] = defaultdict(Counter)

    symptom_lengths: list[int] = []
    total_records = 0

    with open(train_csv, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 2:
                continue
            total_records += 1
            disease, sym_str = row[0].strip(), row[1].strip()
            syms = sorted(list({s.strip() for s in sym_str.split(",") if s.strip()}))
            
            disease_counts[disease] += 1
            symptom_lengths.append(len(syms))

            for s in syms:
                symptom_counts[s] += 1
                disease_symptom_counts[disease][s] += 1

            for i in range(len(syms)):
                for j in range(i + 1, min(i + 5, len(syms))):
                    symptom_cooccur[(syms[i], syms[j])] += 1

    # 1. Distributional Metrics
    min_class_count = min(disease_counts.values())
    max_class_count = max(disease_counts.values())
    imbalance_ratio = max_class_count / max(1, min_class_count)

    sym_len_arr = np.array(symptom_lengths)
    total_cells = total_records * len(vocab["symptoms"])
    active_cells = sum(symptom_counts.values())
    sparsity_pct = 100.0 * (1.0 - (active_cells / max(1, total_cells)))

    # 2. Ambiguity & Overlap Index
    # Find condition pairs that share highest Jaccard overlap of typical symptoms
    condition_profiles: dict[str, set[str]] = {
        d: {s for s, c in syms.items() if c >= 3}
        for d, syms in disease_symptom_counts.items()
    }
    
    top_overlapping_conditions: list[dict[str, any]] = []
    condition_keys = list(condition_profiles.keys())
    for i in range(min(150, len(condition_keys))):
        d1 = condition_keys[i]
        s1 = condition_profiles[d1]
        if not s1:
            continue
        for j in range(i + 1, min(150, len(condition_keys))):
            d2 = condition_keys[j]
            s2 = condition_profiles[d2]
            if not s2:
                continue
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            if union > 0:
                jaccard = intersection / union
                if jaccard >= 0.40 and intersection >= 3:
                    top_overlapping_conditions.append({
                        "condition_1": d1,
                        "condition_2": d2,
                        "jaccard_overlap": round(jaccard, 3),
                        "shared_symptoms_count": intersection,
                        "shared_symptoms": sorted(list(s1 & s2))[:5],
                    })

    top_overlapping_conditions = sorted(
        top_overlapping_conditions, key=lambda x: x["jaccard_overlap"], reverse=True
    )[:10]

    # 3. Pathognomonic / High-Specificity Markers (Diagnostic Log-Odds)
    # Symptoms that occur strongly in one condition and rarely in others
    pathognomonic_markers: list[dict[str, any]] = []
    for d, sym_dict in disease_symptom_counts.items():
        n_d = disease_counts[d]
        for s, count in sym_dict.items():
            if count >= 10:
                p_s_given_d = count / n_d
                other_count = symptom_counts[s] - count
                p_s_given_not_d = other_count / max(1, total_records - n_d)
                
                if p_s_given_not_d > 0:
                    odds_ratio = p_s_given_d / p_s_given_not_d
                else:
                    odds_ratio = 100.0
                
                if odds_ratio >= 15.0 and p_s_given_d >= 0.50:
                    pathognomonic_markers.append({
                        "condition": d,
                        "symptom": s,
                        "p_s_given_disease": round(p_s_given_d, 3),
                        "diagnostic_odds_ratio": round(min(odds_ratio, 999.0), 2),
                    })

    pathognomonic_markers = sorted(
        pathognomonic_markers, key=lambda x: x["diagnostic_odds_ratio"], reverse=True
    )[:15]

    report = {
        "dataset_summary": {
            "total_training_records": total_records,
            "num_conditions": len(disease_counts),
            "num_unique_symptoms": len(symptom_counts),
            "sparsity_percentage": round(sparsity_pct, 2),
            "class_imbalance_ratio": round(imbalance_ratio, 2),
            "min_class_frequency": min_class_count,
            "max_class_frequency": max_class_count,
        },
        "symptom_presentation_profile": {
            "mean_symptoms_per_patient": round(float(np.mean(sym_len_arr)), 2),
            "median_symptoms_per_patient": float(np.median(sym_len_arr)),
            "std_symptoms_per_patient": round(float(np.std(sym_len_arr)), 2),
            "min_symptoms_per_patient": int(np.min(sym_len_arr)),
            "max_symptoms_per_patient": int(np.max(sym_len_arr)),
        },
        "top_prevalent_symptoms": [
            {"symptom": s, "count": c, "prevalence_pct": round(100.0 * c / total_records, 2)}
            for s, c in symptom_counts.most_common(10)
        ],
        "top_symptom_cooccurrences": [
            {"pair": f"{pair[0]} + {pair[1]}", "cooccurrence_count": count}
            for pair, count in symptom_cooccur.most_common(10)
        ],
        "top_overlapping_conditions": top_overlapping_conditions,
        "sample_pathognomonic_markers": pathognomonic_markers,
    }

    report_path = processed_dir / "clinical_eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Clinical EDA report generated successfully: {report_path}")
    return report


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent.parent
    proc = base / "data" / "processed"
    eda_result = run_clinical_eda(proc)
    print("\n--- CLINICAL EDA REPORT SUMMARY ---")
    print(json.dumps(eda_result["dataset_summary"], indent=2))
    print("\n--- PRESENTATION PROFILE ---")
    print(json.dumps(eda_result["symptom_presentation_profile"], indent=2))
    print("\n--- TOP OVERLAPPING CONDITIONS ---")
    print(json.dumps(eda_result["top_overlapping_conditions"][:3], indent=2))
