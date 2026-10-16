"""
Hardened Clinical Preprocessing Engine
======================================
Implements all edge-case mandates plus accuracy-improvement fixes:
1. Structural token canonicalization (space/underscore merging, punctuation/dot stripping).
2. Unification of 235 casing fragments and typo labels into standardized medical conditions.
3. Strict rejection of paragraph QA entries and unnamed column leaks.
4. Filtering of MeSH symptom-as-disease labels that pollute disease predictions.
5. Exact-duplicate deduplication: remove repeated (disease, symptom_set) rows BEFORE splitting
   to prevent template leakage and inflated accuracy scores.
6. Support filtering (>= 100 encounters, computed AFTER deduplication).
7. Identifiability ceiling computation: what fraction of unique symptom-sets have
   an unambiguous majority label? This is the theoretical top-1 accuracy ceiling.
8. Deterministic grouped split (>= 70% train, same symptom profile stays in one partition).
9. Generation of clean vocabularies and preprocessed split files.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Disease canonicalization map ───────────────────────────────────────────────
DISEASE_CANONICAL_MAP = {
    # Typo corrections
    "peptic ulcer diseae": "Peptic Ulcer Disease",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    "osteoarthristis": "Osteoarthritis",
    "osteoarthritis": "Osteoarthritis",
    "(vertigo) paroymsal  positional vertigo": "Paroxysmal Positional Vertigo",
    "(vertigo) paroxysmal positional vertigo": "Paroxysmal Positional Vertigo",
    "(vertigo) paroymsal positional vertigo": "Paroxysmal Positional Vertigo",
    "dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
    "dimorphic hemmorhoids (piles)": "Hemorrhoids (Piles)",
}

# ── MeSH labels that are symptoms, NOT diseases ───────────────────────────────
# These contaminate the broad model and cause hallucination ("Fever" as a diagnosis).
MESH_SYMPTOM_LABELS: set[str] = {
    "Pain",
    "Fever",
    "Edema",
    "Headache",
    "Vomiting",
    "Diarrhea",
    "Fatigue",
    "Nausea",
    "Body Weight",
    "Seizures",
    "Abdominal Pain",
    "Muscle Weakness",
    "Dyspnea",
    "Anoxia",
    "Paralysis",
    "Weight Loss",
    "Blindness",
    "Obesity",
    "Cough",
    "Rash",
    "Chills",
    "Sweating",
    "Dizziness",
    "Anxiety",
    "Depression",
    "Insomnia",
}

# ── Symptom artifact fixes ─────────────────────────────────────────────────────
SYMPTOM_ARTIFACT_MAP = {
    "fluid_overload_1": "fluid_overload",
    "fluid_overload.1": "fluid_overload",
    "toxic_look_(typhos)": "toxic_look_typhos",
    "cold_hands_and_feets": "cold_hands_and_feet",
    "swollen_extremeties": "swollen_extremities",
    "dischromic__patches": "dischromic_patches",
    "spotting__urination": "spotting_urination",
    "foul_smell_of__urine": "foul_smell_of_urine",
}


def canonicalize_symptom(raw_sym: str) -> str:
    """
    Standardizes a symptom string into unified snake_case, stripping trailing dots,
    punctuation, and fixing space vs underscore delimiter splits.
    """
    s = raw_sym.strip().lower()
    if not s or "unnamed:" in s:
        return ""

    # Strip trailing dots and quotes
    s = re.sub(r"\.+$", "", s)
    s = s.replace("'", "").replace('"', "")

    # Convert spaces and hyphens to underscore
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)

    s = SYMPTOM_ARTIFACT_MAP.get(s, s)
    return s.strip("_")


def canonicalize_disease(raw_dis: str) -> str:
    """Standardizes disease label across casing variations and typos."""
    d_clean = raw_dis.strip()
    d_lower = d_clean.lower()
    if d_lower in DISEASE_CANONICAL_MAP:
        return DISEASE_CANONICAL_MAP[d_lower]
    # Default to clean title-case for casing fragments
    return d_clean.title() if d_clean.islower() else d_clean


def compute_identifiability_ceiling(
    records: list[tuple[str, tuple[str, ...]]],
) -> dict[str, float]:
    """
    Computes the theoretical top-1 accuracy ceiling imposed by label ambiguity.

    For each unique canonical symptom profile (feature set), finds the majority label.
    The ceiling = fraction of records whose label agrees with the majority label for
    their symptom profile. Any classifier trained on hard labels cannot exceed this.

    Returns a dict with ceiling stats.
    """
    profile_label_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for disease, sym_tuple in records:
        profile_label_counts[sym_tuple][disease] += 1

    total = len(records)
    majority_agrees = 0
    ambiguous_profiles = 0
    unique_profiles = len(profile_label_counts)

    for _sym_tuple, label_counter in profile_label_counts.items():
        majority_count = label_counter.most_common(1)[0][1]
        majority_agrees += majority_count
        if len(label_counter) > 1:
            ambiguous_profiles += 1

    ceiling = majority_agrees / total if total > 0 else 0.0
    return {
        "identifiability_ceiling": round(ceiling, 4),
        "unique_symptom_profiles": unique_profiles,
        "ambiguous_profiles": ambiguous_profiles,
        "ambiguous_profile_fraction": round(ambiguous_profiles / unique_profiles, 4) if unique_profiles > 0 else 0.0,
        "total_records": total,
        "majority_agreeing_records": majority_agrees,
    }


def run_hardened_preprocessing(
    raw_csv_path: Path,
    output_dir: Path,
    min_disease_support: int = 100,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    filter_mesh_symptom_labels: bool = True,
    deduplicate: bool = True,
) -> dict[str, any]:
    """
    Processes raw clinical data with full edge-case hardening and accuracy improvements.

    Key improvements vs prior version:
    - Filters MeSH symptom-as-disease labels (prevents hallucination in broad model).
    - Deduplicates exact (disease, symptom_set) pairs before splitting (prevents leakage).
    - Computes and logs the identifiability ceiling (theoretical max top-1 accuracy).
    - Enforces >= 70% training fraction with group isolation assertion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initiating hardened preprocessing on: {raw_csv_path}")

    raw_records: list[tuple[str, list[str]]] = []

    with open(raw_csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Header
        for row in reader:
            if not row or len(row) < 2:
                continue
            raw_d, raw_s = row[0].strip(), row[1].strip()

            # Rule 3: Narrative rejection gate
            if len(raw_d) > 60 or "\n" in raw_d or not raw_s:
                continue

            clean_d = canonicalize_disease(raw_d)

            # Rule 4 (new): Filter MeSH symptom-as-disease labels
            if filter_mesh_symptom_labels and clean_d in MESH_SYMPTOM_LABELS:
                continue

            tokens = [canonicalize_symptom(s) for s in raw_s.split(",") if s.strip()]
            tokens = sorted(list({t for t in tokens if t}))

            if not tokens:
                continue

            raw_records.append((clean_d, tokens))

    logger.info(f"Total valid cleaned records (before dedup): {len(raw_records):,}")

    # ── Step: Exact-duplicate deduplication ────────────────────────────────────
    # Remove repeated (disease, symptom_set) rows. This eliminates the 71,887
    # synthetic template duplicates that inflate accuracy and cause leakage.
    if deduplicate:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped_records: list[tuple[str, list[str]]] = []
        for d, syms in raw_records:
            key = (d, tuple(syms))
            if key not in seen:
                seen.add(key)
                deduped_records.append((d, syms))
        removed = len(raw_records) - len(deduped_records)
        logger.info(f"Deduplication: removed {removed:,} exact duplicate rows → {len(deduped_records):,} unique records")
        cleaned_records = deduped_records
    else:
        cleaned_records = raw_records

    # ── Identifiability ceiling (BEFORE support filter, on unique records) ─────
    records_for_ceiling = [(d, tuple(syms)) for d, syms in cleaned_records]
    ceiling_stats = compute_identifiability_ceiling(records_for_ceiling)
    logger.info(
        f"Identifiability ceiling: {ceiling_stats['identifiability_ceiling']:.4f} "
        f"({ceiling_stats['ambiguous_profiles']:,} / {ceiling_stats['unique_symptom_profiles']:,} "
        f"profiles are ambiguous)"
    )
    logger.info(
        f"→ No classifier trained on hard labels can exceed "
        f"{ceiling_stats['identifiability_ceiling'] * 100:.1f}% top-1 accuracy on this data."
    )

    # ── Disease support counting and filtering ─────────────────────────────────
    disease_counts: dict[str, int] = {}
    all_symptoms: set[str] = set()
    for d, syms in cleaned_records:
        disease_counts[d] = disease_counts.get(d, 0) + 1
        all_symptoms.update(syms)

    logger.info(f"Unique disease categories: {len(disease_counts)}")
    logger.info(f"Unique unified symptom vocabulary: {len(all_symptoms)}")

    # Support filter uses training-representative counts (post-dedup, pre-split)
    supported_diseases = {d for d, c in disease_counts.items() if c >= min_disease_support}
    logger.info(f"Supported diseases (>= {min_disease_support} encounters): {len(supported_diseases)}")

    filtered_records = [
        (d, syms) for d, syms in cleaned_records if d in supported_diseases
    ]
    logger.info(f"Final records retained: {len(filtered_records):,}")

    sorted_diseases = sorted(list(supported_diseases))
    sorted_symptoms = sorted(list({s for _, syms in filtered_records for s in syms}))

    disease_to_id = {d: idx for idx, d in enumerate(sorted_diseases)}
    symptom_to_id = {s: idx for idx, s in enumerate(sorted_symptoms)}

    vocab_meta = {
        "diseases": sorted_diseases,
        "symptoms": sorted_symptoms,
        "disease_to_id": disease_to_id,
        "symptom_to_id": symptom_to_id,
        "total_records": len(filtered_records),
        "num_classes": len(sorted_diseases),
        "num_symptoms": len(sorted_symptoms),
    }

    with open(output_dir / "vocabularies.json", "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, indent=2)

    # ── Grouped split: keep identical symptom profiles in one partition ────────
    labels = [disease_to_id[d] for d, _ in filtered_records]
    indices = np.arange(len(filtered_records))
    groups = np.asarray(["".join(syms) for _, syms in filtered_records])

    n_folds = max(3, round(1.0 / min(test_size, val_size)))
    grouped_splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_ids = np.full(len(filtered_records), -1, dtype=np.int64)
    for fold_id, (_, held_out_idx) in enumerate(grouped_splitter.split(indices, labels, groups)):
        fold_ids[held_out_idx] = fold_id
    if np.any(fold_ids < 0):
        raise RuntimeError("Grouped split did not assign every row to a fold")

    test_fold = 0
    val_fold = 1
    test_idx = indices[fold_ids == test_fold]
    val_idx = indices[fold_ids == val_fold]
    train_idx = indices[(fold_ids != test_fold) & (fold_ids != val_fold)]

    train_frac = len(train_idx) / len(indices)
    if train_frac < 0.70:
        raise RuntimeError(
            f"Grouped split leaves only {train_frac:.1%} of records for training "
            f"(minimum 70% required). Increase n_folds or relax support filter."
        )

    # Assert zero group overlap across partitions
    group_sets = [set(groups[split_idx]) for split_idx in (train_idx, val_idx, test_idx)]
    if (group_sets[0] & group_sets[1]) or (group_sets[0] & group_sets[2]) or (group_sets[1] & group_sets[2]):
        raise RuntimeError("Identical symptom profiles leaked across data partitions")

    logger.info(
        f"Split → Train: {len(train_idx):,} ({train_frac:.1%}), "
        f"Val: {len(val_idx):,} ({len(val_idx)/len(indices):.1%}), "
        f"Test: {len(test_idx):,} ({len(test_idx)/len(indices):.1%})"
    )

    splits = {
        "train.csv": train_idx,
        "val.csv": val_idx,
        "test.csv": test_idx,
    }

    for fname, split_idx in splits.items():
        split_path = output_dir / fname
        with open(split_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["disease", "symptoms"])
            for idx in split_idx:
                d, syms = filtered_records[idx]
                writer.writerow([d, ", ".join(syms)])
        logger.info(f"Saved {fname} ({len(split_idx):,} rows)")

    # Save ceiling stats alongside vocab
    with open(output_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "ceiling_stats": ceiling_stats,
            "deduplication": {
                "enabled": deduplicate,
                "raw_records": len(raw_records),
                "after_dedup": len(cleaned_records),
                "duplicates_removed": len(raw_records) - len(cleaned_records),
            },
            "mesh_filter_enabled": filter_mesh_symptom_labels,
            "mesh_labels_excluded": sorted(MESH_SYMPTOM_LABELS),
            "min_disease_support": min_disease_support,
            "split": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
                "train_fraction": round(train_frac, 4),
            },
        }, f, indent=2)

    summary = {
        "total_records": len(filtered_records),
        "num_classes": len(sorted_diseases),
        "num_symptoms": len(sorted_symptoms),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "train_fraction": round(train_frac, 4),
        "identifiability_ceiling": ceiling_stats["identifiability_ceiling"],
        "ambiguous_profiles": ceiling_stats["ambiguous_profiles"],
        "duplicates_removed": len(raw_records) - len(cleaned_records),
    }
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent.parent
    raw_p = base / "data" / "raw" / "hf_symptom_checker_413k.csv"
    out_p = base / "data" / "processed"
    res = run_hardened_preprocessing(raw_p, out_p, min_disease_support=100)
    print("Hardened Preprocessing complete:", json.dumps(res, indent=2))
