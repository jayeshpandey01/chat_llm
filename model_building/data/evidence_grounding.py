"""
Evidence Grounding Database & Provenance Builder
================================================
Constructs high-speed SQLite database (`clinical_evidence.db`) with full-text search (FTS5):
1. Ingests NIH / MedQuAD parsed XML documents (overviews, treatments, red flags).
2. Computes empirical Diagnostic Odds Ratios (DOR) for each symptom-disease pair.
3. Provides query interfaces for TRACE-PSG pointer grounding and authoritative citation lookup.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_evidence_store(
    medquad_dir: Path,
    processed_dir: Path,
    db_path: Path,
) -> sqlite3.Connection:
    """Creates SQLite database containing verified NIH clinical facts and diagnostic odds ratios."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Authoritative Clinical Knowledge Table (NIH MedQuAD)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinical_guidelines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        cui TEXT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        source_institute TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS clinical_guidelines_fts USING fts5(
        topic, question, answer, source_institute, content='clinical_guidelines', content_rowid='id'
    );
    """)

    # 2. Empirical Evidence Attribution Table (TRACE-PSG Log Odds)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evidence_odds (
        disease TEXT NOT NULL,
        symptom TEXT NOT NULL,
        p_s_given_disease REAL NOT NULL,
        odds_ratio REAL NOT NULL,
        is_pathognomonic BOOLEAN NOT NULL,
        PRIMARY KEY (disease, symptom)
    );
    """)

    # Populate MedQuAD guidelines
    xml_files = list(medquad_dir.glob("*.xml"))
    logger.info(f"Ingesting {len(xml_files)} MedQuAD XML files into clinical_evidence.db")
    
    guideline_records = []
    for p in xml_files:
        try:
            tree = ET.parse(p)
            root = tree.getroot()
            focus = root.findtext("Focus") or p.stem
            cui = root.findtext("FocusAnnotations/UMLS/CUIs/CUI") or ""
            source = "NIH / NLM MedlinePlus"

            for qa in root.findall(".//QAPair"):
                q = qa.findtext("Question") or ""
                a = qa.findtext("Answer") or ""
                if q and a:
                    guideline_records.append((focus, cui, q, a, source))
        except Exception as e:
            logger.warning(f"Error parsing XML {p.name}: {e}")

    cursor.executemany(
        "INSERT INTO clinical_guidelines (topic, cui, question, answer, source_institute) VALUES (?, ?, ?, ?, ?)",
        guideline_records,
    )
    cursor.execute(
        "INSERT INTO clinical_guidelines_fts(rowid, topic, question, answer, source_institute) SELECT id, topic, question, answer, source_institute FROM clinical_guidelines"
    )

    # Populate Evidence Odds from training data
    train_csv = processed_dir / "train.csv"
    logger.info(f"Computing empirical odds ratios from {train_csv}")
    
    disease_counts = Counter()
    symptom_counts = Counter()
    disease_symptom_counts = defaultdict(Counter)
    total_records = 0

    with open(train_csv, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 2:
                continue
            total_records += 1
            d, s_str = row[0].strip(), row[1].strip()
            syms = [s.strip() for s in s_str.split(",") if s.strip()]
            disease_counts[d] += 1
            for s in syms:
                symptom_counts[s] += 1
                disease_symptom_counts[d][s] += 1

    odds_records = []
    for d, sym_map in disease_symptom_counts.items():
        n_d = disease_counts[d]
        for s, count in sym_map.items():
            p_s_d = count / n_d
            other_s_count = symptom_counts[s] - count
            p_s_not_d = other_s_count / max(1, total_records - n_d)
            or_val = p_s_d / max(1e-5, p_s_not_d)
            is_path = (or_val >= 15.0 and p_s_d >= 0.50)
            odds_records.append((d, s, round(p_s_d, 4), round(min(or_val, 999.0), 2), is_path))

    cursor.executemany(
        "INSERT OR REPLACE INTO evidence_odds (disease, symptom, p_s_given_disease, odds_ratio, is_pathognomonic) VALUES (?, ?, ?, ?, ?)",
        odds_records,
    )

    conn.commit()
    logger.info(f"Stored {len(guideline_records)} clinical guidelines and {len(odds_records)} evidence odds pairs.")
    return conn


def query_grounded_evidence(
    db_path: Path,
    predicted_disease: str,
    observed_symptoms: list[str],
) -> dict[str, any]:
    """Queries verified clinical guidelines and returns pointer-grounded evidence."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Retrieve TRACE-PSG evidence weights
    pointer_evidence = []
    for s in observed_symptoms:
        cursor.execute(
            "SELECT p_s_given_disease, odds_ratio, is_pathognomonic FROM evidence_odds WHERE disease = ? AND symptom = ?",
            (predicted_disease, s),
        )
        row = cursor.fetchone()
        if row:
            pointer_evidence.append({
                "symptom": s,
                "p_s_given_disease": row[0],
                "diagnostic_odds_ratio": row[1],
                "is_pathognomonic": bool(row[2]),
            })
        else:
            pointer_evidence.append({
                "symptom": s,
                "p_s_given_disease": 0.0,
                "diagnostic_odds_ratio": 1.0,
                "is_pathognomonic": False,
            })

    # Sort evidence by diagnostic significance
    pointer_evidence.sort(key=lambda x: x["diagnostic_odds_ratio"], reverse=True)

    # 2. Retrieve authoritative NIH / MedQuAD guidance
    cursor.execute(
        "SELECT topic, question, answer, source_institute FROM clinical_guidelines WHERE topic LIKE ? LIMIT 3",
        (f"%{predicted_disease}%",),
    )
    guidelines = cursor.fetchall()
    
    if not guidelines:
        # Fallback to FTS query with double-quoted token escaping
        import re
        tokens = re.findall(r"[A-Za-z0-9]+", predicted_disease)
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens[:4])
            try:
                cursor.execute(
                    "SELECT topic, question, answer, source_institute FROM clinical_guidelines_fts WHERE clinical_guidelines_fts MATCH ? LIMIT 2",
                    (fts_query,),
                )
                guidelines = cursor.fetchall()
            except sqlite3.OperationalError:
                guidelines = []

    conn.close()

    formatted_guidelines = [
        {"topic": g[0], "question": g[1], "summary": g[2], "source": g[3]}
        for g in guidelines
    ]

    return {
        "disease": predicted_disease,
        "pointer_evidence": pointer_evidence,
        "authoritative_guidelines": formatted_guidelines,
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent.parent
    medquad = base / "data" / "raw" / "medquad_mplus"
    proc = base / "data" / "processed"
    db = proc / "clinical_evidence.db"
    build_evidence_store(medquad, proc, db)

    # Test evidence grounding lookup
    sample_query = query_grounded_evidence(db, "Pneumonia", ["chest_pain", "cough", "high_fever"])
    print("\n--- SAMPLE GROUNDED EVIDENCE QUERY RESULT ---")
    print(json.dumps(sample_query, indent=2))
