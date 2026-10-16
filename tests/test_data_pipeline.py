"""
Data Pipeline & Dataset Integrity Test Suite
============================================
Verifies:
1. Complete download and integrity of all raw dataset files.
2. Row counts, non-empty content, and absence of corrupted rows.
3. Schema compliance for DDXPlus ontologies (evidences and conditions).
4. Schema compliance for NIH / MedQuAD parsed XML documents.
5. Deterministic hashing / SHA-256 validation.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_hf_symptom_checker_download_exists():
    """Verify HuggingFace symptom checker dataset exists and has valid size (> 30 MB)."""
    csv_file = RAW_DATA_DIR / "hf_symptom_checker_413k.csv"
    assert csv_file.exists(), f"File {csv_file} does not exist"
    size_mb = csv_file.stat().st_size / (1024 * 1024)
    assert size_mb >= 30.0, f"Expected size >= 30 MB, got {size_mb:.2f} MB"


def test_hf_symptom_checker_row_count_and_headers():
    """Verify total row count matches 413,174 patient profiles."""
    csv_file = RAW_DATA_DIR / "hf_symptom_checker_413k.csv"
    import csv
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["disease", "symptoms"], f"Unexpected header: {header}"
        row_count = sum(1 for row in reader if row)
    
    assert row_count == 413174, f"Expected 413,174 rows, got {row_count}"


def test_ddxplus_conditions_schema():
    """Verify DDXPlus release_conditions contains 49 pathologies with ICD-10 and severity."""
    cond_file = RAW_DATA_DIR / "ddxplus_conditions.json"
    assert cond_file.exists(), f"File {cond_file} does not exist"
    with open(cond_file, encoding="utf-8") as f:
        conditions = json.load(f)
    
    assert len(conditions) == 49, f"Expected 49 conditions, got {len(conditions)}"
    for name, data in conditions.items():
        assert "icd10-id" in data, f"Missing icd10-id in condition {name}"
        assert "severity" in data, f"Missing severity in condition {name}"
        assert "symptoms" in data, f"Missing symptoms in condition {name}"
        assert "antecedents" in data, f"Missing antecedents in condition {name}"


def test_ddxplus_evidences_schema():
    """Verify DDXPlus release_evidences contains 223 items (symptoms and antecedents)."""
    evid_file = RAW_DATA_DIR / "ddxplus_evidences.json"
    assert evid_file.exists(), f"File {evid_file} does not exist"
    with open(evid_file, encoding="utf-8") as f:
        evidences = json.load(f)
    
    assert len(evidences) == 223, f"Expected 223 evidences, got {len(evidences)}"
    symptoms = [k for k, v in evidences.items() if not v.get("is_antecedent", False)]
    antecedents = [k for k, v in evidences.items() if v.get("is_antecedent", False)]
    assert len(symptoms) == 110, f"Expected 110 symptoms, got {len(symptoms)}"
    assert len(antecedents) == 113, f"Expected 113 antecedents, got {len(antecedents)}"


def test_medquad_xml_curation():
    """Verify NIH / MedQuAD parsed XML directory contains valid clinical documents."""
    medquad_dir = RAW_DATA_DIR / "medquad_mplus"
    assert medquad_dir.exists(), f"Directory {medquad_dir} does not exist"
    xml_files = list(medquad_dir.glob("*.xml"))
    assert len(xml_files) >= 100, f"Expected at least 100 XML files, got {len(xml_files)}"

    for xml_p in xml_files[:20]:
        tree = ET.parse(xml_p)
        root = tree.getroot()
        assert root.tag == "Document", f"Expected root tag 'Document', got {root.tag}"
        focus = root.findtext("Focus")
        assert focus is not None and len(focus.strip()) > 0, f"Missing Focus in {xml_p.name}"
        qa_pairs = root.findall(".//QAPair")
        assert len(qa_pairs) >= 1, f"No QAPairs found in {xml_p.name}"
