# Industrial Machine Learning Pipeline Architecture Alignment

> **Reference:** Standard Industrial ML Pipeline (Stages 01 to 07 & Supporting Infrastructure)  
> **Application:** HealthTrack AI Clinical Symptom Checker & Evidence Support System  
> **Standard:** Senior AI/ML Engineer (10+ Years Standard)  
> **File:** `model_building/PIPELINE_ARCHITECTURE_ALIGNMENT.md`

---

## 1. Stage-by-Stage Architecture Mapping

Below is the exact mapping between your reference architecture and our HealthTrack AI system:

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                 MAIN PIPELINE FLOW                                     │
 │                                                                                        │
 │  [01 Data Ingestion] ──────► [02 Data Preparation] ─────► [03 Feature Engineering]     │
 │        (✅ DONE)                     (✅ DONE)                     (✅ DONE)            │
 │           │                             │                             │                │
 │           ▼                             ▼                             ▼                │
 │  [04 Model Training] ──────► [05 Model Evaluation] ─────► [06 Deployment]              │
 │      (⏳ NEXT STEP)                  (⏳ NEXT STEP)                  (📝 PLANNED)       │
 │                                                                       │                │
 │                                                                       ▼                │
 │                                                                [07 Monitoring]         │
 │                                                                  (📝 PLANNED)          │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              SUPPORTING INFRASTRUCTURE                                 │
 │                                                                                        │
 │  • Data Sources: Hugging Face (413K) + Mila DDXPlus (1.3M) + NIH MedQuAD (47K)        │
 │  • Data Storage: Local Parquet / CSV + SQLite FTS5 (clinical_evidence.db)              │
 │  • ML Orchestrator: Python CLI / Notebook Engine (process.ipynb)                       │
 │  • Model Store / Registry: Serialized Artifacts (models/*.joblib, metadata.json)       │
 │  • Monitoring & Cache: Cache Hits/Misses, Triage Latency, Feature Drift Auditing       │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. In-Depth State & Implementation Audit

| Pipeline Stage | Architecture Role | HealthTrack AI Implementation | Production Verification Status |
|---|---|---|---|
| **01. Data Ingestion** | Collect data from multiple heterogeneous sources with checksum validation. | `model_building/data/download_datasets.py`: Streaming chunk downloads, local caching, SHA-256 validation across HuggingFace (`413,174` cases), Mila DDXPlus, and NIH MedQuAD XMLs. | ✅ **COMPLETED** (SHA-256 verified in `tests/test_data_pipeline.py`) |
| **02. Data Preparation** | Clean, transform, deduplicate, and validate data integrity. | `model_building/data/preprocess.py`: Resolves 31 delimiter splits (`back_pain` vs `back pain`), unifies 235 casing fragments, strips punctuation/dots, filters narrative QA text, and executes leak-free stratified splitting (80/10/10). | ✅ **COMPLETED** (Passed 15/15 unit & integration tests) |
| **03. Feature Engineering** | Create clinical feature representations from raw input strings and free text. | `model_building/data/vectorizer.py` & `model_building/nlp/symptom_extractor.py`: Multi-hot Compressed Sparse Row (`scipy.sparse.csr_matrix`), NegEx negation detection (*"denies cough"*), and Levenshtein typo correction. | ✅ **COMPLETED** (Verified in `tests/test_nlp_and_vectorizer.py`) |
| **04. Model Training** | Train calibrated probabilistic ML models on prepared feature representations. | Multi-class differential diagnostic classifiers (Calibrated Logistic Regression / LightGBM benchmark + PyTorch Neural Differential Classifier with label smoothing $\alpha = 0.05$). | ⏳ **CURRENT ACTIVE TARGET** |
| **05. Model Evaluation** | Rigorous evaluation against clinical benchmarks. | Top-1 Accuracy, Top-3 Accuracy, Top-5 Differential Coverage, Mean Reciprocal Rank (MRR), and Expected Calibration Error (ECE) on test split (39,759 unseen patients). | ⏳ **COUPLED WITH STAGE 04** |
| **06. Deployment** | Production serving API and web user interface. | FastAPI REST endpoints (`/api/v1/diagnosis/predict`), Pydantic v2 schemas, in-memory LRU query caching ($< 5\text{ ms}$ response), and patient web application. | 📝 **UPCOMING MILESTONE** |
| **07. Monitoring & Feedback** | Monitor performance, latency, distribution drift, and log telemetry. | Query telemetry, symptom frequency monitoring, cache hit ratio tracking, and clinical abstention logging. | 📝 **FINAL INTEGRATION** |

---

## 3. Supporting Infrastructure Mapping

1. **Data Sources & Storage:**
   - Raw source directory: `data/raw/` (immutable downloads with precomputed checksums).
   - Clean processed data: `data/processed/` (`train.csv`, `val.csv`, `test.csv`, `vocabularies.json`).
   - Clinical Knowledge Base: `data/processed/clinical_evidence.db` (SQLite FTS5 full-text search).
2. **ML Orchestration & Execution:**
   - Single-click interactive reference runner: `model_building/process.ipynb`.
   - Programmatic automated test suite: `pytest tests/`.
3. **Model Store & Registry:**
   - Destination directory: `models/` (model weights, hyperparameter configs, calibration curves).

