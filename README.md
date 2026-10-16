# Autonomous Continual Learning LLM Prototype

<p align="center">
  <img src="assets/bunny.jpg" alt="Autonomous Continual Learning LLM Architecture" width="100%" style="border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Ollama-Qwen2.5--0.5B-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama Backend" />
  <img src="https://img.shields.io/badge/Tests-212%20Passed-brightgreen?style=flat-square&logo=pytest&logoColor=white" alt="Pytest Suite" />
  <img src="https://img.shields.io/badge/Code%20Style-Ruff-black?style=flat-square" alt="Ruff" />
</p>

An end-to-end, closed-loop system prototype for **autonomous continual post-training in Large Language Models**. The system integrates live web curation, hybrid retrieval, factual citation grounding, experience replay, parameter-efficient fine-tuning (LoRA), preference alignment (DPO/PPO/GRPO), and non-parametric statistical hypothesis release gating with zero-downtime atomic rollbacks.

> [!NOTE]
> **Implementation Scope**: The default FastAPI application performs live web collection, curation, persistent hybrid indexing, and Ollama-backed grounded inference. Curated documents feed a separate Hugging Face Qwen LoRA post-training worker; candidate checkpoints are promoted into Ollama only when the statistical benchmark release gate approves them. For complete theoretical foundations, mathematical derivations, and architecture specifications, consult the **[11-Chapter Academic Research Monograph](docs/research.md)**.

---

## Key System Capabilities

<p align="center">
  <img src="assets/study_calm.jpg" alt="Continual Learning Workspace" width="100%" style="border-radius: 10px; max-height: 380px; object-fit: cover;" />
</p>

| Pipeline Stage | Module | Engineering Specification |
| :--- | :--- | :--- |
| **1. Web Ingestion** | `ingestion/` | Multi-source scraping via Trafilatura DOM parser, HTML text density ratios, SHA-256 provenance tracking, and rate-limited async fetching. |
| **2. Curation & LSH** | `curation/` | MinHash Locality-Sensitive Hashing (LSH) for near-duplicate elimination ($s = 0.80$, 32 bands $	imes$ 4 rows), 13-gram benchmark contamination filtering, and lexical quality scoring. |
| **3. Hybrid Retrieval** | `retrieval/` | Sparse Okapi BM25 + dense sentence embeddings fused via Reciprocal Rank Fusion (RRF, $k=60$) with bounded in-memory caching. |
| **4. Citation Grounding** | `evaluation/` | Sentence-level ALCE citation recall & precision, RAG Triad faithfulness verification ($> 85\%$ threshold), and fallback citation extraction. |
| **5. Experience Replay** | `curation/` | Stratified replay buffer maintaining a strict **70% Golden Anchor / 30% Streaming Web** mixture ratio to mathematically guard against catastrophic forgetting. |
| **6. Continual SFT & LoRA** | `training/` | Completion-only token masking (zero gradient on user prompts), PyTorch LoRA adapter injection ($r=8, lpha=16$), and adaptive cosine learning-rate schedules. |
| **7. Preference Alignment** | `training/` | Modular alignment engines supporting Direct Preference Optimization (**DPO**, **cDPO**, **IPO**), Actor-Critic **PPO** with GAE, and group-relative policy optimization (**DeepSeek-R1 GRPO**). |
| **8. Statistical Gating** | `evaluation/` | Four-bucket release evaluation using Paired Bootstrap Resampling ($B=1000, 	ext{CI}_{0.95}$), Wilcoxon Signed-Rank tests, and Edwards continuity-corrected McNemar discordance tests. |
| **9. Autonomous Rollback** | `evaluation/` | Atomic state pointer management; automatically promotes validated checkpoints or hot-swaps serving back to the golden anchor upon detected degradation. |

---

## Clinical Diagnostic Machine Learning Subsystem (`model_building/`)

In addition to autonomous LLM post-training, the platform includes a decoupled clinical diagnostic modeling engine trained on large-scale electronic health records and symptom profiles. The architecture addresses data duplication, underfitting/overfitting, symptom-as-disease hallucinations, and 10 critical clinical edge cases.

### Decoupled Model Architecture & Final Accuracies

To avoid representation underfitting caused by merging disparate disease taxonomies into a single noisy classifier, each clinical dataset is trained as an independent specialized model:

| Model | Scope | Algorithm | Test Top-1 Accuracy | Test Top-3 Accuracy | Test Top-5 Accuracy | Macro F1 | Calibration (Brier) | Checkpoint Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Primary Care Benchmark** | 41 Diseases | **Random Forest** | **99.91%** | **100.0%** | **100.0%** | `0.9745` | `0.0362` | `models/model_benchmark_41.joblib` |
| **Model 2: General Multi-System** | 50 Diseases | **CatBoost** | **92.97%** | **99.48%** | **99.90%** | `0.9288` | `0.0984` | `models/model_general_clinical.joblib` |
| **Model 3A: Broad Sparse Baseline** | 580 Diseases | **SGD (L1 + Log Loss)** | **52.31%** | **59.67%** | **61.94%** | `0.4912` | `0.5180` | `models/sklearn_baseline.joblib` |
| **Model 3B: Broad Deep ResMLP** | 580 Diseases | **PyTorch ResMLP (8.9M)** | **52.24%** | **59.39%** | **61.14%** | `0.4988` | `0.0494` (ECE) | `models/pytorch_resmlp.pt` |

#### Identifiability Ceiling
On the broad 580-disease cohort (289,652 deduplicated patient encounters), both the linear baseline (52.31%) and 8.9M-parameter ResMLP (52.24%) encounter the theoretical **Identifiability Ceiling of 56.89%**. 16,687 unique symptom profiles in raw MeSH data have identical symptoms mapped to conflicting diseases. High-certainty primary care and general medicine are therefore handled by decoupled specialized models, while ambiguous presentations utilize split conformal prediction.

---

### Clinical Edge-Case Sentinels & Safeguards

The inference pipeline wraps underlying models with 5 deterministic clinical sentinels:

1. **Emergency Acuity Sentinel (`model_building/models/acuity_sentinel.py`):**
   - Implements Emergency Severity Index (ESI Level 1 & 2) triage overrides.
   - Detects life threats (Myocardial Infarction, Stroke, Pulmonary Embolism, Sepsis, Meningitis, or Acute Cardiac Triad).
   - If an emergency condition crosses calibrated safety thresholds, an urgent triage alert is prepended regardless of model rank.

2. **Active Bayesian Questioning Engine (`model_building/models/active_questioning.py`):**
   - Resolves solitary non-specific symptom presentations ($|S| \le 1$, e.g. "fatigue").
   - Computes binary split entropy across candidate conditions using 101,328 precomputed odds ratios in `data/processed/clinical_evidence.db`.
   - Generates the top-3 discriminative clinical clarifying questions to narrow down the differential diagnosis.

3. **Adaptive Conformal Prediction Sets (`model_building/models/conformal_predictor.py`):**
   - Implements Split Conformal Prediction (Adaptive Prediction Sets - APS).
   - Replaces arbitrary top-k cutoffs with adaptive differential prediction sets $\mathcal{C}_{0.95}(\mathbf{x})$ mathematically guaranteeing $\mathbb{P}(Y_{\text{true}} \in \mathcal{C}_{0.95}) \ge 95\%$.
   - Yields singleton sets for clear presentations, and expanded differential sets for ambiguous viral overlaps.

4. **Demographic Biological Filter (`model_building/data/demographic_rules.py`):**
   - Enforces physiological boundaries on patient age and biological sex.
   - Eliminates biological impossibilities (e.g. Prostate Cancer on females, Ovarian Cysts on males, pediatric conditions on elderly, degenerative conditions on children).

5. **Helmholtz Free Energy OOD Sentinel (`model_building/models/diagnostic_engine.py`):**
   - Calculates Helmholtz free energy $E(\mathbf{x}) = -T \log \sum \exp(z_k / T)$ to detect and reject non-medical, conversational, or out-of-distribution queries with zero hallucinated diagnoses.

---

### Saved Artifacts & Reproducible Pipeline

- **Saved Model Checkpoints (`models/`):**
  - `model_benchmark_41.joblib` (27.8 MB, Random Forest champion, 99.91% test accuracy)
  - `model_general_clinical.joblib` (41.9 MB, CatBoost champion, 92.97% test accuracy)
  - `pytorch_resmlp.pt` (35.7 MB, 8.9M parameter ResMLP with calibrated temperature scaling)
  - `sklearn_baseline.joblib` (11.3 MB, sparse SGD L1 baseline)
  - `model_registry.json` (metadata summary and metrics catalog)
- **Data & Evidence (`data/processed/`):**
  - `clinical_evidence.db` (10.0 MB SQLite FTS5 database with NIH MedQuAD guidelines and evidence odds)
  - `vocabularies.json` (4,865 symptoms, 580 diseases)
  - `train.csv` (231,720 rows, 80.0% train split), `val.csv` (28,966 rows, 10.0%), `test.csv` (28,966 rows, 10.0%)
- **Interactive Research Notebook (`model_building/process.ipynb`):**
  - 14 clean sequential code cells executing the entire pipeline from download and preprocessing to model evaluation and triage in ~5 seconds.

---

## Production Clinical Serving Gateway & SSE Stream (`apps/clinical_routes.py`)

A production-grade, zero-waste healthcare serving architecture implementing the closed clinical loop:
$$\text{User ExtractedSymptom} \longrightarrow \text{Predict} \longrightarrow \text{User Input + Context} \longrightarrow \text{Synthesize} \longrightarrow \text{Create Output (SSE)}$$

### Zero-Waste Cost & Spark Tier Enforcement

Strictly engineered for zero database quota exhaustion on Firebase Free (Spark) tier:
- **1-Read Context Assembly**: All patient profile demographics, chronic conditions, and recent consultation summaries are consolidated in a single document read.
- **1-Write Turn Persistence**: Individual streaming tokens do not hit the database. Only upon SSE stream completion (`[DONE]`) is exactly 1 atomic write performed to update the session.
- **Tenant-Isolated Storage**: Medical lab PDFs and rash photos uploaded to Firebase Cloud Storage (or local disk fallback) under `users/{uid}/documents/` with strict authorization checks preventing cross-tenant access.
- **Continuous RLHF Buffer**: Every consultation trajectory is automatically recorded to `data/clinical_interaction_buffer.jsonl` for offline preference alignment (DPO/GRPO).

### Production Clinical Endpoints

| HTTP Method | Route | Description | Auth Required | DB Operation |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/auth/profile` | Saves patient onboarding profile (age, sex, meds, allergies) | Bearer JWT | 1 Write |
| `GET` | `/api/auth/profile` | Retrieves patient onboarding profile | Bearer JWT | 1 Read |
| `POST` | `/api/predict` | Standalone symptom-to-disease predictor with demographic rules | Optional | 0 DB Calls |
| `POST` | `/api/members` | Registers independent family member (Dad, Mom, Child, Self) | Bearer JWT | 1 Write |
| `GET` | `/api/members` | Lists all registered family members under account | Bearer JWT | 1 Read |
| `GET` | `/api/members/{member_id}` | Retrieves member profile, chronic conditions & timeline | Bearer JWT | 1 Read |
| `PUT` | `/api/members/{member_id}` | Updates member demographics, medications, or allergies | Bearer JWT | 1 Write |
| `POST` | `/api/members/{member_id}/conditions` | Logs historical diagnosis/episode to longitudinal health record | Bearer JWT | 1 Write |
| `GET` | `/api/members/{member_id}/timeline` | Returns chronological medical history for family member | Bearer JWT | 1 Read |
| `POST` | `/api/sessions` | Creates consultation session bound to specific `member_id` | Bearer JWT | 1 Write |
| `GET` | `/api/sessions` | Lists sessions, optionally filtered by `?member_id={id}` | Bearer JWT | 1 Read |
| `GET` | `/api/sessions/{session_id}` | Retrieves session context and recent turns | Bearer JWT | 1 Read |
| `POST` | `/api/upload` | Uploads medical lab PDFs or rash photos (Base64 or Multipart) | Bearer JWT | 1 Upload |
| `GET` | `/api/files/{uid}/{filename}` | Downloads patient uploaded document/image | Bearer JWT | 0 DB Calls |
| `POST` | `/api/chat/stream` | Real-time Server-Sent Events (SSE) streaming consultation with acute-on-chronic reasoning | Bearer JWT | **1 Read + 1 Write** |

---

## Quickstart & Installation

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended)
- **Local LLM Backend**: [Ollama](https://ollama.com/) (running `qwen2.5:0.5b`)

### 1. Install `uv` & Dependencies

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize all runtime and development dependencies
uv sync --extra dev
```

### 2. Run Verification Pipeline

Execute the automated test suite (212 unit tests across all 6 milestone subsystems):

```bash
bash scripts/verify.sh
```

Or run pytest directly:

```bash
uv run pytest
```

### 3. Bootstrap Ollama (Optional)

If setting up Ollama on macOS or Linux, download the base `qwen2.5:0.5b` model:

```bash
bash scripts/bootstrap_m0.sh
```

---

## Running the Live Query-to-Learning Service

### Step 1: Inspect Detected Hardware

The engine automatically selects the optimal device strategy (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback):

```bash
uv run python -c "from core.hardware import get_device_strategy; s = get_device_strategy(); print(f'Device: {s.device_name} | Precision: {s.precision} | Accumulation Steps: {s.gradient_accumulation_steps}')"
```

### Step 2: Start the FastAPI Gateway

Launch the background service:

```bash
mkdir -p logs
nohup uv run uvicorn apps.server:create_app --factory --host 0.0.0.0 --port 8000 > logs/server.log 2>&1 &
```

Confirm health status:

```bash
curl -fsS --retry 10 --retry-connrefused --retry-delay 1 http://localhost:8000/health | python -m json.tool
```

### Step 3: Execute an Augmented Query

Submit a query. The engine researches the topic, curates and indexes accepted source documents, retrieves supporting passages, and generates a grounded response with passage citations:

```bash
curl -sS -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"How does Direct Preference Optimization eliminate the reward model?","top_k":3}'   | python -m json.tool
```

> [!TIP]
> If the small model hallucinates or ignores citation formatting, the **CitationVerifier** intercepts the response and returns a directly grounded citation extract from the top-scoring passage.

### Step 4: Trigger Autonomous Continual Post-Training

Accepted documents persist in `data/knowledge_corpus.jsonl`. When 5 or more new reviewed records accumulate, the background LoRA post-training worker automatically queues. You can also trigger learning manually:

```bash
curl -sS -X POST http://localhost:8000/trigger_learning   -H "Content-Type: application/json"   -d '{"force":true}'   | python -m json.tool
```

### Step 5: Monitor Learning Job & Gate Evaluation

Poll the job status using the returned `job_id`:

```bash
curl -sS http://localhost:8000/learning/jobs/<JOB_ID> | python -m json.tool
```

- **`PROMOTED`**: Checkpoint cleared all 4 release gate criteria and was hot-swapped into Ollama serving.
- **`BLOCKED`**: Candidate exhibited regression or failed statistical confidence; serving remains safely anchored to the previous checkpoint.
- **`FAILED`**: Training crashed or data was unreviewed.

To view the live replay buffer distribution and active jobs:

```bash
curl -sS http://localhost:8000/status | python -m json.tool
```

### Step 6: Atomic Manual Rollback

If required, instantly roll back the active serving model to the initial golden anchor:

```bash
curl -sS -X POST http://localhost:8000/models/rollback | python -m json.tool
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System readiness, active model checkpoint, and device diagnostics. |
| `POST` | `/query` | Grounded RAG query with live research, citation verification, and auto-fallback. |
| `POST` | `/trigger_learning` | Queues bounded LoRA SFT worker on the stratified replay buffer. |
| `GET` | `/learning/jobs/{job_id}` | Polls training metrics, statistical gate report, and promotion status. |
| `GET` | `/status` | Reports anchor vs. web replay counts, index size, and active jobs. |
| `POST` | `/models/rollback` | Atomically reverts the serving pointer to the baseline golden anchor. |

---

## Project Structure

```
autonomous-learning-llm/
├── core/                       # Core abstractions, hardware detection, Pydantic schemas
│   ├── config.py               # Global environment settings & thresholds
│   ├── hardware.py             # CUDA / Apple Silicon MPS / CPU adaptive strategy
│   ├── schemas.py              # Strict Pydantic V2 data contracts
│   └── ollama_client.py        # Robust asynchronous Ollama client
├── ingestion/                  # Multi-source web crawling & DOM extraction
│   ├── scraper.py              # Trafilatura HTML cleaner & density parser
│   └── provenance.py           # SHA-256 content-addressable manifest logger
├── curation/                   # Text deduplication, quality scoring, replay buffer
│   ├── minhash.py              # MinHash LSH near-duplicate index
│   ├── quality_filter.py       # Heuristic language & stopword density filters
│   └── replay_buffer.py        # 70% Anchor / 30% Web stratified experience replay
├── retrieval/                  # Hybrid evidence retrieval subsystem
│   ├── bm25.py                 # Okapi BM25 sparse keyword index
│   ├── embeddings.py           # Dense semantic similarity encoder
│   └── hybrid_search.py        # Reciprocal Rank Fusion (RRF) rank combiner
├── training/                   # Model architectures & training algorithms
│   ├── m1_autograd/            # Reverse-mode automatic differentiation DAG engine
│   ├── m1_transformer/         # MiniGPT causal autoregressive decoder
│   ├── m3_sft/                 # LoRA adapter injection & completion loss trainer
│   └── m5_preference/          # Bradley-Terry RM, DPO, cDPO, IPO, PPO, and GRPO
├── evaluation/                 # Benchmark suites, citation auditing, release gates
│   ├── benchmark_suite.py      # 30-case four-bucket stratified golden benchmark
│   ├── citation_verifier.py    # ALCE proposition entailment & RAG Triad faithfulness
│   ├── llm_judge.py            # Position-debiased pairwise LLM evaluator
│   ├── statistical_gating.py   # Bootstrap CI (B=1000), Wilcoxon, and McNemar tests
│   └── rollback_orchestrator.py# Atomic deployment state manager & hot-swap pointer
├── orchestration/              # Closed-loop query, training, and lifecycle pipelines
│   └── loop_orchestrator.py    # End-to-end continuous learning state machine
├── model_building/             # Clinical diagnostic ML subsystem & reproducible pipelines
│   ├── data/                   # Preprocessing, deduplication, vectorizer, evidence grounding
│   ├── models/                 # Diagnostic engine, ensemble router, sentinels, ResMLP
│   ├── nlp/                    # Clinical symptom extraction & inflection normalization
│   ├── pipeline/               # End-to-end clinical inference pipeline, cache, explanations
│   └── process.ipynb           # End-to-end reproducible research & evaluation notebook
├── models/                     # Saved clinical model artifacts (joblib, pt checkpoints)
├── apps/                       # Production serving interfaces
│   ├── server.py               # FastAPI gateway with continuous learning hot-swapping
│   ├── clinical_routes.py      # Production healthcare routes & SSE chat stream
│   ├── firebase_service.py     # Zero-waste Firestore & Firebase Storage service
│   └── context_synthesizer.py  # FDA CDSS-compliant ultra-compact context assembler
├── scripts/                    # Bootstrap, verification, and end-to-end smoke test utilities
├── docs/                       # Comprehensive academic monograph (docs/research.md)
└── tests/                      # 268 comprehensive unit and integration tests
```

---

## Security & Operational Notes

- **Network Security**: The API endpoints (especially `/trigger_learning` and `/models/rollback`) are intended for internal service orchestration. Do not expose this service through public tunnels (e.g. ngrok, Cloudflare Tunnel) without adding mutual TLS, reverse proxy authentication, or API keys.
- **Stateless Post-Training Worker**: The Hugging Face LoRA post-training worker runs as an isolated asynchronous task to preserve serving gateway responsiveness.
- **Audit Logging**: Every web document, training job, evaluation report, and model rollback is immutably logged under `data/provenance_manifest.jsonl` and `logs/`.

---

## HealthTrack AI: Setup, Run Steps & Architecture Decisions

### 1. Quickstart & Setup Instructions

#### Prerequisites
- **Python 3.10, 3.11, or 3.12**
- **Node.js 18+ & npm**
- **Ollama CLI** (`brew install ollama` or https://ollama.com)

#### Option A: One-Command Fullstack Launch
```bash
# Grants permissions and starts both FastAPI Backend (8000) and Next.js Frontend (3000)
chmod +x scripts/run_fullstack.sh
./scripts/run_fullstack.sh
```

#### Option B: Manual Service Launch
```bash
# 1. Start Ollama Daemon
ollama serve &
ollama pull qwen2.5:0.5b

# 2. Setup Python Virtual Environment & Install Dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start FastAPI Clinical Backend
uvicorn apps.server:app --host 0.0.0.0 --port 8000 --reload

# 4. In a separate terminal, start Next.js Frontend
cd frontend
npm install
npm run dev
# App is live at: http://localhost:3000
```

#### Running the Test Suite
```bash
.venv/bin/python -m pytest tests/ -v
# Verified: 304 unit and integration tests passing in ~37 seconds
```

---

### 2. Assessment Requirements Mapping

| Requirement | Category | Implementation & Evidence |
| :--- | :--- | :--- |
| **Free-text Symptom Input** | **Must-Have** | Next.js consultation interface (`frontend/app/consultation`) with free-text symptom box, multi-turn dialogue, and dynamic prompt submission. |
| **ML Model on Dataset** | **Must-Have** | Decoupled specialized ensemble (`model_building/`): Random Forest (**99.91%** accuracy on 41 diseases), CatBoost (**92.97%** on 50 diseases), and calibrated PyTorch ResMLP (580 diseases). |
| **Results Page with Explanations** | **Must-Have** | Structured diagnostic HUD displaying Top-3 differential diagnoses, calibrated confidence percentages, plain-language clinical rationale, targeted inquiries, and ESI-1/2 emergency alerts. |
| **Data Preprocessing** | **Should-Have** | `model_building/data/preprocess.py`: Lowercase normalization, MeSH symptom vectorization, and exact deduplication pruning 71,814 duplicate encounter rows. |
| **Caching Layer** | **Should-Have** | `orchestration/semantic_cache.py`: Dual-tier cache combining exact SHA-256 normalized hash lookup with dense semantic similarity matching ($\ge 0.92$), reducing repeated query latency to $< 10$ ms. |
| **User Authentication & History** | **Could-Have** | Passwordless 6-digit SMTP Email OTP authentication (`core/mailer.py`, `apps/clinical_routes.py`), SQLite/Firestore session persistence, and multi-family member profile switching (Self, Child, Parent). |
| **Architecture Documentation** | **Must-Have** | Documented in this README, `clinical_pipeline.md`, and the 11-chapter monograph `docs/research.md`. |

---

### 3. Key Design Decisions & Trade-Offs

1. **Decoupled Ensemble vs. Monolithic Model**:
   - *Decision*: Separate primary care (41 conditions), general multi-system (50 conditions), and broad cohort (580 conditions) into specialized models rather than one unified classifier.
   - *Trade-off*: Managing multiple serialized model artifacts (`.joblib` and `.pt`) versus suffering from representation underfitting due to the MeSH Identifiability Ceiling (16,687 clinical profiles sharing identical symptoms across conflicting diseases).
2. **Local Ollama Inference vs. Remote Proprietary APIs**:
   - *Decision*: Host the primary clinical language model locally on Ollama (`qwen2.5:0.5b` and DPO-retrained candidate weights).
   - *Trade-off*: Lower parameter capacity on CPU/MPS versus complete HIPAA compliance, zero egress costs, deterministic latency, and offline operational resilience.
3. **Dedicated OpenRouter Multimodal Intake**:
   - *Decision*: Restrict remote API usage exclusively to OpenRouter VLM (`openai/gpt-4o-mini`) for vision/lab document analysis with strict `max_tokens=400`.
   - *Trade-off*: Modest external API dependency versus attempting to run heavy 11B+ vision parameters on consumer laptops.
4. **Adaptive Conformal Prediction Sets ($\ge 95\%$ Guarantee)**:
   - *Decision*: Replace static top-$k$ outputs with split conformal prediction sets $\mathcal{C}_{0.95}(\mathbf{x})$.
   - *Trade-off*: Sometimes returning a larger differential set for ambiguous viral symptoms versus risking false certainty on high-stakes diagnostic decisions.

---

### 4. AI Usage Disclosure

In compliance with the assessment guidelines, the use of AI tools during development is documented below:

- **AI Tools Used**: Claude (Anthropic), Cursor IDE, and Gemini.
- **Where Applied**:
  1. *Boilerplate Scaffolding*: Generating repetitive Pydantic schema field definitions and Next.js Tailwind UI layout scaffolding.
  2. *Synthetic Rejection Sampling Candidates*: Assisting in formulating contrastive negative demonstration pairs (sycophantic responses, allergy contraindications, and pediatric aspirin warnings) for the preference tuning dataset.
  3. *Unit Test Expansion*: Scaffolding repetitive parameter sweeps and edge-case mocks in `tests/`.
- **What Was Designed, Implemented & Verified by the Candidate**:
  1. *Clinical & ML Architecture*: Conceptualized the decoupled ensemble strategy, MeSH identifiability bounds, active Bayesian questioning engine, and conformal prediction guarantees.
  2. *Algorithmic Formulations*: Implemented the DPO loss formulation, Dual Reward Engine (Safety + Helpfulness), Shannon entropy model collapse guards, and 70/30 anchor mixture mathematics.
  3. *Debugging & Integration*: Diagnosed and resolved critical bugs—including replay buffer duplicate proliferation, C4 markdown filter syntax rejections, VLM emergency marker normalization, and multi-turn symptom accumulation amnesia.
  4. *Empirical Verification*: Validated all 304 unit and integration tests, calibrated model checkpoints, and executed the live statistical release gating pipeline.



