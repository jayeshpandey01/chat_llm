# Clinical Machine Learning Research & Dataset Handling Protocol

> **Document Type:** Research Reference & Engineering Specification  
> **Audience:** Senior Machine Learning Engineers & Clinical Data Scientists  
> **Application:** HealthTrack AI Clinical Symptom Checker & Evidence-Grounded Diagnostic Support System  
> **Repository Target:** `model_building/CLINICAL_RESEARCH_AND_DATASET_HANDLING.md`

---

## 1. Overview & Research Motivation

In diagnostic clinical decision support systems, machine learning models face distinct failure modes compared to typical NLP/tabular applications:
1. **The Hallucination & Fabricated Causality Dilemma:** Generative models and unconstrained multi-class classifiers frequently associate spurious symptoms with rare conditions due to semantic similarity rather than verified clinical evidence.
2. **Combinatorial Feature Sparsity & Extreme Missingness:** Patients present with only 2 to 6 symptoms out of hundreds. Absence of a reported symptom is rarely *Missing Completely At Random (MCAR)*; it is *Missing Not At Random (MNAR)*, representing the absence of an observed physiological sign.
3. **Clinical Ambiguity & Shared Syndromes:** Overlapping respiratory or infectious presentations (e.g., Common Cold vs Pneumonia vs Bronchial Asthma) cannot be resolved by forced single-label output without measuring conditional entropy and differential probability.

To address these challenges with production rigor, this document details the **underlying research literature**, **evidence-grounded paradigms**, and the **exact algorithmic protocols for acquiring, cleaning, analyzing, and serving clinical datasets**.

---

## 2. Key Research Papers & Theoretical Foundations

### 2.1. DDXPlus: Benchmark for Automatic Symptom Detection & Differential Diagnosis
* **Citation:** Tchango, A. F., Jiang, Z., Zipper, A., Hernandez Cazares, R., Abdelaziz, M., & Bouchard, M. (2022). *DDXPlus: A New Dataset For Automatic Medical Diagnosis.* Advances in Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks.
* **Paper Link:** [arXiv:2205.09148](https://arxiv.org/abs/2205.09148) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cc4f653066cd64ff19eb0fc362a4a34b-Abstract-Datasets_and_Benchmarks.html)
* **Core Contribution:**
  - Traditional diagnosis datasets are purely binary ($0/1$). DDXPlus introduces multi-choice, categorical evidences (e.g., pain scale 1-10, fever duration) alongside socio-demographic antecedents (age, sex, smoking history, travel, family history).
  - Explicitly formalizes diagnosis as a **ranked differential diagnosis distribution** rather than an isolated single label.
* **Direct Application to Our System:**
  - We use the DDXPlus medical ontology (`release_evidences.json`, `release_conditions.json`) to standardize symptom definitions and validate our differential diagnosis ranking $P(D_1), P(D_2), \dots$ against gold-standard clinical differential lists.

---

### 2.2. MedQuAD: Authoritative Medical Question Answering & Knowledge Verification
* **Citation:** Ben Abacha, A., & Demner-Fushman, D. (2019). *A question-entailment approach to question answering.* BMC Bioinformatics, 20(1), 511.
* **Paper Link:** [BMC Bioinformatics / NIH](https://doi.org/10.1186/s12859-019-3119-4) | [PubMed Central PMC6880538](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6880538/)
* **Core Contribution:**
  - Curated 47,457 question-answer pairs directly from 12 authoritative NIH institutes (MedlinePlus, CDC, NIDDK, NHLBI, GARD, Cancer.gov).
  - Mapped questions to Unified Medical Language System (UMLS) Concept Unique Identifiers (CUIs) across 37 validated clinical question types (`Diagnosis`, `Treatment`, `Precautions`, `Red Flags`).
* **Direct Application to Our System:**
  - Serves as our **Zero-Hallucination Grounding Anchor**. When our diagnostic engine predicts a condition, the diagnostic report links directly to NIH/MedlinePlus vetted treatment instructions, precautions, and emergency markers.

---

### 2.3. TRACE-PSG: Traceable Diagnosis with Pointer-Grounded Evidence
* **Citation:** Wu, S., Tang, S., Liu, C., Wu, J., Jiang, W., Xiao, X., Liu, H., & Meng, G. (2026). *TRACE-PSG: Traceable Sleep Diagnosis Support with Pointer-Grounded Evidence.* Medical Image Computing and Computer Assisted Intervention (MICCAI 2026), Lecture Notes in Computer Science (LNCS 16886), Springer.
* **Core Contribution:**
  - Replaces free-floating text output with **Pointer-Grounded Evidence**: each diagnostic assertion contains pointer tokens referencing the exact clinical measurements that triggered it.
  - Implements an **Abstention Sentinel**: when input evidence is insufficient or contradictory, the model abstains or requests clarifying evidence instead of fabricating a diagnosis.
* **Direct Application to Our System:**
  - HealthTrack AI enforces pointer grounding: the diagnostic output produces a verifiable trace linking each suggested condition to the exact input symptoms, empirical odds ratios, and pathognomonic indicators.

---

### 2.4. MedAgent: Retrieval-Augmented Clinical Decision Support with Verifiable Grounding
* **Citation:** Wang, F., Guo, Z., & Ye, Z. (2026). *MedAgent: A Retrieval-Augmented Clinical Decision Support Agent with Verifiable Evidence Grounding for Evidence-Based Medicine.* medRxiv Preprint.
* **Paper Link:** [medRxiv:10.64898/2026.06.15.26355735](https://doi.org/10.64898/2026.06.15.26355735)
* **Core Contribution:**
  - Proves that clinical AI systems must anchor every claim to explicit retrieved text spans rather than parametric LLM weights.
  - Utilizes structured clinical schemas (PICO framework) and reward penalties for ungrounded citations.
* **Direct Application to Our System:**
  - All precautions, home care advice, and triage recommendations are statically bounded to NIH-verified text spans, completely preventing ungrounded advice.

---

## 3. Dataset Triangulation Matrix & Sourcing Strategy

Rather than relying on a single flat CSV file, our pipeline unifies three distinct dataset pillars:

| Pillar | Dataset | Primary Source | Records / Size | Clinical Function |
|---|---|---|---|---|
| **Pillar 1** | `akhileshav8/symptom_checker` | Hugging Face Hub | 413,174 rows | Trains sparse, combinatorial symptom representation across 41 conditions and 132 features. |
| **Pillar 2** | `DDXPlus` | Mila / NeurIPS (Figshare) | 1.3M synthetic cases | Provides ontology validation, non-binary evidences, antecedents, and differential diagnosis ranking. |
| **Pillar 3** | `MedQuAD` | NIH / NLM (GitHub) | 47,457 QA pairs | Delivers verified clinical guidance, precautions, UMLS concept mapping, and zero-hallucination citations. |

---

## 4. How to Handle Each Particular Dataset

### 4.1. Handling `akhileshav8/symptom_checker` (Hugging Face)

#### Characteristics:
- **Format:** Parquet / CSV stream containing `disease` (target label) and `symptoms` (comma-separated string).
- **Scale:** 413,174 examples.
- **Nature:** High combinatorial density reflecting power sets of observed symptoms.

#### Engineering & Preprocessing Protocol:
1. **Streaming & Ingestion:**
   - Ingest via buffered chunk reader or parquet table stream to avoid high memory spikes.
2. **Text Normalization & Cleansing:**
   - Strip whitespace, lowercase strings, and replace space variations (e.g., `dischromic _patches` $\to$ `dischromic_patches`, `spotting_ urination` $\to$ `spotting_urination`).
   - Standardize disease labels (e.g., `Diabetes ` with trailing space $\to$ `Diabetes`, `Peptic ulcer diseae` typo $\to$ `Peptic ulcer disease`).
3. **Vectorization:**
   - Parse comma-separated tokens into a sparse multi-hot binary indicator matrix $\mathbf{X} \in \{0, 1\}^{N \times 132}$.
4. **Data Splitting Strategy:**
   - **Do not perform simple random split.** High combinatorial overlap can cause train-to-test leakage.
   - Use **Stratified Shuffle Split** conditioned on the disease label ($80\%$ train, $10\%$ validation, $10\%$ test), with random seed pinning ($42$) for reproducibility.

```python
# Canonical Symptom Normalization Rule
def clean_symptom_token(token: str) -> str:
    cleaned = token.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    # Fix known clinical typo artifacts
    typo_map = {
        "dischromic__patches": "dischromic_patches",
        "spotting__urination": "spotting_urination",
        "foul_smell_of_urine": "foul_smell_of_urine",
    }
    return typo_map.get(cleaned, cleaned)
```

---

### 4.2. Handling `DDXPlus` (Mila / NeurIPS Benchmark)

#### Characteristics:
- **Format:** JSON dictionaries (`release_evidences.json`, `release_conditions.json`) and compressed CSVs (`release_train_patients.zip`, `release_test_patients.zip`).
- **Scale:** Over 1.3 million patient records, 49 pathologies, 223 total evidences (110 symptoms + 113 antecedents).
- **Complexity:** Non-binary data (binary, multi-choice, categorical values, differential probability distributions).

#### Engineering & Preprocessing Protocol:
1. **Acquisition:**
   - Query the Figshare API (`https://api.figshare.com/v2/articles/22687585`) to fetch the English version files.
2. **Evidence Disaggregation:**
   - Separate symptoms ($S$) from patient antecedents ($A$) (age, biological sex, smoking history, family medical history).
   - Unpack categorical variables using one-hot or ordinal representations based on clinical severity.
3. **Differential Distribution Extraction:**
   - Extract the `DIFFERENTIAL_DIAGNOSIS` column (tuples of `[pathology, probability]`) to evaluate top-3 and top-5 model calibration.
4. **Cross-Ontology Alignment:**
   - Construct a mapping dictionary linking DDXPlus evidence codes ($E\_xx$) to our unified clinical feature taxonomy.

---

### 4.3. Handling `MedQuAD` (NIH / NLM Clinical Knowledge Base)

#### Characteristics:
- **Format:** 12 thematic directories containing 47,457 structured XML files.
- **Hierarchy:** `<Document>` $\to$ `<Focus>`, `<FocusAnnotations>`, `<QAPairs>` $\to$ `<QAPair pid="...">`.
- **Focus:** Peer-reviewed NIH guidelines from MedlinePlus, CDC, NIDDK, and Cancer.gov.

#### Engineering & Preprocessing Protocol:
1. **Selective Topic Filtering:**
   - Prioritize `4_MPlus_Health_Topics_QA`, `9_CDC_QA`, `5_NIDDK_QA`, and `8_NHLBI_QA_XML` corresponding to primary care conditions.
2. **Clinical Entity Extraction:**
   - Parse XML trees using `xml.etree.ElementTree`.
   - Index entries by canonical disease name and UMLS CUI.
3. **Categorized Evidence Structuring:**
   - Extract answer texts categorized by clinical intent:
     - `overview`: General definition of the pathology.
     - `precautions`: Immediate non-pharmacological care and contraindications.
     - `red_flags`: Emergency symptoms requiring immediate emergency care.
4. **SQLite Evidence Store:**
   - Compile parsed records into a local, high-speed SQLite database (`clinical_evidence.db`) with full-text search indexing (FTS5) for zero-latency retrieval during inference.

---

## 5. Statistical Exploratory Data Analysis (EDA) Pipeline

To guarantee data quality before modeling, the following statistical analyses must be automatically generated:

```
                  Raw Ingested Clinical Data
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │       1. Distributional Profiling        │
        │ - Class Balance Ratio: Max(N_d) / Min(N_d│
        │ - Symptom Sparsity: % Zero-valued cells  │
        │ - Presentation Length: μ and σ of |s|    │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │       2. Information Theoretic Audit     │
        │ - Mutual Information I(S_i; D)           │
        │ - Normalized Mutual Information (NMI)    │
        │ - Entropy Reduction per Clinical Sign    │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │       3. Association & Overlap Index     │
        │ - Pairwise Jaccard Co-occurrence         │
        │ - Conditional Odds Ratios P(S|D)/P(S|~D) │
        │ - Ambiguity Index: Overlapping Syndromes │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │       4. Data Quality Report & Artifacts │
        │ - data_quality_report.json               │
        │ - evidence_odds_ratios.parquet           │
        └──────────────────────────────────────────┘
```

### 5.1. Mathematical Formulations

1. **Normalized Mutual Information (NMI):**
   $$NMI(S_i; D) = \frac{2 \cdot I(S_i; D)}{H(S_i) + H(D)}$$
   *Purpose:* Identifies symptoms that strongly isolate specific conditions versus generic non-specific symptoms.

2. **Empirical Diagnostic Odds Ratio (DOR):**
   $$\text{DOR}(s_i, d_j) = \frac{P(s_i = 1 \mid D = d_j) \cdot P(s_i = 0 \mid D \neq d_j)}{P(s_i = 1 \mid D \neq d_j) \cdot P(s_i = 0 \mid D = d_j)}$$
   *Purpose:* Supplies mathematically auditable weights for TRACE-PSG pointer grounding.

3. **Condition Ambiguity Index ($A_{j,k}$):**
   $$A_{j,k} = \frac{|\mathcal{S}_j \cap \mathcal{S}_k|}{|\mathcal{S}_j \cup \mathcal{S}_k|}$$
   *Purpose:* Flags disease pairs with high diagnostic overlap (e.g., $A > 0.65$), directing the system to ask targeted differential questions.

---

## 6. Zero-Hallucination Evidence Grounding Protocol

```
                     Patient Symptom Input
                                │
                                ▼
         Extracted Canonical Tokens: [s_1, s_2, ..., s_k]
                                │
                                ▼
              Calibrated Probabilistic Predictor
                                │
                                ▼
       Ranked Differential Diagnosis: [(D_1, P_1), (D_2, P_2), ...]
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │         TRACE-PSG Pointer Grounding         │
         │ - Filter s_i where Observed = True          │
         │ - Compute individual evidence log-odds      │
         │ - Highlight pathognomonic markers           │
         └─────────────────────────────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │       NIH / MedQuAD Provenance Lookup       │
         │ - Query clinical_evidence.db for D_1        │
         │ - Attach official CDC/MedlinePlus citations │
         │ - Retrieve verified precautions & red flags │
         └─────────────────────────────────────────────┘
                                │
                                ▼
                 Auditable Clinical Output Card
           (Zero Generative Hallucination Guaranteed)
```

Every response generated by the system must adhere to the following schema:
```json
{
  "predicted_condition": "string",
  "confidence_score": 0.0,
  "differential_diagnosis": [
    {"condition": "string", "probability": 0.0}
  ],
  "pointer_grounded_evidence": [
    {
      "symptom": "string",
      "observed": true,
      "log_odds_ratio": 0.0,
      "significance": "pathognomonic | strong | supportive"
    }
  ],
  "authoritative_grounding": {
    "source_institute": "National Institutes of Health (MedlinePlus / CDC)",
    "topic_id": "string",
    "clinical_summary": "string",
    "verified_precautions": ["string"],
    "emergency_red_flags": ["string"]
  },
  "abstention_triggered": false
}
```

---

## 7. Verification & Quality Gates

Before any dataset split is accepted for model training, it must pass automated quality gates:
- **Gate 1: Zero Nulls & Complete Normalization:** All symptom strings must be stripped, valid snake_case tokens with no whitespace or special characters.
- **Gate 2: Target Stratification:** Every split (train, validation, test) must maintain identical class distribution proportions within $\pm 1\%$.
- **Gate 3: Evidence Provenance Completeness:** 100% of target conditions in the diagnostic ontology must have corresponding verified records in the NIH/MedQuAD clinical database.
- **Gate 4: Deterministic Auditability:** All raw downloads must match precomputed SHA-256 checksums to ensure reproducibility.

