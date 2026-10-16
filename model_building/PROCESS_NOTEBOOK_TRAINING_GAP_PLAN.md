# `process.ipynb` Training Review and Gap Resolution Plan

**Purpose:** research-backed review of the notebook and its directly connected preprocessing, vectorization, training, and routing code. This document is a plan; it does not claim models have been retrained or targets achieved.

## Executive summary

The notebook is an orchestration notebook, not a self-contained training pipeline. It currently executes preprocessing with an 80/10/10 split for the 729-class task, calls two separate 70/30 models derived from the same raw CSV, and then routes between those models by comparing confidence values. The split percentages meet the request to train on at least 70% only for the two filtered models. For the main pipeline, 80% is already used for training; moving to 70% would reduce its training data. Interpret “use 70%” as a **minimum training fraction**, preserve 80/10/10 for the main task, and use a separate 70/15/15 or 70/10/20 split only where a dedicated comparison protocol is needed.

The most consequential gaps are data provenance/label validity, duplicate-pattern leakage, inconsistent preprocessing across models, unsupported claims of high accuracy, premature test-set evaluation, too few neural epochs with no early stopping, dense per-row materialization, incomplete dataset-specific handling, and an unvalidated confidence router. “10/10” cannot be promised by a model choice or epoch count; it must mean a reproducible comparison with leakage checks, uncertainty/calibration reporting, and honest limits on synthetic or weak-label data.

## Notebook flow and connected files reviewed

| Notebook cells | Code path | Current behavior / issue |
|---|---|---|
| 0–2 | `download_datasets.py`, `preprocess.py` | Downloads HF data and DDXPlus ontologies; preprocesses only the HF CSV into 80/10/10. `min_disease_support=100` filters using whole-dataset counts before splitting. |
| 3–6 | `run_clinical_eda.py`, `evidence_grounding.py`, `symptom_extractor.py`, `vectorizer.py` | EDA/evidence and a free-text demo. Dataset vectorization is multi-hot; a 729-class vocabulary is built globally before split. |
| 7 | `sklearn_baseline.py` | Sparse SGD logistic baseline; default 25 passes; scores validation and test in one call. |
| 8 | `pytorch_resmlp.py` | ResMLP with three epochs in notebook, batch 512, dense rows, `drop_last=True`; fits temperature on validation and computes test metrics. No epoch-level validation, early stopping, or best-checkpoint restore. |
| 9 | `diagnostic_engine.py` | Demo uses the 729-class model. |
| 10–11 | `train_benchmark_41.py`, `train_general_clinical.py` | Two filtered models from the same HF CSV, each with 70/30 train/test and no validation partition. Dense 41-model matrix; top-50 uses sparse data. |
| 12 | `ensemble_router.py` | Runs both models and compares raw top confidences across different label spaces; selection logic does not establish which model owns the task or validate routing. |

Additional directly relevant files reviewed: `CLINICAL_MODEL_TRAINING_PLAN.md`, `CLINICAL_RESEARCH_AND_DATASET_HANDLING.md`, `CLINICAL_PREPROCESSING_ARCHITECTURE.md`, `CLINICAL_DATA_AUDIT_AND_EDGE_CASES.md`, `ACCURACY_IMPROVEMENT_AND_EDGE_CASE_RESEARCH.md`, `PIPELINE_ARCHITECTURE_ALIGNMENT.md`, `model_condition.md`, and `data/processed/clinical_eda_report.json`.

## Findings and specific gaps

### P0 — Fix before treating scores as meaningful

1. **The HF file is mixed and its labels are not established as 729 verified diseases.** The pipeline’s own audit describes symptom-as-condition rows, narrative QA contamination, duplicated/casing-fragmented labels, and ambiguous symptom sets. The current `DISEASE_CANONICAL_MAP` is only a small subset of the audit’s stated 235 casing/typo variants. A minimum support threshold is not clinical validation. Build a provenance-backed label taxonomy, retain raw label/source, review mappings, and report excluded/uncertain labels. Do not call the resulting labels diagnoses without qualification.
2. **Exact repeated symptom vectors can cross random partitions.** The audit reports 17,088 symptom patterns associated with multiple labels; source rows may also be repeated/reordered synthetic templates. Deduplicate exact `(canonical feature set, label)` records and keep all records sharing a canonical feature-set group in one partition. Report both ordinary stratified results and grouped holdout results; group split can affect class balance, so explicitly report classes absent from a split. Where actual patient IDs do not exist, do not claim patient-level independence.
3. **Leakage in vocabulary/support filtering and evaluation workflow.** Vocabulary construction and the support cutoff happen before splitting. For a fixed, predeclared symptom ontology, vocabulary can be schema-only; learned filters, label selection, feature selection, thresholds and tuning must be fit on training only. Notebook currently evaluates test during every training call. Keep a validation set for model choice and calibration; access the untouched test set once after decisions freeze.
4. **Conflicting task definitions and split sizes.** Processed main task is 80/10/10 (good: 80% train), while filtered tasks use 70/30 and have no validation data. Do not tune on their test partition. Use train/validation/test, e.g. 70/15/15 for the smaller filtered datasets or repeated nested CV on development data plus one final holdout. Clearly state whether “70%” means at least 70% train (recommended) or exactly 70%.
5. **The numbers in existing markdown are not substantiated by code/artifacts.** Claims such as 55.2%, 99.96%, 93.05%, 99.33%, and dataset sizes are not reproducible from checked-in metric outputs. The committed EDA JSON says 729 conditions, 4,973 symptoms and 318,064 *training* records; other docs state 745 classes or 5,101 features. Recompute counts from a pinned raw-file checksum and generated splits, save a run manifest, and remove/label historical claims unless accompanied by the exact data version, split, code version, and metrics.

### P1 — Modeling and training quality

6. **Dataset-specific models are not yet dataset-specific in a clean experimental sense.** The 41-condition and top-50 tasks are slices of one HF artifact; MedQuAD is a QA/evidence corpus, not a diagnosis-label training set; DDXPlus has structured binary/categorical/multi-choice evidence, antecedents, ground-truth pathology and differential labels and must have its own schema/model/evaluation. Keep distinct task contracts and artifacts. Do not concatenate raw targets or assume MedQuAD is a supervised classifier dataset.
7. **Weak baselines and unjustified architecture selection.** Retain sparse multinomial logistic/SGD as a baseline. For sparse binary symptom profiles, benchmark regularized logistic regression and a sparse-friendly boosted tree (e.g. XGBoost/LightGBM if available); compare ExtraTrees only with realistic memory use. The 729-class, 318k-row task may justify ResMLP, but only if it beats these baselines on held-out grouped data. For DDXPlus, start with CatBoost/gradient-boosted trees for mixed categorical features plus a separate MLP only as a measured challenger; preserve categorical semantics and differential target if using it. No one architecture is “proper” by declaration.
8. **Three epochs is a placeholder, not a defensible training budget.** The notebook forces `epochs=3`, the function default is 5, and there is no validation loss tracking or early stopping. Add per-epoch train/validation loss and task metrics; select best validation checkpoint; early-stop with patience and LR schedule. Begin with 20–50 maximum epochs as a search ceiling, not a requirement; actual stopping is validation-driven. Increase epochs only if validation loss/metric still improves. Set `drop_last=False` so every training example is used; log examples seen and effective fraction. Keep seed control for Python/NumPy/PyTorch and device-aware deterministic settings.
9. **Neural input path wastes memory/compute.** CSR samples are converted one row at a time to dense arrays in `__getitem__`, then stacked by DataLoader. Use a CSR-aware batch collator or sparse linear input path, or pre-batch conversion; benchmark throughput/RAM. Notebook batch size 512 is not a correctness issue but should be hardware-tuned. Avoid asserting 99.92% sparsity for every split/data version without recomputation.
10. **Class imbalance and ambiguity are under-addressed.** EDA reports class imbalance ratio 25.94 and many label-conflicting patterns. Compare unweighted loss against train-only class weights and/or balanced sampling; report per-class recall, macro-F1, balanced accuracy, support-stratified results and confusion groups. Label smoothing is not a substitute for multi-label/differential targets. Where reliable differential labels exist (DDXPlus), model/evaluate the differential ranking directly. For single gold labels with irreducible ambiguity, retain hard-label task but report ambiguity-stratified performance and abstention/coverage-risk curves.
11. **Calibration and ranking metrics incomplete/inconsistent.** PyTorch computes ECE only, while sklearn reports no calibration; sklearn MRR computes a full argsort for each example and classes mapping is fragile. Use common batched metric implementation: top-1/top-3/top-5, MRR, macro-F1, balanced accuracy, NLL, Brier score, reliability diagram/ECE (binning method documented), and confidence intervals. Fit calibration on validation only (temperature scaling for logits, appropriate calibrator for tree probabilities); test calibration once. Temperature scaling preserves class ranking but cannot improve accuracy; ECE alone is bin-dependent.
12. **Router confidence comparison is invalid without evidence.** A 41-class posterior and 50-class posterior have different class priors, label support and calibration. Route by explicit supported domain/label coverage or evaluate a learned router on held-out routing data; normalize label ontology and calibrate per model before any shared probability comparison. If no validated routing policy exists, choose a single explicit task model or return separate results with source/coverage metadata.

### P2 — Reproducibility and notebook operations

13. Add a data/model manifest for every run: source URL/release, SHA-256, row counts before/after filters, label and feature counts, split seed and group key, preprocessing version, git commit, package versions, device, hyperparameters, epoch history, selected checkpoint, metrics, and artifact paths.
14. Separate “download”, “preprocess”, “train candidate”, “select/calibrate”, and “final test” cells. Make expensive cells safely skippable; prevent training cell reruns from overwriting a final artifact without a run ID. Notebook currently imports modules under `model_building.models`; verify all notebook cells work from both repo root and notebook directory after implementation.
15. Use explicit exceptions/data-contract checks: file/checksum present, expected columns, no unknown labels/features after transform, non-empty splits, class counts, split disjointness, finite logits/probabilities, saved artifact reloads with matching metadata.

## Recommended algorithm and data plan

| Data/task | Recommended first model | Challenger | Required treatment |
|---|---|---|---|
| HF 41-condition benchmark slice | Sparse multinomial logistic regression; report as a benchmark, not clinical validation | ExtraTrees / boosted trees | Canonicalize labels/features identically; group identical feature vectors; 70% minimum train; validation and final test; expect synthetic/template leakage risk. |
| HF broad 729-class task | SGD/logistic sparse baseline | ResMLP and sparse-friendly gradient boosting | Audit label ontology and pattern ambiguity first; compare on identical grouped splits; macro and top-k metrics; confidence intervals and abstention. |
| HF “top 50” slice | Regularized logistic regression | ExtraTrees/boosted trees | State that top-50 is a selected-label task; choose classes on training data only; no 70/30-without-validation. Filter by support and symptom count only under a predeclared, clinically justified protocol, and report removed proportion. |
| DDXPlus | CatBoost/boosted trees with proper categorical and multi-choice encoding; pathology classifier and differential-ranking task as separate targets | MLP/TabTransformer | Use official split if available; map evidence ontology; preserve age/sex/antecedents and unknown/not-asked states; evaluate pathology top-k and reference differential ranking separately; state data are synthetic. |
| MedQuAD | No diagnosis classifier | Retrieval/FTS or embedding retrieval | Keep as a cited evidence retrieval corpus; measure retrieval relevance/coverage and provenance. Do not treat question-answer pairs as disease labels. |

“Each dataset should have its own kind of model” should be implemented as independent preprocessing contracts, label spaces, trained artifacts and evaluation reports. Different tasks within one source (e.g. HF 41 vs broad label set) may be separate task-specific models, but should be identified as subsets, not independent datasets.

## Execution plan and acceptance criteria

### Phase 1 — Establish trustworthy data contract

- Pin and checksum each source release; regenerate all counts instead of copying doc numbers.
- Resolve disease alias map against audited raw labels; maintain canonical label, original label, mapping rationale and confidence.
- Produce data audit tables: source counts, excluded rows/reasons, duplicate exact rows, duplicate feature groups, support histogram, symptom-count histogram, label overlap/ambiguity, unknown tokens.
- Set deterministic splits with groups for identical canonical feature vectors; preserve at least 70% training examples where feasible and reserve validation/test. Use 80/10/10 for the broad task unless grouped split constraints require adjustment.
- Acceptance: train/val/test group IDs are disjoint; split counts sum exactly; all labels in evaluation are supported by training or separately identified as unseen; source checksum and manifest recorded.

### Phase 2 — Run fair per-task baselines

- Same splits and features for logistic, ExtraTrees/boosting and ResMLP within each compatible task.
- Use train-only class balancing; hyperparameter selection exclusively on validation (or nested development CV). Do not use final test in notebook model-selection loop.
- ResMLP: maximum epoch ceiling 20–50, early stopping on validation macro-F1/NLL (choose primary metric before run), best checkpoint restore, `drop_last=False`, LR scheduler driven by validation plateau/cosine with a justified schedule, save epoch history.
- Acceptance: no model is promoted based on training score. Comparison report includes mean/CI across seeds or resamples, resource use, all required metrics, and grouped-vs-random split delta.

### Phase 3 — Calibration, abstention and router

- Fit calibration on validation data only. Choose abstention thresholds to meet a predeclared coverage-risk target on validation, then freeze them.
- Test selective risk, calibration, classwise errors and performance for 1-symptom, high-ambiguity, rare-class and out-of-vocabulary subsets.
- Make model routing explicit and validate on held-out route examples. If routing cannot beat/select a safe task model, remove confidence-based arbitration.
- Acceptance: final test is run once after configuration freeze; provide confidence intervals and calibration plots; clearly disclose synthetic/weak-label limits.

### Phase 4 — Notebook and docs cleanup

- Refactor `process.ipynb` into concise ordered stages, set main task train fraction to the agreed minimum/ratio (recommended keep 80%), point cells at the validated train functions, and include per-dataset artifact/metric paths.
- Align training-plan, audit and architecture docs with measured classes/features/row counts and actual implemented behavior. Remove unsupported accuracy promises.
- Acceptance: fresh-kernel run from documented working directory; deterministic artifact metadata; notebook results match saved JSON reports. No fixed “10/10” or accuracy target is declared without evidence.

## Research basis (primary sources)

1. Tchango et al., **DDXPlus: A New Dataset For Automatic Medical Diagnosis**, NeurIPS 2022. The paper describes ~1.3M synthetic patients with pathology, symptoms, antecedents and differential diagnoses; this supports a separate structured-evidence model and differential-specific evaluation, while requiring synthetic-data caveats. [NeurIPS paper page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cae73a974390c0edd95ae7aeae09139c-Abstract-Datasets_and_Benchmarks.html) · [paper PDF](https://papers.neurips.cc/paper_files/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)
2. Grinsztajn, Oyallon & Varoquaux, **Why do tree-based models still outperform deep learning on typical tabular data?**, NeurIPS 2022. Their benchmark finds tree models strong on medium-sized tabular data and motivates benchmarking, not assuming an MLP wins. [Paper](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)
3. Chen & Guestrin, **XGBoost: A Scalable Tree Boosting System**, KDD 2016. Includes sparsity-aware learning, relevant to a sparse symptom matrix as a candidate baseline. [KDD paper](https://doi.org/10.1145/2939672.2939785)
4. Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017. Establishes temperature scaling as a simple post-hoc calibration method; calibrate on held-out validation and remember it does not change ranking. [PMLR paper](https://proceedings.mlr.press/v70/guo17a.html)
5. scikit-learn documentation on [cross-validation and grouped data](https://scikit-learn.org/stable/modules/cross_validation.html) and [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html). Grouping is needed when rows share a dependence source; group splitter does not itself preserve class stratification, so inspect class coverage.
6. Hollmann et al., **Accurate predictions on small data with a tabular foundation model**, Nature 2025. TabPFN is aimed at small/medium tables (paper scope up to about 10k rows/500 features); it is not a primary candidate for the 318k-row, ~5k-feature task, though it could be an optional small-slice experiment if compute and license permit. [Nature paper](https://www.nature.com/articles/s41586-024-08328-6)

## Decisions to resolve before implementation

- Interpret “70% should be use” as **at least 70% training**; recommended main split remains 80/10/10. If exact 70% is required, use 70/15/15 to preserve both validation and final test.
- Confirm whether the broad 729-class HF model is still in scope after weak-label audit, or whether primary delivery should focus on the cleaner 41-class benchmark plus separately processed DDXPlus.
