# Research and Execution Plan for a 95% Accuracy Goal

**Scope:** `model_building/process.ipynb`, its HF symptom CSV tasks, DDXPlus, MedQuAD retrieval, and model evaluation. This is a research plan; it does not claim a model has reached 95% on a valid final holdout.

## 1. What “95% accuracy” should mean here

A single 95% number is not meaningful until the prediction task, label set, split, and metric are fixed. This repository currently contains multiple incompatible tasks:

| Task | Current evidence | 95% assessment |
|---|---|---|
| HF 41-condition slice | Fresh grouped runs: 99.91% test top-1 on both seeds; macro-F1 0.9745 and 0.9742. Both runs contain all 41 classes in train, validation, and test. | Exceeds 95% on two grouped splits of this narrow HF subset. The data appears templated/synthetic and is not independent clinical validation. |
| HF “top 50” slice | The saved 93.05% result came from an older random 70/30 split; it is stale after adding grouped validation/test splits. | 95% is an experiment target, not an established result. Re-run with group splits before model tuning; top-50 labels and the >=2-symptom restriction define a narrower task. |
| HF broad 729-label task | Current saved run: SGD top-1 55.09%; ResMLP top-1 55.32%, top-5 64.83%, ECE 0.1086. It early-stopped at epoch 9 when validation loss stopped improving. | 95% top-1 is not supported by current data/evidence. Increasing epochs, learning rate, batch size, or hidden width is unlikely to resolve label ambiguity or improve the information available in identical symptom profiles. A valid 95% top-1 goal may require a better-labelled dataset or a narrower declared task. |
| DDXPlus pathology prediction | The repo has the two ontology JSONs, but not its patient train/validation/test files. The official release describes 49 pathologies, 223 binary/categorical/multi-choice evidences, antecedents, a ground-truth pathology, and a reference differential. | The best fit for a separate structured-evidence task. A 95% pathology target is plausible to investigate on the official synthetic benchmark, but must be measured on the official test split and described as synthetic-benchmark accuracy. The paper itself warns against direct real-world deployment without evaluation. |
| MedQuAD | Local use is retrieval of NIH/MedlinePlus material, not labelled patient diagnosis examples. The current evidence lookup for “Pneumonia” returned no guidelines. | Do not train a diagnosis classifier from MedQuAD Q&A pairs. Evaluate retrieval relevance/coverage separately. |

The desired headline target should be stated as one of: **top-1 accuracy >= 95% on DDXPlus official test for the ground-truth pathology**, **top-1 >= 95% on the 41-condition HF subset**, or a **top-k differential coverage target**. Do not mix metrics or datasets to reach a number. For an actual patient-facing system, report performance on prospectively collected or independently sourced real cases; the synthetic HF/DDXPlus score alone does not establish that result.

## 2. Repository-specific findings from the rerun

The saved notebook output reports:

- 413,174 raw HF rows with exactly two source columns: `disease` and `symptoms`.
- 412,862 rows after preprocessing checks; 1,827 raw disease strings; 729 labels retained by the support rule; 397,582 retained examples; 5,101 symptoms in the generated vocabulary.
- Duplicate audit: 37,098 repeated `(label, symptom-set)` groups, 71,887 repeated rows beyond the first, and **17,111 symptom sets associated with multiple labels**.
- Broad-task ResMLP: 6.85M parameters, trained with batch 1,024 and learning rate 0.002; validation loss bottomed at epoch 3 (2.5670) and rose afterward; early stopping at epoch 9. Test accuracy was 55.32% and ECE 0.1086.
- SGD baseline test accuracy was 55.09%, nearly the same as the ResMLP. This suggests the current limit is substantially data/target-related rather than simply an undersized neural network.
- The initial 41-condition run reached 99.82% on a grouped test split but failed at artifact serialization because SciPy sparse matrices do not support `len()`. The trainer now uses matrix row counts, writes the artifact atomically, and runs validation-based sklearn candidate selection. Two fresh grouped runs completed and saved the primary artifact: seed 74 (11,371/2,151/2,267; ExtraTrees test top-1 99.91%, macro-F1 0.9745) and seed 152 (11,388/2,137/2,264; ExtraTrees test top-1 99.91%, macro-F1 0.9742). All 41 labels occur in each partition. The seed search chooses a split solely to ensure class coverage, never based on model scores.
- The saved top-50 output (93.05%) and router output predate the grouped split and are stale. They are not valid evidence for the current code.
- In the clinical demo, “coughing” is extracted as a token distinct from `cough`; the evidence store returned no Pneumonia guideline. End-to-end text results therefore include extraction/retrieval limitations that classifier accuracy alone will not measure.

The exact symptom-set conflicts impose a fundamental ambiguity on single-label prediction: one feature vector maps to more than one training label. Before tuning, measure an **empirical identifiability ceiling** by grouping on canonical feature sets and reporting the fraction of rows whose most frequent label agrees with the row’s label. Compute this by split and class. If that ceiling is below 95% on the broad task, no deterministic classifier on those features can deliver 95% top-1 on that evaluation distribution without extra information or a different target.

## 3. Research findings and model selection

### 3.1 Tabular algorithms

Grinsztajn, Oyallon, and Varoquaux benchmarked 45 tabular datasets and found tree ensembles competitive or superior to deep networks on medium-sized tabular data, even after tuning. The result supports benchmarking tree models rather than assuming a wider/deeper MLP will win; it does not guarantee that trees win on this particular symptom dataset. [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)

XGBoost’s original KDD paper describes sparsity-aware tree boosting, making it a relevant challenger for the HF binary sparse symptom matrix. [XGBoost paper](https://doi.org/10.1145/2939672.2939785) CatBoost’s paper and official implementation are relevant for DDXPlus categorical evidence and mixed structured features. [CatBoost NeurIPS paper](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)

Recommended benchmark set per task:

1. **Multinomial sparse logistic regression / SGD** as the transparent low-cost baseline.
2. **XGBoost histogram multiclass** on CSR features for the HF symptom matrix. Tune depth, learning rate, row/column subsampling, regularization, and early stopping against validation only.
3. **ExtraTrees** as an existing reference, with grouped splits and train/validation/test metrics; do not rely on its old random-split score.
4. **CatBoost or XGBoost** for DDXPlus after preserving binary/categorical/multi-choice evidence semantics, `AGE`, `SEX`, and antecedents. The 49-class gold pathology task and differential-set task must be trained/evaluated separately.
5. **ResMLP** only as a measured challenger. Use the current early stopping as a base; tune hidden size and learning rate through a fixed validation search, not by increasing both until the desired test score appears.

### 3.2 Label ambiguity and differential targets

DDXPlus was designed with a ground-truth pathology and a differential distribution because a differential is not equivalent to a one-hot disease label. Its paper reports pathology accuracy and differential recall/precision/F1 separately. The authors found that training directly on differential diagnoses improves differential-oriented metrics, while ground-truth pathology training is more appropriate for ground-truth top-1; those outputs answer different questions. [DDXPlus NeurIPS paper and results](https://papers.nips.cc/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf)

The paper’s interaction-based baselines report high ground-truth pathology top-1 in some settings, alongside materially different differential metrics, and explicitly describe the results as initial baselines on synthetic patients. Its results justify testing a 95% target for that **specific fully specified benchmark protocol**, not promising 95% for the HF 729-class task or real patients.

Cole et al. study learning when only one positive label is supplied for a genuinely multi-label example. This is useful conceptual background for incomplete labels, but should not be applied mechanically to conflicting HF labels: first establish whether alternate labels are clinically valid co-diagnoses, noisy labels, or separate diagnoses. [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Cole_Multi-Label_Learning_From_Single_Positive_Labels_CVPR_2021_paper.html)

### 3.3 Evaluation quality and real-world limits

The BMJ audit of 23 symptom checkers found 34% correct-first diagnoses (95% CI 31–37%) on standardized vignettes; the 2022 systematic review reported primary diagnostic accuracy between 19% and 38% across studies. These are not directly comparable with this repo’s synthetic supervised splits, but they show why a benchmark score should not be marketed as patient-facing diagnostic accuracy. [BMJ audit](https://www.bmj.com/content/351/bmj.h3480) · [npj Digital Medicine systematic review](https://www.nature.com/articles/s41746-022-00667-w)

A 2026 review recommends prospective assessment in the intended home setting, an appropriate reference standard, and adequate sample size; it cautions that vignettes or samples from care facilities alone do not establish home-use validity. [Digital Health 2026 review](https://journals.sagepub.com/doi/10.1177/20552076261487359)

Temperature scaling (Guo et al., ICML 2017) may improve confidence calibration when fitted on validation data, but it preserves class ranking and cannot turn 55% top-1 into 95%. [PMLR paper](https://proceedings.mlr.press/v70/guo17a.html)

## 4. Research repositories and reusable implementations

- [Mila-IQIA DDXPlus repository](https://github.com/mila-iqia/ddxplus): authoritative schema/documentation and references to official train, validation, and test patient files. The repo says examples are synthetic and documents binary, categorical, and multi-choice evidence plus ground-truth pathology and differential diagnosis.
- [DMLC XGBoost repository](https://github.com/dmlc/xgboost): maintained implementation and tutorials for sparse-aware tree boosting; use the project’s official documentation for parameters and early stopping rather than copying an unverified symptom-checker fork.
- [CatBoost repository](https://github.com/catboost/catboost): official implementation, classification tutorials, categorical feature handling, validation, early stopping, and parameter tuning examples.
- [scikit-learn grouped cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html): group-aware splitting concepts; grouped splits are required for repeated symptom profiles, and class coverage must be checked explicitly.
- [Official DDXPlus project page](https://mila-iqia.github.io/ddxplus/): release files and data schema. The current repo has only `ddxplus_conditions.json` and `ddxplus_evidences.json`, so first acquire the patient split files before training this independent model.

Avoid selecting an implementation because its README reports “99%+” without independently reproducing its dataset version, official split, feature availability, and scoring protocol. Such a result may use fully observed synthetic cases, a different split, or leakage-prone preprocessing.

## 5. Proposed execution plan

### Phase 0 — Freeze task definitions and data versions

- Pin source URL/release and SHA-256 for HF, each DDXPlus file, and MedQuAD.
- Preserve source label and canonical label separately; publish the exact alias map and reasons for every collapse/exclusion.
- Keep four different artifacts/contracts: HF 41-class subset, HF top-50 subset, HF broad 729-label task, and DDXPlus. MedQuAD stays a retrieval/evidence corpus.
- Report row counts and all filtering steps. Do not describe the two HF subsets as independent datasets.

**Gate:** each task has a written target, label map, input schema, source checksum, and evaluation protocol before training.

### Phase 1 — Diagnose the achievable ceiling before model search

- Run the existing raw duplicate/ambiguity audit after the same canonicalization used at inference.
- Report exact row duplicates, repeated feature-label groups, feature sets with conflicting labels, conflicting-label support, symptom count, label support, and class priors.
- Calculate the majority-label ceiling for exact feature-set groups. Also report results separately for unambiguous and conflicting feature sets.
- Compare a random stratified split (diagnostic only) with a grouped split by exact canonical symptom profile. Never select a model using the random-split result.
- For the broad 729 labels, decide among: retain hard-label classification with an explicit ambiguity/abstention outcome; obtain better labels; or define a supported top-N task based on training-only selection. Do not turn conflicting labels into a multi-label target until clinical semantics support co-diagnosis.

**Gate:** if the grouped identifiability ceiling is below 95%, mark 95% top-1 as infeasible for that target/features and move to a scoped target or new data.

### Phase 2 — Train fair candidate models

- Recreate one immutable grouped train/validation/test split per HF task. For the main task, keep at least 70% training; current 80/10/10 grouped split is suitable if all class coverage checks pass. For the 41/top-50 slices, use the grouped 7-fold assignment with one fold each for validation and test (about 71.4/14.3/14.3), not old 70/30 results.
- Fit vocabularies, top-N class selection, feature filtering, balancing weights, and thresholds using training only. The input contract should reject unknown fields rather than silently discard them; the HF CSV has only two columns, while DDXPlus features must be separately encoded.
- Evaluate logistic/SGD, ExtraTrees, and XGBoost on identical splits. Tune only with validation. Keep at least three fixed seeds for training/model instability estimates where computationally practical.
- For ResMLP, log train and validation loss/metrics, preserve best checkpoint, use early stopping, and compare against tree baselines. Increase epochs only if validation curves support it. Higher hidden size or learning rate is an experiment, not a guaranteed improvement.
- Use official DDXPlus patient train/validation/test files. Train (a) a gold pathology classifier and (b) a differential predictor as distinct models/heads and report the correct metrics for each. Preserve `AGE`, `SEX`, and all relevant evidence types; encode “not asked/default” distinctly from negative evidence.

**Gate:** no row/profile overlap across partitions; every metric tied to the immutable split manifest; each evaluation split has documented class coverage; test is not consulted for model/hyperparameter selection.

### Phase 3 — Define and test the 95% acceptance target

For one chosen task, pre-register the exact target, for example:

- `top1_accuracy >= 0.95` on the official/grouped test set for a named task; and
- lower 95% bootstrap confidence bound reported, not just point estimate; and
- macro-F1 / balanced accuracy and per-class recall reported beside micro accuracy; and
- no class absent from the test set without an explicit coverage note; and
- calibration (NLL, Brier, ECE with binning method) reported separately from ranking; and
- duplicate/feature-group isolation asserted.

For differential outputs, use recall/precision/F1 of the reference differential, ground-truth pathology inclusion in the list, and top-1 ground-truth accuracy as separate outcomes. A 95% differential inclusion rate is not the same as 95% top-1 accuracy.

For a broader symptom-checker claim, create a separate external test protocol with independently sourced cases and clinician-adjudicated reference diagnoses. Evaluate the full text pathway, including extraction, negation, unknown symptom handling, ranking, and abstention—not just a vector-to-label model.

### Phase 4 — Error analysis and iteration

- Inspect errors by label, symptom count, conflicting-feature status, common/rare support, and source subgroup.
- For text input, build an annotated extraction set covering synonyms, inflections (`cough`/`coughing`), misspellings, negation, temporality, severity, and unknown symptoms. Fix extraction and measure its precision/recall before attributing end-to-end errors to the classifier.
- Add tests for split leakage, metrics, artifact reload, calibration, and abstention after the target pipeline is implemented.
- If top-1 remains below threshold, examine label noise and data sufficiency first; do not repeatedly enlarge the MLP or oversample exact duplicates.

### Phase 5 — Reporting

- Keep a machine-readable run manifest: dataset/checksum, preprocessing version, label map, split/group hashes, seeds, candidate parameters, validation selection, final test metrics, confidence intervals, package/device versions, and model checksum.
- Replace old hard-coded performance claims (including prior 99.96%, 93.05%, and 99.33% claims) with results linked to specific manifests.
- Report benchmark accuracy as benchmark accuracy. State that DDXPlus is synthetic and that external clinical validation is still required.

## 6. Recommended priority order

1. **Completed:** Re-run the 41-condition artifact save after the sparse-count fix, compare sklearn candidates on validation, verify group isolation/class coverage, and repeat on a second deterministic grouped split. Both runs clear 95% top-1 on this narrow benchmark task only.
2. Re-run the top-50 model under its current grouped validation/test split; discard the stale 93.05% random-split result.
3. Compute the broad 729-task identifiability ceiling and a grouped split error report before trying new neural hyperparameters.
4. Acquire official DDXPlus patient splits and build a dedicated mixed-evidence classifier plus a differential-specific model. Use this as the most defensible route to a synthetic benchmark target near 95%, while clearly restricting the claim to that benchmark.
5. Only after task-specific benchmarks are stable, evaluate the complete free-text product with an independent clinical evaluation set.

## 7. References

### Papers and articles

1. Tchango et al. (2022), [DDXPlus: A New Dataset for Automatic Medical Diagnosis, NeurIPS](https://papers.nips.cc/paper/2022/file/cae73a974390c0edd95ae7aeae09139c-Paper-Datasets_and_Benchmarks.pdf).
2. Grinsztajn, Oyallon & Varoquaux (2022), [Why do tree-based models still outperform deep learning on typical tabular data?, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf).
3. Chen & Guestrin (2016), [XGBoost: A Scalable Tree Boosting System, KDD](https://doi.org/10.1145/2939672.2939785).
4. Prokhorenkova et al. (2018), [CatBoost: unbiased boosting with categorical features, NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
5. Cole et al. (2021), [Multi-Label Learning From Single Positive Labels, CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Cole_Multi-Label_Learning_From_Single_Positive_Labels_CVPR_2021_paper.html).
6. Guo et al. (2017), [On Calibration of Modern Neural Networks, ICML/PMLR](https://proceedings.mlr.press/v70/guo17a.html).
7. Semigran et al. (2015), [Evaluation of symptom checkers for self diagnosis and triage, BMJ](https://www.bmj.com/content/351/bmj.h3480).
8. Wallace et al. (2022), [The diagnostic and triage accuracy of digital and online symptom checker tools: a systematic review, npj Digital Medicine](https://www.nature.com/articles/s41746-022-00667-w).
9. Donner-Banzhoff et al. (2026), [Symptom checkers: Gaps in their evidence base and suggestions for accuracy evaluation, Digital Health](https://journals.sagepub.com/doi/10.1177/20552076261487359).

### Official repositories and documentation

- [Mila-IQIA DDXPlus data/schema repository](https://github.com/mila-iqia/ddxplus).
- [XGBoost implementation and official examples](https://github.com/dmlc/xgboost).
- [CatBoost implementation and tutorials](https://github.com/catboost/catboost).
- [scikit-learn grouped cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).
