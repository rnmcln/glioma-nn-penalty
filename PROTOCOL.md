# Locked analysis protocol

**Title:** The neural-network penalty in public glioma survival prediction: a reproducible TCGA–REMBRANDT benchmark
**Version:** 1.0 (locked 28 June 2026)
**Design:** Retrospective benchmarking and model-appraisal study using public data only.

This protocol fixes the analytical choices before results are interpreted. The committed code in `src/config.py` is the machine-readable counterpart of this document; where the two could diverge, `config.py` governs execution and this file records intent.

## 1. Objective and hypothesis

Primary question: in public glioma datasets, does a small TensorFlow neural network improve overall-survival prediction over transparent clinical/molecular models after fair preprocessing, calibration assessment, and external validation?

The study is framed to make a negative result interpretable. The prespecified decision rule (Section 8) defines what would count as a clinically meaningful improvement; failing that rule is a substantive finding, not an inconclusive one.

## 2. Data sources (public)

| Cohort | Role | Source | Access |
|---|---|---|---|
| TCGA LGG+GBM | Development | GlioVis `TCGA_GBMLGG.Rds`; cross-checked against cBioPortal `lgggbm_tcga_pub` (TCGA, Cell 2016) | Programmatic |
| REMBRANDT | External validation | GlioVis `Rembrandt.Rds` (GEO GSE108476) | Programmatic |
| Gravendeel | External validation (secondary) | GlioVis `Gravendeel.Rds` (GEO GSE16011) | Programmatic |

GlioVis provides an identical clinical schema and gene-symbol expression across cohorts, which is the harmonisation backbone and the analytic source. cBioPortal is retrieved as an independent public reference for TCGA molecular variables; sample-level concordance was not performed in this run (the cohorts are differently curated and differ in size). No imaging data are used. CGGA is not used.

## 3. Population

Adult diffuse glioma with available overall-survival time and vital status. Histologies retained: glioblastoma, astrocytoma, oligodendroglioma, and mixed/oligoastrocytoma. Excluded: non-tumour, pilocytic astrocytoma, and records lacking survival time or vital status. LGG and GBM are analysed together in the primary analysis; high-grade (III–IV) and lower-grade (I–II) subgroups are examined where sample size permits.

## 4. Outcome and censoring

Primary endpoint: 2-year (24-month) overall survival as a binary label. For horizon H, a patient is labelled dead (1) if death occurred with time ≤ H, alive (0) if observed time ≥ H, and excluded if censored before H (event = 0 and time < H). Excluded counts are reported per cohort (Table 1). Sensitivity endpoints: 1-year (high-grade-relevant) and 5-year (lower-grade-relevant). Time-to-event modelling (Cox/penalised Cox) is an optional secondary benchmark.

## 5. Predictor sets (prespecified)

- **A — clinical minimal:** age, sex, grade, histology.
- **B — clinical + molecular:** A plus IDH status, 1p/19q codeletion, MGMT promoter status (added only where available and harmonisable).
- **C — limited expression:** a prespecified 40-gene panel (`config.GENE_PANEL`) present in all cohorts; expression z-scored within each cohort.
- **D — exploratory high-dimensional:** optional; feature selection strictly inside cross-validation.

Variables absent or non-harmonisable in a cohort are not forced. External validation uses only features available in both source and target cohort; the shared set actually used is recorded.

## 6. Preprocessing (leakage controlled)

Per-cohort variable harmonisation; survival in months; vital status as 1 = death. Numeric variables median-imputed then standardised; categorical variables mode-imputed then one-hot encoded. All fitting (imputation, scaling, encoding, any selection) occurs inside the training fold (internal CV) or on the training cohort only (external validation). Variables with > 40% missingness are dropped unless essential (age, grade). Missingness is reported per variable and cohort (Table 2).

## 7. Models

Transparent: base-rate; logistic regression on age + grade; elastic-net penalised logistic regression. Standard ML: random forest; histogram gradient boosting (XGBoost if available). Neural networks: three prespecified TensorFlow MLPs, all with binary cross-entropy, Adam, class weighting, and early stopping, and none tuned outside cross-validation:

- compact: Dense 32 → Dropout 0.2 → Dense 16 → Dropout 0.2 → sigmoid;
- medium: Dense 128 → BatchNorm → Dropout 0.3 → Dense 64 → Dropout 0.3 → Dense 32 → Dropout 0.3 → sigmoid, L2 1e-4;
- regularised: Dense 64 → Dropout 0.4 → Dense 32 → Dropout 0.4 → sigmoid, L2 1e-3.

The best neural network in each setting is compared against the best transparent model. Secondary time-to-event benchmark: Cox and penalised Cox (C-index; internal and external).

## 8. Validation and decision rule

Internal: repeated stratified cross-validation in TCGA (5 repeats × 5 folds for the primary feature-set analyses; 3 repeats × 5 folds for secondary sensitivity settings; out-of-fold probabilities averaged across repeats). External: train on full TCGA, test on Gravendeel (main external cohort) and REMBRANDT (transportability stress test, limited harmonisation) using shared features.

Metrics: AUROC, AUPRC, Brier score, calibration intercept and slope, calibration plot, decision-curve net benefit, and threshold-based operating characteristics where interpretable. Uncertainty: percentile bootstrap CIs; the neural network is compared with the best transparent model by paired bootstrap over the same patients.

Prespecified clinically meaningful improvement: AUROC gain ≥ 0.03 over the best transparent model, with no worse Brier score and an acceptable calibration slope (0.8–1.25). If the network does not meet all three, the conclusion is that it adds no clinically meaningful value.

## 9. Sensitivity analyses

TCGA repeated CV; feature sets A vs B vs C; 1-/2-/5-year endpoints; high- vs lower-grade subgroups; complete-case vs imputed; TCGA→REMBRANDT and TCGA→Gravendeel external transfer.

## 10. Reproducibility and safeguards

Fixed seed (`config.SEED`). No preprocessing, scaling, encoding, or gene selection on the full dataset before splitting. Reporting is not limited to AUROC. Claims of clinical usefulness require calibration and decision-curve support. Outdated/incomplete glioma classifications are reported rather than over-converted. Missingness and non-harmonisable variables are reported, not concealed. Data versions: GlioVis objects should be pinned to a commit SHA for archival reproduction (see README).
