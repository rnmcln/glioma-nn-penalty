# glioma-nn-penalty

A reproducible benchmark of transparent clinical/molecular models against a small
TensorFlow neural network for glioma overall-survival prediction, with internal
cross-validation in TCGA and external validation in REMBRANDT and Gravendeel.

The study is a model-appraisal benchmark, not a demonstration that a neural
network can be trained. It asks whether neural-network complexity is justified by
calibrated, externally valid, clinically meaningful gains over simple models.

## Headline finding

Across internal validation, external validation, all three feature sets, all
endpoints, and grade subgroups, none of the three prespecified TensorFlow MLPs
(compact, medium, regularised) exceeded the best transparent model by the
prespecified margin (AUROC gain ≥ 0.03 with no worse Brier and acceptable
calibration): 0 of 18 prespecified settings. In internal TCGA validation, elastic-net
penalised logistic regression was at least as discriminating and the best
calibrated. Findings were stable under inverse-probability-of-censoring weighting
and echoed by a Cox time-to-event benchmark. See `RESULTS_SUMMARY.md` and
`tables/table5_nn_vs_transparent.csv`.

Run depth: internal CV used 5×5 repeated cross-validation for the primary
feature-set analyses (A, B, C at 2 years) and 3×5 for secondary sensitivity
settings, with 2000 bootstrap replicates throughout (configurable via
`GNN_CV_REPEATS`, `GNN_CV_FOLDS`, `GNN_N_BOOTSTRAP`).

## Repository layout

```
glioma-nn-penalty/
  PROTOCOL.md                 locked analysis protocol
  src/                        library code (data, models, evaluation, plotting)
  scripts/                    ordered pipeline (01..15) + run_internal_full.py
  data/{raw,processed,dictionary}
  results/{metrics,predictions,model_objects,logs}
  figures/  tables/  manuscript/
```

## Environment

Python 3.10. Install dependencies:

```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
```

Key packages: numpy, pandas, scikit-learn, statsmodels, lifelines, matplotlib,
tensorflow==2.17.1, requests, pyarrow, rdata (reads GlioVis .Rds in pure Python;
no R required).

## Exact reproduction

```bash
# 1. Download all public data (GlioVis .Rds, cBioPortal TCGA, GEO matrices)
python scripts/01_download_data.py

# 2. Harmonise cohorts; write data dictionary, Table 1 and Table 2
python scripts/02_build_cohort.py

# 3. Internal repeated CV in TCGA for feature sets A, B, C (all models, shared folds)
python scripts/03_train_baselines.py

# 4. Fit and characterise the final TF MLP on full TCGA (hyperparameter table)
python scripts/04_train_tensorflow.py

# 5. External validation: TCGA -> REMBRANDT and TCGA -> Gravendeel
python scripts/05_external_validation.py

# 6. Sensitivity analyses (1-/5-year endpoints, subgroups, complete-case)
python scripts/06_sensitivity_analyses.py

# 7. Metrics, bootstrap CIs, paired NN-vs-transparent comparison, DCA -> tables
python scripts/07_make_tables.py

# 8. Figures 1-4 + supplementary figures (PNG + PDF)
python scripts/08_make_figures.py

# 9. Secondary time-to-event benchmark (Cox / penalised Cox)
python scripts/09_survival_benchmark.py

# 10. Extra sensitivity: included-vs-censored, IPCW, feature availability
python scripts/10_extra_sensitivity.py

# 11. Modern tabular baseline (FT-Transformer) for the primary settings
python scripts/11_modern_tabular.py

# 12. Repeated-CV variance and nested-CV sensitivity
python scripts/12_variance_nested.py

# 13. Robustness: expression scaling and encoding/padding concordance
python scripts/13_robustness_checks.py

# 14. Data provenance (SHA-256 manifest of raw inputs)
python scripts/14_data_provenance.py

# 15. Reader-facing settings (Table S1) and decision table (Table S10/Table 3)
python scripts/15_clean_tables.py
```

For a one-command full reproduction use `bash run_all.sh`, which calls the
resumable internal-CV driver `scripts/run_internal_full.py` (covering all 13
internal settings, including the IDH-wildtype, IDH-mutant, glioblastoma, and
landmark analyses) and then the external, survival, sensitivity, table, and figure
steps. Scripts 03 and 06 are lightweight single-setting entry points subsumed by
the driver and are not required for a full run. Main figures are
the monochrome design flow, an AUROC forest across internal and external
validation, the within-stratum discrimination figure, and calibration plus
decision curves in Gravendeel; 600-dpi TIFF and vector PDF are written for
submission. The manuscript is formatted for Neuro-Oncology Advances.

For the full internal CV across all settings with checkpointing/resume, use the
driver (invoke repeatedly until it prints ALL_COMPLETE):

```bash
python scripts/run_internal_full.py
```

Resampling depth is set via environment variables, e.g. `GNN_CV_REPEATS=5
GNN_N_BOOTSTRAP=2000`.

Archival reproducibility: pin the GlioVis source to a commit by editing
`GLIOVIS_BASE` in `src/data_download.py` (replace `master` with a SHA).

## Outputs

- `tables/table1_cohort.csv`, `table2_missingness.csv`
- `tables/table3_internal_performance.csv`, `table4_external_performance.csv`
- `tables/table5_nn_vs_transparent.csv` (prespecified decision rule applied)
- `tables/table6_decision_curve.csv`, `tables/supplement_model_details.csv`
- `figures/figure1..4.{png,pdf}`
- `results/predictions/*.csv` (out-of-fold and external probabilities)

## Known limitations (see PROTOCOL.md and manuscript)

REMBRANDT lacks age, sex, and molecular markers in the harmonised source, so
TCGA→REMBRANDT external validation uses grade and histology only (and the
expression panel for set C). Gravendeel lacks MGMT. Expression transfer crosses
platforms (RNA-seq vs microarray) and is mitigated by per-cohort standardisation,
not eliminated. These constraints are reported, not concealed.

## Data use

All data are public. TCGA via cBioPortal/GlioVis; REMBRANDT (GSE108476) and
Gravendeel (GSE16011) via GEO/GlioVis. Cite the original data papers and GlioVis
(Bowman et al., Neuro-Oncology 2017) when using this repository.

## Licence

MIT (see LICENSE).
