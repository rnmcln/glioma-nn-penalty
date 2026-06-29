#!/usr/bin/env bash
# End-to-end reproduction of the analysis (tables in tables/, figures in figures/,
# intermediate results in results/). The manuscript and supplement .docx are built
# separately and are not part of this repository.
#
# Full-scale settings are the defaults below; override via the environment, e.g.
#   GNN_CV_REPEATS=5 GNN_N_BOOTSTRAP=2000 bash run_all.sh
set -euo pipefail
export GNN_CV_REPEATS="${GNN_CV_REPEATS:-5}"   # 5x5 primary / 3x5 secondary (set in scripts)
export GNN_N_BOOTSTRAP="${GNN_N_BOOTSTRAP:-2000}"
export TF_CPP_MIN_LOG_LEVEL=2
cd "$(dirname "$0")"

python3 scripts/01_download_data.py
python3 scripts/02_build_cohort.py

# Internal cross-validation for all 13 internal settings (resumable driver;
# supersedes the single-setting scripts 03 and 06). Loop until complete.
until python3 scripts/run_internal_full.py | tee /dev/stderr | grep -q ALL_COMPLETE; do
  echo "…continuing internal CV"
done

python3 scripts/04_train_tensorflow.py        # final TF fits + hyperparameter table
python3 scripts/05_external_validation.py      # TCGA -> Gravendeel / REMBRANDT
python3 scripts/09_survival_benchmark.py       # Cox / penalised Cox C-index
python3 scripts/10_extra_sensitivity.py        # included-vs-censored, IPCW, feature availability
python3 scripts/11_modern_tabular.py           # FT-Transformer baseline
python3 scripts/12_variance_nested.py          # repeated-CV variance + nested CV
python3 scripts/13_robustness_checks.py        # expression scaling + encoding concordance
python3 scripts/14_data_provenance.py          # SHA-256 data manifest
python3 scripts/15_clean_tables.py             # reader-facing settings + decision tables
python3 scripts/07_make_tables.py              # metrics, bootstrap CIs, decision tables
python3 scripts/08_make_figures.py             # figures 1-4 + supplementary figures
echo "Done. See tables/, figures/, results/."
