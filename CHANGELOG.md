# Changelog

## 0.3.3

Public release of the analysis code, locked protocol, processed cohort data,
data dictionary and manifest, results tables, and figures for the glioma
neural-network benchmark.

The pipeline covers internal repeated cross-validation in TCGA, external
validation in Gravendeel and REMBRANDT, time-to-event (Cox / penalised Cox) and
inverse-probability-of-censoring-weighted sensitivity analyses, a landmark
analysis, a modern tabular baseline (FT-Transformer), within-stratum analyses by
IDH status and histology, repeated-CV variance and nested-CV checks, and
expression-scaling and encoding robustness checks.

See `README.md` for environment setup and exact reproduction commands, and
`RESULTS_SUMMARY.md` for the headline findings.
