# Results summary (v0.2, full-scale run)

Real public data. Internal validation: 5 repeats x 5-fold CV for primary
feature-set analyses (A, B, C at 2 years); 3 repeats x 5-fold for secondary
settings; 2000 bootstrap replicates throughout. Three prespecified TensorFlow
MLPs (compact, medium, regularised) plus transparent and standard-ML models.

## Cohorts (adult diffuse glioma with outcome)

| Cohort | role | n | 2-yr labelled | deaths | event rate | censored<24mo excl. |
|---|---|---|---|---|---|---|
| TCGA | development | 667 | 406 | 156 | 0.38 | 261 |
| Gravendeel | main external | 264 | 260 | 167 | 0.64 | 4 |
| REMBRANDT | stress test | 342 | 341 | 202 | 0.59 | 1 |

## Primary internal result (TCGA, set B, 2-year OS)

| Model | AUROC (95% CI) | Brier | Calib. intercept | Calib. slope |
|---|---|---|---|---|
| Penalised LR (best transparent) | 0.916 (0.886–0.942) | 0.106 | 0.00 | 0.98 |
| Regularised MLP (best NN) | 0.913 (0.882–0.941) | 0.108 | -0.33 | 1.11 |
| Compact MLP | 0.911 | 0.108 | -0.35 | 1.05 |
| Medium MLP | 0.905 | 0.118 | -0.32 | 1.24 |
| Random forest | 0.907 | 0.110 | -0.01 | 1.05 |
| Gradient boosting | 0.907 | 0.110 | -0.02 | 0.80 |
| Age+grade LR | 0.896 | 0.123 | 0.00 | 1.01 |

Best NN vs best transparent: ΔAUROC -0.003 (95% CI -0.009 to 0.004).

## Within-stratum collapse

Pooled AUROC ~0.91 falls to **approximately 0.56–0.69 within molecular strata**, with no model
class superior (overlapping CIs):

| Stratum | n | event rate | penalised LR | best MLP |
|---|---|---|---|---|
| Pooled (set B, 2y) | 406 | 0.38 | 0.916 | 0.913 |
| IDH-wildtype, 2y | 173 | 0.79 | 0.653 | 0.681 |
| IDH-mutant, 5y | 110 | 0.45 | 0.692 | 0.647 |
| Glioblastoma, 1y | 128 | 0.46 | 0.609 | 0.607 |

So pooled discrimination largely reflects recovery of molecular class.

## Modern tabular baseline and robustness

- FT-Transformer: internal AUROC 0.891 (vs penalised LR 0.917); no advantage.
- Nested-CV penalised LR 0.917 (vs 0.915 fixed); repeated-CV AUROC SD ≤0.004.
- Expression scaling scheme: ΔAUROC ≤0.015; encoding/padding concordance: 0.001.

## Prespecified decision rule

Met by a neural network in **0 of 18 settings** (feature sets A/B/C; 1/2/5-year;
grade-high/lower; complete-case; Gravendeel and REMBRANDT external). The only
settings with ΔAUROC >= 0.03 were the REMBRANDT clinical sets (grade+histology
only); these failed the composite rule because calibration/accuracy did not
improve.

## External validation (AUROC)

- Gravendeel (main): penalised LR ~0.82–0.83; best MLP ~0.82–0.83 (no advantage).
- REMBRANDT (stress test, grade+histology): penalised LR 0.68; medium MLP 0.72
  (ranking gain only; calibration not improved).

## Censoring and robustness

- Excluded (censored <24mo) TCGA patients were younger, less often GBM, more often
  IDH-mutant, with shorter follow-up (Table S2): a non-random exclusion.
- IPCW-weighted primary analysis matched the exclusion-based result (penalised LR
  AUROC 0.916; best MLP ~0.91).

## Time-to-event benchmark (C-index, penalised Cox)

TCGA internal ~0.85; Gravendeel ~0.69; REMBRANDT ~0.61. Consistent with the
binary analysis.

## Interpretation

No prespecified neural network added clinically meaningful, calibrated, or
externally valid value over transparent models. Penalised logistic regression was
at least as discriminating and better calibrated; net benefit was comparable.
