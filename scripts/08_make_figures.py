"""Step 08 - manuscript and supplementary figures (revised).

Main: Fig1 design; Fig2 AUROC forest across internal+external; Fig3 discrimination
within molecular strata; Fig4 calibration + decision curve in the main external
cohort (Gravendeel). Supplement: internal stacked performance; all-model
calibration and decision curves; neural-network learning curves.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config, plotting, pipeline
from src.io_results import load_predictions
from src.evaluation import point_metrics
from src.endpoints import make_binary_endpoint
from src.utils import get_logger, set_global_seed

log = get_logger()
t3 = pd.read_csv(config.TABLES / "table3_internal_performance.csv")
t4 = pd.read_csv(config.TABLES / "table4_external_performance.csv")
COH_MODELS = ["base_rate", "logistic_age_grade", "penalised_logistic", "random_forest",
              "hist_gboost", "tf_compact", "tf_medium", "tf_regularised"]


def row_ci(df, setting, model):
    r = df[(df.setting == setting) & (df.model == model)]
    if len(r) == 0:
        return None
    r = r.iloc[0]
    return {"auroc": float(r["auroc"]), "lo": float(r["auroc_lo"]), "hi": float(r["auroc_hi"])}


def main():
    set_global_seed()
    plotting.figure_study_design(); log.info("fig1")

    # Fig 2: forest across cohorts
    rows = []
    cohmap = [("TCGA (internal)", t3, "internal_tcga_B_clinical_molecular_os_2y"),
              ("Gravendeel (external)", t4, "external_tcga_to_gravendeel_B_clinical_molecular_os_2y"),
              ("REMBRANDT (external)", t4, "external_tcga_to_rembrandt_B_clinical_molecular_os_2y")]
    for coh, df, setting in cohmap:
        for m in COH_MODELS:
            ci = row_ci(df, setting, m)
            if ci:
                rows.append({"cohort": coh, "model": m, **ci})
    plotting.figure_forest_cohorts(rows); log.info("fig2")

    # Fig 3: strata
    srows = []
    stratamap = [("Pooled (LGG+GBM)", "internal_tcga_B_clinical_molecular_os_2y"),
                 ("IDH-wildtype", "internal_tcga_B_idhwt_os_2y"),
                 ("IDH-mutant", "internal_tcga_B_idhmut_os_5y"),
                 ("Glioblastoma", "internal_tcga_B_gbm_os_1y")]
    for st, setting in stratamap:
        for m in COH_MODELS:
            ci = row_ci(t3, setting, m)
            if ci:
                srows.append({"stratum": st, "model": m, **ci})
    plotting.figure_strata(srows); log.info("fig3")

    # Fig 4: calibration + DCA in Gravendeel
    yg, pg, mg = load_predictions("external_tcga_to_gravendeel_B_clinical_molecular_os_2y")
    pa_g = {m: point_metrics(yg, pg[m])["auroc"] for m in mg}
    plotting.figure_calib_dca(yg, pg, pa_g, "Gravendeel (external)", stem="figure4_calibration_dca"); log.info("fig4")

    # Supplement: internal stacked + all-model calib/DCA
    yi, pi, mi = load_predictions("internal_tcga_B_clinical_molecular_os_2y")
    perf = {}
    for m in mi:
        ci = row_ci(t3, "internal_tcga_B_clinical_molecular_os_2y", m)
        br = t3[(t3.setting == "internal_tcga_B_clinical_molecular_os_2y") & (t3.model == m)].iloc[0]
        perf[m] = {"auroc": {"point": ci["auroc"], "lo": ci["lo"], "hi": ci["hi"]},
                   "brier": {"point": float(br["brier"]), "lo": float(br["brier_lo"]), "hi": float(br["brier_hi"])}}
    plotting.figure_internal_stacked(perf)
    plotting.figure_all_calibration(yi, pi)
    plotting.figure_all_dca(yi, pi)
    log.info("supp calib/dca/internal")

    # Supplement: learning curves (fit 3 architectures on full TCGA set B)
    try:
        from src.models_tensorflow import TFMLPClassifier
        tcga = pipeline.load_cohort("tcga"); feats = pipeline.resolve_features("B_clinical_molecular", tcga)
        lab, _ = make_binary_endpoint(tcga, 24)
        hist = {}
        for arch in ("compact", "medium", "regularised"):
            clf = TFMLPClassifier(features=feats, arch=arch, verbose=0).fit(lab, lab["y"].values)
            hist[arch] = clf.model_.history.history
        plotting.figure_learning_curves(hist); log.info("supp learning curves")
    except Exception as e:
        log.warning("learning curves skipped: %s", e)


if __name__ == "__main__":
    main()
