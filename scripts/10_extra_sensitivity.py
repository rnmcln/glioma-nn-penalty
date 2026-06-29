"""Step 10 - additional sensitivity and transparency analyses.

(a) TCGA included vs censored-before-24-month comparison (supplementary table).
(b) IPCW-weighted AUROC/Brier on the primary B/2-year out-of-fold predictions,
    as an alternative to excluding patients censored before the horizon.
(c) Feature availability / harmonisation matrix across cohorts.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from src import config, pipeline
from src.io_results import load_predictions
from src.models_baseline import TRANSPARENT_MODELS
from src.utils import get_logger

log = get_logger()
HORIZON = config.ENDPOINTS[config.PRIMARY_ENDPOINT]


# (a) included vs censored-before-horizon -------------------------------------
def included_vs_censored():
    df = pipeline.load_cohort("tcga", with_expr=False)
    t = pd.to_numeric(df["os_months"], errors="coerce")
    e = pd.to_numeric(df["os_event"], errors="coerce")
    censored_before = (e == 0) & (t < HORIZON)
    included = ((e == 1) & (t <= HORIZON)) | (t >= HORIZON)
    rows = []
    def summ(mask, label):
        g = df[mask]
        rows.append({
            "group": label, "n": int(mask.sum()),
            "median_age": round(g["age"].median(), 1),
            "pct_GBM": round(100 * (g["histology"] == "GBM").mean(), 1),
            "pct_grade_IV": round(100 * (g["gbm"] == 1).mean(), 1),
            "pct_IDH_mut": round(100 * (g["idh_status"] == "Mutant").mean(), 1),
            "pct_codel": round(100 * (g["codel_1p19q"] == "Codeleted").mean(), 1),
            "pct_MGMT_meth": round(100 * (g["mgmt_status"] == "Methylated").mean(), 1),
            "median_OS_months": round(g["os_months"].median(), 1),
        })
    summ(included, "included_2yr")
    summ(censored_before, "censored_before_24mo")
    pd.DataFrame(rows).to_csv(config.TABLES / "tableS_included_vs_censored.csv", index=False)
    log.info("wrote tableS_included_vs_censored.csv")


# (b) IPCW-weighted metrics ----------------------------------------------------
def ipcw_weights(times, events, horizon):
    """Uno-style IPCW weights using KM of the censoring distribution."""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=(1 - events))  # censoring as the 'event'
    def G(tt):
        return float(kmf.predict(min(tt, horizon)))
    w = np.zeros(len(times))
    for i, (tt, ev) in enumerate(zip(times, events)):
        if ev == 1 and tt <= horizon:
            g = G(tt); w[i] = 1.0 / g if g > 0 else 0.0
        elif tt >= horizon:
            g = G(horizon); w[i] = 1.0 / g if g > 0 else 0.0
        else:
            w[i] = 0.0
    return w


def ipcw_sensitivity():
    tag = f"internal_tcga_{config.PRIMARY_FEATURE_SET}_{config.PRIMARY_ENDPOINT}"
    try:
        y, preds, models = load_predictions(tag)
    except FileNotFoundError:
        log.warning("primary predictions missing; run internal CV first"); return
    df = pipeline.load_cohort("tcga", with_expr=False)
    pred_df = pd.read_csv(config.PREDICTIONS / f"{tag}.csv")
    merged = pred_df.merge(df[["sample", "os_months", "os_event"]], on="sample", how="left")
    w = ipcw_weights(merged["os_months"].values, merged["os_event"].values, HORIZON)
    keep = w > 0
    rows = []
    for m in models:
        p = merged[f"prob__{m}"].values
        yt = merged["y_true"].values
        au = roc_auc_score(yt[keep], p[keep], sample_weight=w[keep])
        br = brier_score_loss(yt[keep], np.clip(p[keep], 1e-7, 1 - 1e-7), sample_weight=w[keep])
        rows.append({"model": m, "ipcw_auroc": round(au, 3), "ipcw_brier": round(br, 3)})
    pd.DataFrame(rows).to_csv(config.TABLES / "tableS_ipcw_primary.csv", index=False)
    log.info("wrote tableS_ipcw_primary.csv")


# (c) feature availability matrix ---------------------------------------------
def feature_availability():
    rows = []
    feats = ["age", "sex", "grade", "histology", "idh_status", "codel_1p19q", "mgmt_status"]
    for c in ["tcga", "gravendeel", "rembrandt"]:
        df = pipeline.load_cohort(c, with_expr=False)
        r = {"cohort": c}
        for f in feats:
            frac = df[f].notna().mean() if f in df.columns else 0.0
            r[f] = "available" if frac > 0.6 else ("partial" if frac > 0.05 else "absent")
        r["expression_panel"] = "available"
        rows.append(r)
    pd.DataFrame(rows).to_csv(config.TABLES / "tableS_feature_availability.csv", index=False)
    log.info("wrote tableS_feature_availability.csv")


if __name__ == "__main__":
    included_vs_censored()
    feature_availability()
    ipcw_sensitivity()
