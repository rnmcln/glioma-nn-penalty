"""Step 07 - compute metrics, bootstrap CIs, paired NN-vs-transparent comparison,
and write the internal/external performance tables and the decision summary.
"""
import sys, pathlib, glob, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src import config
from src.io_results import load_predictions
from src.evaluation import (point_metrics, bootstrap_ci, paired_bootstrap,
                            meaningful_improvement)
from src.decision_curve import net_benefit_table
from src.models_baseline import TRANSPARENT_MODELS
from src.utils import get_logger, save_json

log = get_logger()
TRANSPARENT_NONTRIVIAL = TRANSPARENT_MODELS - {"base_rate"}


def metrics_for_tag(tag, path):
    y, preds, models = load_predictions(tag)
    rows = []
    for m in models:
        pm = point_metrics(y, preds[m])
        au = bootstrap_ci(y, preds[m], "auroc")
        br = bootstrap_ci(y, preds[m], "brier")
        ap = bootstrap_ci(y, preds[m], "auprc")
        rows.append({
            "setting": tag, "model": m, "n": pm["n"], "event_rate": round(pm["event_rate"], 3),
            "auroc": round(pm["auroc"], 3) if pm["auroc"] == pm["auroc"] else None,
            "auroc_lo": round(au["lo"], 3), "auroc_hi": round(au["hi"], 3),
            "auprc": round(pm["auprc"], 3) if pm["auprc"] == pm["auprc"] else None,
            "brier": round(pm["brier"], 3), "brier_lo": round(br["lo"], 3), "brier_hi": round(br["hi"], 3),
            "calib_intercept": round(pm["calibration_intercept"], 3) if pm["calibration_intercept"]==pm["calibration_intercept"] else None,
            "calib_slope": round(pm["calibration_slope"], 3) if pm["calibration_slope"]==pm["calibration_slope"] else None,
        })
    return y, preds, models, pd.DataFrame(rows)


TF_VARIANTS = ["tf_compact", "tf_medium", "tf_regularised"]


def nn_vs_transparent(tag, y, preds, models):
    tf_present = [m for m in TF_VARIANTS if m in models]
    if not tf_present:
        return None
    cand = [m for m in models if m in TRANSPARENT_NONTRIVIAL]
    if not cand:
        return None
    # best TF variant and best transparent model, each by AUROC in this setting
    best_tf = max(tf_present, key=lambda m: point_metrics(y, preds[m])["auroc"])
    best = max(cand, key=lambda m: point_metrics(y, preds[m])["auroc"])
    pb = paired_bootstrap(y, preds[best_tf], preds[best], "auroc")
    pm_nn = point_metrics(y, preds[best_tf])
    pm_ref = point_metrics(y, preds[best])
    rule = meaningful_improvement(pb["delta"], pm_nn["brier"], pm_ref["brier"], pm_nn["calibration_slope"])
    return {"setting": tag, "best_tf_variant": best_tf, "best_transparent": best,
            "auroc_nn": round(pm_nn["auroc"], 3), "auroc_transparent": round(pm_ref["auroc"], 3),
            "delta_auroc": round(pb["delta"], 3), "delta_lo": round(pb["lo"], 3),
            "delta_hi": round(pb["hi"], 3), "paired_p": round(pb["p_value"], 3),
            "brier_nn": round(pm_nn["brier"], 3), "brier_transparent": round(pm_ref["brier"], 3),
            "calib_slope_nn": round(pm_nn["calibration_slope"], 3),
            **{k: rule[k] for k in ("passes_auroc_margin", "not_worse_brier",
                                    "calibration_acceptable", "clinically_meaningful")}}


def main():
    import time, json as _json
    cache = config.METRICS / "_tabcache"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    # Phase 1: per-setting metrics + decision, cached so the step is resumable.
    for path in sorted(glob.glob(str(config.PREDICTIONS / "*.csv"))):
        tag = pathlib.Path(path).stem
        cf = cache / f"{tag}.json"
        if cf.exists():
            continue
        if time.time() - t0 > 38:
            print("PARTIAL: more settings to tabulate; rerun step 07")
            break
        y, preds, models, mdf = metrics_for_tag(tag, path)
        rec = {"is_internal": tag.startswith("internal"), "rows": mdf.to_dict("records"),
               "decision": nn_vs_transparent(tag, y, preds, models)}
        if tag == f"internal_tcga_{config.PRIMARY_FEATURE_SET}_{config.PRIMARY_ENDPOINT}":
            dca = net_benefit_table(y, preds)
            for r in dca:
                r["setting"] = tag
            rec["dca"] = dca
        _json.dump(rec, open(cf, "w"))

    # Phase 2: assemble from cache (only if all settings cached)
    all_tags = [pathlib.Path(p).stem for p in glob.glob(str(config.PREDICTIONS / "*.csv"))]
    if not all((cache / f"{t}.json").exists() for t in all_tags):
        log.info("tabulation incomplete; rerun step 07 to finish")
        return
    internal, external, decisions, dca_rows = [], [], [], []
    for t in all_tags:
        rec = _json.load(open(cache / f"{t}.json"))
        mdf = pd.DataFrame(rec["rows"])
        (internal if rec["is_internal"] else external).append(mdf)
        if rec.get("decision"):
            decisions.append(rec["decision"])
        if rec.get("dca"):
            dca_rows.extend(rec["dca"])

    if internal:
        pd.concat(internal, ignore_index=True).to_csv(config.TABLES / "table3_internal_performance.csv", index=False)
    if external:
        pd.concat(external, ignore_index=True).to_csv(config.TABLES / "table4_external_performance.csv", index=False)
    if decisions:
        pd.DataFrame(decisions).to_csv(config.TABLES / "table5_nn_vs_transparent.csv", index=False)
        save_json(decisions, config.METRICS / "decision_summary.json")
    if dca_rows:
        pd.DataFrame(dca_rows).to_csv(config.TABLES / "table6_decision_curve.csv", index=False)
    log.info("Tables written to %s", config.TABLES)


if __name__ == "__main__":
    main()
