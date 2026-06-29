"""Step 11 - modern tabular baseline (FT-Transformer-lite), focused robustness.

Head-to-head on identical folds for the primary setting (TCGA, set B, 2-year):
FT-Transformer vs penalised logistic regression vs the best MLP (regularised).
Also external (Gravendeel, REMBRANDT). 5-fold CV, single repeat, shared folds.
Resumable: writes per-fold checkpoint; rerun until it prints DONE.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import config, pipeline
from src.endpoints import make_binary_endpoint
from src.models_baseline import build_baselines
from src.models_tensorflow import TFMLPClassifier
from src.models_tabular import FTTransformerClassifier
from src.evaluation import point_metrics
from src.io_results import save_predictions
from src.utils import get_logger, set_global_seed, save_json

log = get_logger()
CK = config.PREDICTIONS / "_ckpt"; CK.mkdir(parents=True, exist_ok=True)


def models_for(df, feats):
    return {
        "penalised_logistic": build_baselines(df, feats)["penalised_logistic"],
        "tf_regularised": TFMLPClassifier(features=feats, arch="regularised"),
        "ft_transformer": FTTransformerClassifier(features=feats),
    }


def main():
    set_global_seed()
    tcga = pipeline.load_cohort("tcga"); feats = pipeline.resolve_features("B_clinical_molecular", tcga)
    lab, info = make_binary_endpoint(tcga, 24); lab = lab.reset_index(drop=True)
    y = lab["y"].values.astype(int); n = len(lab)
    names = list(models_for(lab, feats).keys())
    ckf = CK / "ftt_primary.npz"
    if ckf.exists():
        d = np.load(ckf, allow_pickle=True)
        psum = {m: d[f"s__{m}"] for m in names}; pcnt = {m: d[f"c__{m}"] for m in names}; done = set(d["folds"].tolist())
    else:
        psum = {m: np.zeros(n) for m in names}; pcnt = {m: np.zeros(n) for m in names}; done = set()
    skf = list(StratifiedKFold(5, shuffle=True, random_state=config.SEED).split(lab, y))
    import time; t0 = time.time()
    for fi, (tr, te) in enumerate(skf):
        if fi in done:
            continue
        if time.time() - t0 > 34 and len(done) > 0:
            break
        ms = models_for(lab.iloc[tr], feats)
        for m, est in ms.items():
            est.fit(lab.iloc[tr], y[tr]); psum[m][te] += est.predict_proba(lab.iloc[te])[:, 1]; pcnt[m][te] += 1
        done.add(fi)
        np.savez(ckf, folds=np.array(sorted(done)), **{f"s__{m}": psum[m] for m in names}, **{f"c__{m}": pcnt[m] for m in names})
        log.info("primary fold %d done", fi)
    if len(done) < 5:
        print(f"PARTIAL primary {len(done)}/5"); return
    preds = {m: psum[m] / np.maximum(pcnt[m], 1) for m in names}
    save_predictions("modern_internal_tcga_B_2y", y, preds, sample=lab["sample"].values)

    # external (cheap, single fit)
    rows = []
    for m in names:
        pm = point_metrics(y, preds[m]); rows.append({"setting": "internal_B_2y", "model": m, "auroc": round(pm["auroc"], 3), "brier": round(pm["brier"], 3), "calib_slope": round(pm["calibration_slope"], 3)})
    for tname in ("gravendeel", "rembrandt"):
        tdf = pipeline.load_cohort(tname)
        shared = [f for f in feats if f in tdf.columns and tdf[f].notna().any()]
        te_lab, _ = make_binary_endpoint(tdf, 24); yte = te_lab["y"].values
        ms = models_for(lab, shared)
        ext = {}
        for m, est in ms.items():
            est.fit(lab, y); ext[m] = est.predict_proba(te_lab)[:, 1]
            pm = point_metrics(yte, ext[m]); rows.append({"setting": f"external_{tname}_B_2y", "model": m, "auroc": round(pm["auroc"], 3), "brier": round(pm["brier"], 3), "calib_slope": round(pm["calibration_slope"], 3)})
        save_predictions(f"modern_external_{tname}_B_2y", yte, ext)
    pd.DataFrame(rows).to_csv(config.TABLES / "tableS_modern_tabular.csv", index=False)
    ckf.unlink(missing_ok=True)
    print("DONE")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
