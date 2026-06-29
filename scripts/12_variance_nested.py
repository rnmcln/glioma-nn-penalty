"""Step 12 - per-repeat (repeated-CV) variance and a nested-CV sensitivity.

(a) Per-repeat AUROC and calibration slope for three representative models on the
    primary setting, reported as mean +/- SD across repeats (addresses the concern
    that pooled out-of-fold bootstrap CIs ignore resampling variance).
(b) Nested cross-validation for elastic-net penalised logistic regression (inner
    tuning of C and l1_ratio), to show that tuning the best transparent model does
    not change the ranking (addresses the fixed-hyperparameter concern).
Resumable via a per-repeat checkpoint.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src import config, pipeline
from src.endpoints import make_binary_endpoint
from src.models_baseline import build_baselines
from src.models_tensorflow import TFMLPClassifier
from src.preprocessing import make_preprocessor
from src.evaluation import point_metrics
from src.utils import get_logger, set_global_seed

log = get_logger()
CK = config.PREDICTIONS / "_ckpt"; CK.mkdir(parents=True, exist_ok=True)
REPS = 5


def reps_models(df, feats):
    return {"penalised_logistic": build_baselines(df, feats)["penalised_logistic"],
            "hist_gboost": build_baselines(df, feats)["hist_gboost"],
            "tf_regularised": TFMLPClassifier(features=feats, arch="regularised")}


def per_repeat(lab, feats, y):
    ck = CK / "perrep.json"
    state = json.load(open(ck)) if ck.exists() else {"done": [], "rows": []}
    done = set(state["done"]); rows = state["rows"]; t0 = time.time()
    names = list(reps_models(lab, feats).keys())
    for rep in range(REPS):
        if rep in done:
            continue
        if time.time() - t0 > 32 and len(done) > 0:
            break
        skf = StratifiedKFold(5, shuffle=True, random_state=config.SEED + rep)
        oof = {m: np.zeros(len(lab)) for m in names}
        for tr, te in skf.split(lab, y):
            ms = reps_models(lab.iloc[tr], feats)
            for m, est in ms.items():
                est.fit(lab.iloc[tr], y[tr]); oof[m][te] = est.predict_proba(lab.iloc[te])[:, 1]
        for m in names:
            pm = point_metrics(y, oof[m]); rows.append({"repeat": rep, "model": m,
                "auroc": pm["auroc"], "calib_slope": pm["calibration_slope"], "brier": pm["brier"]})
        done.add(rep)
        json.dump({"done": sorted(done), "rows": rows}, open(ck, "w"))
        log.info("per-repeat %d done", rep)
    return len(done) >= REPS, rows, ck


def main():
    set_global_seed()
    tcga = pipeline.load_cohort("tcga"); feats = pipeline.resolve_features("B_clinical_molecular", tcga)
    lab, _ = make_binary_endpoint(tcga, 24); lab = lab.reset_index(drop=True); y = lab["y"].values.astype(int)

    finished, rows, ck = per_repeat(lab, feats, y)
    if not finished:
        print("PARTIAL per-repeat; rerun step 12"); return
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        slope_mean=("calib_slope", "mean"), slope_sd=("calib_slope", "std"),
        brier_mean=("brier", "mean"), brier_sd=("brier", "std")).round(3).reset_index()
    summ.to_csv(config.TABLES / "tableS_per_repeat_variance.csv", index=False)
    log.info("wrote tableS_per_repeat_variance.csv")

    # (b) nested CV for penalised LR
    pre = make_preprocessor(lab, feats)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=config.SEED))])
    grid = {"clf__C": [0.1, 1.0, 10.0], "clf__l1_ratio": [0.2, 0.5, 0.8]}
    outer = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    aurocs = []
    for tr, te in outer.split(lab, y):
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=-1)
        gs.fit(lab.iloc[tr], y[tr])
        p = gs.predict_proba(lab.iloc[te])[:, 1]
        aurocs.append(point_metrics(y[te], p)["auroc"])
    nested = float(np.mean(aurocs))
    fixed = float(df[df.model == "penalised_logistic"]["auroc"].mean())
    pd.DataFrame([{"model": "penalised_logistic_nested", "nested_cv_auroc": round(nested, 3),
                   "fixed_hyperparam_auroc": round(fixed, 3),
                   "difference": round(nested - fixed, 3)}]).to_csv(config.TABLES / "tableS_nested_cv.csv", index=False)
    ck.unlink(missing_ok=True)
    print("DONE")
    print(summ.to_string(index=False))
    print(f"Nested-CV penalised LR AUROC {nested:.3f} vs fixed {fixed:.3f}")


if __name__ == "__main__":
    main()
