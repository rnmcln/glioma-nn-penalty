"""Step 13 - two robustness checks requested in review.

(a) Expression scaling for external transfer (R3.3): compare per-cohort z-scoring
    (default) against fitting the scaler on TCGA and applying it to the target.
(b) Encoding/padding concordance (R3.4): compare the fast pipeline (fixed
    categorical levels; fixed-shape padding for the MLP) against a standard
    per-fold pipeline (learned categories; no padding) on the primary setting.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from src import config, pipeline
from src.endpoints import make_binary_endpoint
from src.io_results import load_predictions
from src.evaluation import point_metrics
from src.utils import get_logger, set_global_seed

log = get_logger()


def expr_scaling_check():
    import pandas as pd
    clin_t = pd.read_csv(config.PROCESSED / "clinical_tcga.csv")
    et = pd.read_csv(config.PROCESSED / "expr_panel_tcga.csv").rename(columns={"Sample": "sample"})
    genes = [c for c in et.columns if c != "sample"]
    rows = []
    for tname in ("gravendeel", "rembrandt"):
        clin_x = pd.read_csv(config.PROCESSED / f"clinical_{tname}.csv")
        ex = pd.read_csv(config.PROCESSED / f"expr_panel_{tname}.csv").rename(columns={"Sample": "sample"})
        for scheme in ("per_cohort_z", "train_fit_scaler"):
            if scheme == "per_cohort_z":
                Xt = (et[genes] - et[genes].mean()) / et[genes].std(ddof=0)
                Xx = (ex[genes] - ex[genes].mean()) / ex[genes].std(ddof=0)
            else:
                mu, sd = et[genes].mean(), et[genes].std(ddof=0)
                Xt = (et[genes] - mu) / sd
                Xx = (ex[genes] - mu) / sd  # TCGA scaler applied to target
            tr = clin_t.merge(et[["sample"]].assign(**{g: Xt[g].values for g in genes}), on="sample")
            te = clin_x.merge(ex[["sample"]].assign(**{g: Xx[g].values for g in genes}), on="sample")
            ytr_lab, _ = make_binary_endpoint(tr, 24); yte_lab, _ = make_binary_endpoint(te, 24)
            clf = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000)
            Xtr = ytr_lab[genes].fillna(0).values; Xte = yte_lab[genes].fillna(0).values
            clf.fit(Xtr, ytr_lab["y"].values)
            p = clf.predict_proba(Xte)[:, 1]
            rows.append({"check": "expr_scaling", "target": tname, "scheme": scheme,
                         "auroc": round(point_metrics(yte_lab["y"].values, p)["auroc"], 3)})
    return rows


def encoding_concordance():
    """Standard per-fold pipeline (learned categories, no padding) vs saved fast run."""
    set_global_seed()
    tcga = pipeline.load_cohort("tcga"); feats = pipeline.resolve_features("B_clinical_molecular", tcga)
    lab, _ = make_binary_endpoint(tcga, 24); lab = lab.reset_index(drop=True); y = lab["y"].values.astype(int)
    num = [f for f in feats if pd.api.types.is_numeric_dtype(lab[f])]
    cat = [f for f in feats if f not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])  # learned categories
    oof = np.zeros(len(lab))
    skf = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    for tr, te in skf.split(lab, y):
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000))])
        pipe.fit(lab.iloc[tr], y[tr]); oof[te] = pipe.predict_proba(lab.iloc[te])[:, 1]
    std_auroc = point_metrics(y, oof)["auroc"]
    # saved fast-pipeline value
    ys, preds, _ = load_predictions("internal_tcga_B_clinical_molecular_os_2y")
    fast_auroc = point_metrics(ys, preds["penalised_logistic"])["auroc"]
    return [{"check": "encoding_concordance", "model": "penalised_logistic",
             "standard_per_fold_auroc": round(std_auroc, 3),
             "fast_pipeline_auroc": round(fast_auroc, 3),
             "difference": round(std_auroc - fast_auroc, 3)}]


def main():
    rows = expr_scaling_check() + encoding_concordance()
    pd.DataFrame(rows).to_csv(config.TABLES / "tableS_robustness.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    log.info("wrote tableS_robustness.csv")


if __name__ == "__main__":
    main()
