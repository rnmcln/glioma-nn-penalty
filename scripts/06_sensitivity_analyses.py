"""Step 06 - sensitivity analyses (internal TCGA unless noted).

Covers: alternative endpoints (1-yr, 5-yr); high-grade vs lower-grade subgroups;
complete-case vs imputed. Predictions are saved with descriptive tags and are
picked up automatically by step 07 for tabulation. Configure which analyses run
via GNN_SENS (comma list of: endpoints,subgroups,completecase). Default: all.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import config, pipeline
from src.io_results import save_predictions
from src.utils import get_logger, set_global_seed, save_json

log = get_logger()
FS = "B_clinical_molecular"


def run_and_save(df, feats, horizon, tag):
    res = pipeline.internal_cv(df, feats, horizon, include_tf=True)
    save_predictions(tag, res["y_true"], res["preds"], sample=res["sample"])
    save_json({"setting": tag, "features": feats, "endpoint_info": res["endpoint_info"],
               "models": res["models"]}, config.METRICS / f"{tag}_meta.json")
    log.info("Saved %s (n=%d)", tag, res["n"])


def main():
    set_global_seed()
    which = os.environ.get("GNN_SENS", "endpoints,subgroups,completecase").split(",")
    df = pipeline.load_cohort("tcga")
    feats = pipeline.resolve_features(FS, df)

    if "endpoints" in which:
        for ep in ("os_1y", "os_5y"):
            run_and_save(df, feats, config.ENDPOINTS[ep], f"internal_tcga_{FS}_{ep}")

    if "subgroups" in which:
        for grp, sub in (("high", df[df["grade_group"] == "high"]),
                         ("lower", df[df["grade_group"] == "lower"])):
            if len(sub) >= 80:
                run_and_save(sub.reset_index(drop=True), feats,
                             config.ENDPOINTS[config.PRIMARY_ENDPOINT],
                             f"internal_tcga_{FS}_{config.PRIMARY_ENDPOINT}_grade-{grp}")

    if "completecase" in which:
        cc = df.dropna(subset=feats).reset_index(drop=True)
        run_and_save(cc, feats, config.ENDPOINTS[config.PRIMARY_ENDPOINT],
                     f"internal_tcga_{FS}_{config.PRIMARY_ENDPOINT}_completecase")


if __name__ == "__main__":
    main()
