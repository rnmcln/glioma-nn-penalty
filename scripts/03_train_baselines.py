"""Step 03 - internal repeated cross-validation in TCGA.

Runs ALL models (transparent baselines, standard ML, and the small TF MLP) on
IDENTICAL folds so the comparison is fair, and saves out-of-fold predictions.
Feature sets and endpoint are configurable via environment variables:
    GNN_SETS   (default 'A_clinical,B_clinical_molecular,C_expression')
    GNN_ENDPOINT (default primary, 'os_2y')
    GNN_TF     (default '1'; set '0' to skip the neural network)
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import config, pipeline
from src.io_results import save_predictions
from src.utils import get_logger, set_global_seed, save_json

log = get_logger()


def main():
    set_global_seed()
    sets = os.environ.get("GNN_SETS", "A_clinical,B_clinical_molecular,C_expression").split(",")
    endpoint = os.environ.get("GNN_ENDPOINT", config.PRIMARY_ENDPOINT)
    horizon = config.ENDPOINTS[endpoint]
    include_tf = os.environ.get("GNN_TF", "1") == "1"

    df = pipeline.load_cohort("tcga")
    for fs in sets:
        feats = pipeline.resolve_features(fs, df)
        log.info("Internal CV | set=%s endpoint=%s features=%d (%s)", fs, endpoint, len(feats), feats[:6])
        res = pipeline.internal_cv(df, feats, horizon, include_tf=include_tf)
        tag = f"internal_tcga_{fs}_{endpoint}"
        save_predictions(tag, res["y_true"], res["preds"], sample=res["sample"])
        save_json({"setting": tag, "features": feats, "n": res["n"],
                   "endpoint_info": res["endpoint_info"], "models": res["models"]},
                  config.METRICS / f"{tag}_meta.json")
        log.info("Saved predictions: %s (n=%d)", tag, res["n"])


if __name__ == "__main__":
    main()
