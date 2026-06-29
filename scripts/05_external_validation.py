"""Step 05 - external validation: train on full TCGA, test on REMBRANDT and Gravendeel.

Features are automatically restricted to those available (not all-missing) in
BOTH the training and target cohort, so each external test is fair. The set of
shared features actually used is recorded for transparent reporting.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import config, pipeline
from src.io_results import save_predictions
from src.utils import get_logger, set_global_seed, save_json

log = get_logger()


def main():
    set_global_seed()
    endpoint = os.environ.get("GNN_ENDPOINT", config.PRIMARY_ENDPOINT)
    horizon = config.ENDPOINTS[endpoint]
    include_tf = os.environ.get("GNN_TF", "1") == "1"

    tcga = pipeline.load_cohort("tcga")
    targets = {"rembrandt": pipeline.load_cohort("rembrandt"),
               "gravendeel": pipeline.load_cohort("gravendeel")}

    for fs in ["A_clinical", "B_clinical_molecular", "C_expression"]:
        feats = pipeline.resolve_features(fs, tcga)
        for tname, tdf in targets.items():
            tag0 = f"external_tcga_to_{tname}_{fs}_{endpoint}"
            if (config.PREDICTIONS / f"{tag0}.csv").exists() and "--force" not in sys.argv:
                log.info("skip existing %s", tag0); continue
            res = pipeline.external_validate(tcga, tdf, feats, horizon, include_tf=include_tf)
            if len(res["shared_features"]) == 0:
                log.info("Skip %s->%s set=%s: no shared features", "tcga", tname, fs)
                continue
            tag = f"external_tcga_to_{tname}_{fs}_{endpoint}"
            save_predictions(tag, res["y_true"], res["preds"])
            save_json({"setting": tag, "shared_features": res["shared_features"],
                       "n_train": res["n_train"], "n_test": res["n_test"],
                       "train_info": res["train_info"], "test_info": res["test_info"],
                       "models": res["models"]}, config.METRICS / f"{tag}_meta.json")
            log.info("Saved %s | shared=%s n_test=%d", tag, res["shared_features"][:6], res["n_test"])


if __name__ == "__main__":
    main()
