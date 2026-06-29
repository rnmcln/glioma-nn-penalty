"""Step 09 - secondary time-to-event benchmark (Cox / penalised Cox, optional RSF).

Internal repeated-CV C-index in TCGA and external C-index in REMBRANDT and
Gravendeel, on feature sets A and B. Uses full follow-up (no horizon exclusion).
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config, pipeline
from src.models_survival import cox_internal_cv, cox_external, _HAS_SKSURV
from src.utils import get_logger, set_global_seed

log = get_logger()


def main():
    set_global_seed()
    tcga = pipeline.load_cohort("tcga")
    targets = {"rembrandt": pipeline.load_cohort("rembrandt"),
               "gravendeel": pipeline.load_cohort("gravendeel")}

    rows = []
    for fs in ["A_clinical", "B_clinical_molecular"]:
        feats = pipeline.resolve_features(fs, tcga)
        internal = cox_internal_cv(tcga, feats)
        for model, (mean, sd) in internal.items():
            rows.append({"analysis": "internal_cv", "feature_set": fs, "model": model,
                         "cindex": round(mean, 3), "cindex_sd": round(sd, 3), "target": "TCGA"})
        for tname, tdf in targets.items():
            ext = cox_external(tcga, tdf, feats)
            for model in ("cox", "cox_penalised"):
                rows.append({"analysis": "external", "feature_set": fs, "model": model,
                             "cindex": round(ext[model], 3), "cindex_sd": None,
                             "target": tname, "shared_features": ";".join(ext["shared_features"])})
        log.info("survival done | set=%s", fs)

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES / "table7_survival_cindex.csv", index=False)
    log.info("Wrote table7_survival_cindex.csv (sksurv=%s)", _HAS_SKSURV)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
