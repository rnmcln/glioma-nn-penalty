"""Driver for full-scale internal CV across all settings (resumable).

Invoke repeatedly; each call advances incomplete settings within a wall-clock
budget and checkpoints. A setting is complete when its meta JSON records the
target CV depth. Run until it prints ALL_COMPLETE.
"""
import os, sys, time, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import config, pipeline
from src.cv_runner import run_setting
from src.utils import set_global_seed, get_logger

log = get_logger()
TARGET = f"{config.CV_REPEATS}x{config.CV_FOLDS}"


def jobs():
    tcga = pipeline.load_cohort("tcga")
    fb = pipeline.resolve_features("B_clinical_molecular", tcga)
    J = [
        ("internal_tcga_A_clinical_os_2y", tcga, pipeline.resolve_features("A_clinical", tcga), 24),
        ("internal_tcga_B_clinical_molecular_os_2y", tcga, fb, 24),
        ("internal_tcga_C_expression_os_2y", tcga, pipeline.resolve_features("C_expression", tcga), 24),
        ("internal_tcga_B_clinical_molecular_os_1y", tcga, fb, 12),
        ("internal_tcga_B_clinical_molecular_os_5y", tcga, fb, 60),
        ("internal_tcga_B_clinical_molecular_os_2y_grade-high",
         tcga[tcga["grade_group"] == "high"].reset_index(drop=True), fb, 24),
        ("internal_tcga_B_clinical_molecular_os_2y_grade-lower",
         tcga[tcga["grade_group"] == "lower"].reset_index(drop=True), fb, 24),
        ("internal_tcga_B_clinical_molecular_os_2y_completecase",
         tcga.dropna(subset=fb).reset_index(drop=True), fb, 24),
        # Within-stratum analyses: the genuinely hard prognostic questions, testing
        # whether any model (especially the NN) adds value beyond molecular class.
        ("internal_tcga_B_idhwt_os_2y",
         tcga[tcga["idh_status"] == "WT"].reset_index(drop=True), fb, 24),
        ("internal_tcga_B_idhwt_os_1y",
         tcga[tcga["idh_status"] == "WT"].reset_index(drop=True), fb, 12),
        ("internal_tcga_B_idhmut_os_5y",
         tcga[tcga["idh_status"] == "Mutant"].reset_index(drop=True), fb, 60),
        ("internal_tcga_B_gbm_os_1y",
         tcga[tcga["histology"] == "GBM"].reset_index(drop=True), fb, 12),
        # Landmark at 6 months (at-risk at 6 months), 2-year endpoint.
        ("internal_tcga_B_landmark6m_os_2y",
         tcga[tcga["os_months"] >= 6].reset_index(drop=True), fb, 24),
    ]
    return J


def is_done(tag):
    # complete when final out-of-fold predictions exist and no checkpoint remains
    pred = config.PREDICTIONS / f"{tag}.csv"
    ck = config.PREDICTIONS / "_ckpt" / f"{tag}.npz"
    return pred.exists() and not ck.exists()


def main():
    set_global_seed()
    t0 = time.time()
    total_budget = int(os.environ.get("GNN_BUDGET", 34))
    any_partial = False
    for tag, df, feats, horizon in jobs():
        if is_done(tag):
            continue
        remaining = total_budget - (time.time() - t0)
        if remaining < 6:
            any_partial = True
            break
        res = run_setting(tag, df, feats, horizon, budget_s=remaining)
        log.info("%s -> %s (%s/%s repeats)", tag, res["status"], res["repeats"], res.get("target", config.CV_REPEATS))
        if res["status"] == "partial":
            any_partial = True
            break
    remaining_jobs = [t for (t, *_ ) in jobs() if not is_done(t)]
    if not remaining_jobs:
        print("ALL_COMPLETE")
    else:
        print(f"REMAINING: {len(remaining_jobs)} -> {remaining_jobs[0]}")


if __name__ == "__main__":
    main()
