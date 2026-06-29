"""Step 02 - harmonise cohorts and write the data dictionary + cohort/missingness tables."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
from src.data_harmonise import harmonise_all
from src.endpoints import make_binary_endpoint
from src.utils import get_logger

log = get_logger()

DICTIONARY = [
    ("cohort", "Source cohort: tcga | rembrandt | gravendeel", "derived"),
    ("sample", "Sample identifier (GlioVis)", "available"),
    ("age", "Age at diagnosis (years)", "available (TCGA, Gravendeel); missing (REMBRANDT)"),
    ("sex", "Sex (Male/Female)", "available (TCGA, Gravendeel); missing (REMBRANDT)"),
    ("grade", "WHO grade I-IV (ordinal 1-4)", "available; partly missing (REMBRANDT)"),
    ("grade_group", "high (III-IV) vs lower (I-II)", "derived"),
    ("gbm", "Grade IV / GBM indicator", "derived"),
    ("histology", "Harmonised histology (GBM/Astrocytoma/Oligodendroglioma/Mixed-OA)", "available"),
    ("idh_status", "IDH mutation status (Mutant/WT)", "available (TCGA); partial (Gravendeel); missing (REMBRANDT)"),
    ("codel_1p19q", "1p/19q codeletion (Codeleted/Non-codeleted)", "available (TCGA); derived/partial (Gravendeel); missing (REMBRANDT)"),
    ("mgmt_status", "MGMT promoter methylation", "available (TCGA); missing (REMBRANDT, Gravendeel)"),
    ("os_months", "Overall survival time (months)", "available"),
    ("os_event", "Vital status (1=death, 0=alive/censored)", "available"),
]


def main():
    combined, summary = harmonise_all(save=True)
    log.info("Harmonisation summary:\n%s", summary.to_string(index=False))

    # data dictionary
    pd.DataFrame(DICTIONARY, columns=["variable", "description", "availability"]).to_csv(
        config.DICTIONARY / "data_dictionary.csv", index=False)

    # Table 1 (cohort description) and Table 2 (missingness)
    rows, miss_rows = [], []
    key_vars = ["age", "sex", "grade", "histology", "idh_status", "codel_1p19q", "mgmt_status"]
    for c, g in combined.groupby("cohort"):
        _, info = make_binary_endpoint(g, config.ENDPOINTS[config.PRIMARY_ENDPOINT])
        rows.append({
            "cohort": c, "n": len(g),
            "median_age": round(g["age"].median(), 1) if g["age"].notna().any() else None,
            "pct_female": round(100 * (g["sex"] == "Female").mean(), 1) if g["sex"].notna().any() else None,
            "pct_GBM": round(100 * (g["histology"] == "GBM").mean(), 1),
            "pct_grade_IV": round(100 * (g["gbm"] == 1).mean(), 1) if g["gbm"].notna().any() else None,
            "pct_IDH_mut": round(100 * (g["idh_status"] == "Mutant").mean(), 1) if g["idh_status"].notna().any() else None,
            "median_OS_months": round(g["os_months"].median(), 1),
            "n_2yr_labelled": info["n_labelled"],
            "n_2yr_deaths": info["n_events_death"],
            "event_rate_2yr": round(info["event_rate"], 3),
            "n_censored_excluded_2yr": info["n_censored_before_horizon_excluded"],
        })
        for v in key_vars:
            miss_rows.append({"cohort": c, "variable": v,
                              "missing_fraction": round(g[v].isna().mean(), 3)})
    pd.DataFrame(rows).to_csv(config.TABLES / "table1_cohort.csv", index=False)
    pd.DataFrame(miss_rows).pivot(index="variable", columns="cohort",
                                  values="missing_fraction").to_csv(config.TABLES / "table2_missingness.csv")
    log.info("Wrote table1_cohort.csv and table2_missingness.csv")


if __name__ == "__main__":
    main()
