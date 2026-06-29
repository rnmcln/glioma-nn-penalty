"""Harmonise TCGA, REMBRANDT, and Gravendeel into a common schema.

Primary harmonised source: the GlioVis data objects (.Rds), which encode all
three cohorts with an identical clinical schema and gene-symbol expression
matrices. This is the cleanest reproducible route and is cross-checked against
the cBioPortal TCGA download for the molecular variables (see scripts/02).

Harmonised clinical columns produced for every cohort:
    cohort, sample, age, sex, grade, grade_group, gbm, histology,
    idh_status, codel_1p19q, mgmt_status, os_months, os_event

Variables that do not exist in a cohort are emitted as missing (NaN) and the
missingness is recorded, rather than imputed across cohorts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger()

GLIOVIS_FILES = {"tcga": "TCGA_GBMLGG.Rds",
                 "rembrandt": "Rembrandt.Rds",
                 "gravendeel": "Gravendeel.Rds"}

# adult diffuse glioma histologies to keep; everything else dropped
KEEP_HISTOLOGY = {
    "GBM": "GBM",
    "Astrocytoma": "Astrocytoma",
    "Oligodendroglioma": "Oligodendroglioma",
    "Oligoastrocytoma": "Mixed/OA",
    "Mixed glioma": "Mixed/OA",
}
DROP_HISTOLOGY = {"Non-tumor", "Pilocytic Astrocytoma", "Unknown", "Unclassified", None, np.nan}

GRADE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}


def _load_rds(name: str):
    import rdata
    path = config.RAW / GLIOVIS_FILES[name]
    obj = rdata.read_rds(str(path))
    return obj["pData"].copy(), obj["expr"].copy()


def _norm_idh(series: pd.Series) -> pd.Series:
    def f(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if "mut" in s or s in ("yes", "1", "r132h"):
            return "Mutant"
        if "wt" in s or "wild" in s or s in ("no", "0", "normal"):
            return "WT"
        return np.nan
    return series.map(f)


def _norm_codel(df: pd.DataFrame) -> pd.Series:
    if "Chr.1p_19q.codeletion" in df.columns:
        return df["Chr.1p_19q.codeletion"].map(
            lambda v: "Codeleted" if str(v).strip().lower() in ("codel", "yes", "codeleted", "1")
            else ("Non-codeleted" if not pd.isna(v) else np.nan))
    if "IDH_codel.subtype" in df.columns:
        return df["IDH_codel.subtype"].map(
            lambda v: "Codeleted" if isinstance(v, str) and "codel" in v.lower()
            else ("Non-codeleted" if isinstance(v, str) else np.nan))
    if {"LOH_1p", "LOH_19q"}.issubset(df.columns):
        def comb(r):
            a, b = str(r["LOH_1p"]).lower(), str(r["LOH_19q"]).lower()
            if a in ("nan", "none") or b in ("nan", "none"):
                return np.nan
            loss = lambda x: x in ("loss", "yes", "1", "true")
            return "Codeleted" if (loss(a) and loss(b)) else "Non-codeleted"
        return df.apply(comb, axis=1)
    return pd.Series(np.nan, index=df.index)


def _norm_mgmt(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in ("MGMT.promoter.status", "MGMT_status", "MGMT") if c in df.columns), None)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return df[col].map(lambda v: "Methylated" if str(v).strip().lower().startswith("methyl")
                       else ("Unmethylated" if str(v).strip().lower().startswith("unmethyl") else np.nan))


def harmonise_cohort(name: str):
    pdata, expr = _load_rds(name)
    n0 = len(pdata)
    out = pd.DataFrame(index=pdata.index)
    out["cohort"] = name
    out["sample"] = pdata["Sample"].values

    # demographics
    out["age"] = pd.to_numeric(pdata["Age"], errors="coerce") if "Age" in pdata.columns else np.nan
    if "Gender" in pdata.columns:
        out["sex"] = pdata["Gender"].map(lambda v: str(v).strip().capitalize() if not pd.isna(v) else np.nan)
    else:
        out["sex"] = np.nan

    # grade / histology
    grade = pdata["Grade"].map(lambda v: GRADE_MAP.get(str(v).strip(), np.nan))
    out["grade"] = grade
    out["grade_group"] = grade.map(lambda g: "high" if g in (3, 4) else ("lower" if g in (1, 2) else np.nan))
    out["gbm"] = grade.map(lambda g: 1 if g == 4 else (0 if g in (1, 2, 3) else np.nan))
    out["histology"] = pdata["Histology"].map(lambda v: KEEP_HISTOLOGY.get(v, np.nan))

    # molecular
    idh_col = next((c for c in ("IDH.status", "IDH1_status", "IDH_status") if c in pdata.columns), None)
    out["idh_status"] = _norm_idh(pdata[idh_col]) if idh_col else np.nan
    out["codel_1p19q"] = _norm_codel(pdata)
    out["mgmt_status"] = _norm_mgmt(pdata)

    # outcome: GlioVis 'survival' = months, 'status' = 1 death / 0 alive
    out["os_months"] = pd.to_numeric(pdata["survival"], errors="coerce")
    out["os_event"] = pd.to_numeric(pdata["status"], errors="coerce")

    # exclusions: keep adult diffuse glioma with usable outcome
    keep_hist = pdata["Histology"].isin(KEEP_HISTOLOGY.keys())
    usable = out["os_months"].notna() & out["os_event"].notna()
    mask = keep_hist & usable
    clinical = out.loc[mask].reset_index(drop=True)

    # expression: align to retained samples, keep Sample + prespecified panel genes
    meta_cols = {"Sample", "Histology", "Grade", "Recurrence", "Subtype", "CIMP_status",
                 "survival", "status"}
    gene_cols = [c for c in expr.columns if c not in meta_cols]
    panel_present = [g for g in config.GENE_PANEL if g in gene_cols]
    expr2 = expr[["Sample"] + panel_present].copy()
    expr2 = expr2[expr2["Sample"].isin(clinical["sample"])].reset_index(drop=True)

    info = {
        "cohort": name, "n_raw": int(n0), "n_kept": int(len(clinical)),
        "n_dropped_histology": int((~keep_hist).sum()),
        "n_dropped_no_outcome": int((keep_hist & ~usable).sum()),
        "n_genes_available": len(gene_cols),
        "n_panel_genes": len(panel_present),
    }
    log.info("Harmonised %s: kept %d/%d (genes=%d)", name, len(clinical), n0, len(gene_cols))
    return clinical, expr2, info


def harmonise_all(save: bool = True):
    clin_frames, infos = [], []
    for name in GLIOVIS_FILES:
        clinical, expr, info = harmonise_cohort(name)
        infos.append(info)
        clin_frames.append(clinical)
        if save:
            clinical.to_csv(config.PROCESSED / f"clinical_{name}.csv", index=False)
            expr.to_csv(config.PROCESSED / f"expr_panel_{name}.csv", index=False)
    combined = pd.concat(clin_frames, ignore_index=True)
    if save:
        combined.to_csv(config.PROCESSED / "clinical_all.csv", index=False)
        pd.DataFrame(infos).to_csv(config.PROCESSED / "harmonisation_summary.csv", index=False)
    return combined, pd.DataFrame(infos)


if __name__ == "__main__":
    _, summary = harmonise_all()
    print(summary.to_string(index=False))
