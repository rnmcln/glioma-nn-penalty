"""Reproducible public-data download.

TCGA  : cBioPortal REST API, study 'lgggbm_tcga_pub' (Merged LGG+GBM, Cell 2016).
          Provides clinical + molecular annotation (IDH, 1p/19q, MGMT, grade,
          histology, age, sex, OS).
REMBRANDT : GEO GSE108476 series-matrix (clinical/survival; sparse molecular).
Gravendeel: GEO GSE16011 series-matrix (clinical/survival; sparse molecular).

Everything is fetched from public endpoints and cached under data/raw.
No authentication required.
"""
from __future__ import annotations
import gzip
import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import config
from .utils import get_logger

log = get_logger()
TIMEOUT = 60


# ---------------------------------------------------------------------------
# cBioPortal (TCGA)
# ---------------------------------------------------------------------------
def _cbio_get(path: str, params: dict | None = None):
    url = f"{config.CBIOPORTAL_API}/{path.lstrip('/')}"
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # pragma: no cover - network resilience
            log.warning("cBioPortal retry %d (%s): %s", attempt + 1, path, e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"cBioPortal request failed: {url}")


def download_tcga(force: bool = False) -> pd.DataFrame:
    """Download TCGA LGG+GBM patient+sample clinical data, return wide table."""
    out = config.RAW / "tcga_lgggbm_clinical_raw.csv"
    if out.exists() and not force:
        log.info("TCGA raw cache hit: %s", out)
        return pd.read_csv(out)

    study = config.CBIOPORTAL_STUDY
    log.info("Downloading TCGA clinical data (%s) from cBioPortal", study)
    frames = []
    for dtype in ("PATIENT", "SAMPLE"):
        data = _cbio_get(
            f"studies/{study}/clinical-data",
            params={"clinicalDataType": dtype, "projection": "DETAILED", "pageSize": 100000},
        )
        if not data:
            continue
        df = pd.DataFrame(data)
        # rows: patientId/sampleId + clinicalAttributeId + value -> pivot wide
        idcol = "patientId"
        wide = (
            df.pivot_table(index=idcol, columns="clinicalAttributeId", values="value", aggfunc="first")
            .reset_index()
        )
        wide.columns.name = None
        frames.append(wide)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="patientId", how="outer", suffixes=("", "_sample"))
    merged.to_csv(out, index=False)
    log.info("TCGA raw saved: %s rows=%d cols=%d", out, len(merged), merged.shape[1])
    return merged


# ---------------------------------------------------------------------------
# GEO series-matrix parsing
# ---------------------------------------------------------------------------
def _geo_matrix_urls(gse: str) -> list[str]:
    stub = re.sub(r"\d{1,3}$", "nnn", gse)  # GSE16011 -> GSE16nnn
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/matrix/"
    try:
        r = requests.get(base, timeout=TIMEOUT)
        r.raise_for_status()
        files = re.findall(r'href="([^"]*series_matrix\.txt\.gz)"', r.text)
        return [base + f for f in sorted(set(files))]
    except Exception as e:
        log.warning("GEO matrix listing failed for %s: %s", gse, e)
        return [base + f"{gse}_series_matrix.txt.gz"]


def _parse_series_matrix(text: str) -> pd.DataFrame:
    """Extract sample-level metadata (!Sample_* header lines) into a tidy frame."""
    rows = {}
    char_counter = 0
    for line in text.splitlines():
        if not line.startswith("!Sample_"):
            continue
        parts = line.split("\t")
        key = parts[0].lstrip("!")
        vals = [p.strip().strip('"') for p in parts[1:]]
        if key == "Sample_characteristics_ch1":
            key = f"{key}_{char_counter}"
            char_counter += 1
        rows[key] = vals
    if not rows:
        return pd.DataFrame()
    n = max(len(v) for v in rows.values())
    rows = {k: (v + [None] * (n - len(v))) for k, v in rows.items()}
    return pd.DataFrame(rows)


def download_geo(gse: str, force: bool = False) -> pd.DataFrame:
    out = config.RAW / f"{gse}_clinical_raw.csv"
    if out.exists() and not force:
        log.info("GEO raw cache hit: %s", out)
        return pd.read_csv(out)
    log.info("Downloading GEO %s series matrix", gse)
    frames = []
    for url in _geo_matrix_urls(gse):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            text = gzip.decompress(r.content).decode("utf-8", errors="replace")
            df = _parse_series_matrix(text)
            if not df.empty:
                frames.append(df)
        except Exception as e:  # pragma: no cover
            log.warning("GEO download/parse failed (%s): %s", url, e)
    if not frames:
        raise RuntimeError(f"No usable series matrix for {gse}")
    clinical = pd.concat(frames, ignore_index=True)
    clinical.to_csv(out, index=False)
    log.info("GEO %s raw saved: rows=%d cols=%d", gse, len(clinical), clinical.shape[1])
    return clinical


# ---------------------------------------------------------------------------
# GlioVis harmonised data objects (.Rds) -- primary harmonised source
# ---------------------------------------------------------------------------
# Pinned to a specific commit of the shiny_GlioVis repository for archival
# reproducibility (see data/dictionary/data_manifest.json for SHA-256 checksums).
GLIOVIS_COMMIT = "eae2ce7852ba"
GLIOVIS_BASE = f"https://raw.githubusercontent.com/msquatrito/shiny_GlioVis/{GLIOVIS_COMMIT}/data/datasets"
GLIOVIS_FILES = {"TCGA_GBMLGG.Rds", "Rembrandt.Rds", "Gravendeel.Rds"}


def download_gliovis(force: bool = False) -> list[Path]:
    paths = []
    for fname in GLIOVIS_FILES:
        out = config.RAW / fname
        if out.exists() and out.stat().st_size > 0 and not force:
            log.info("GlioVis cache hit: %s", out)
            paths.append(out)
            continue
        url = f"{GLIOVIS_BASE}/{fname}"
        log.info("Downloading GlioVis object %s", fname)
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        out.write_bytes(r.content)
        log.info("Saved %s (%.1f MB)", out, len(r.content) / 1e6)
        paths.append(out)
    return paths


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    download_gliovis(force=force)
    return {
        "tcga": download_tcga(force=force),
        "rembrandt": download_geo(config.GEO_REMBRANDT, force=force),
        "gravendeel": download_geo(config.GEO_GRAVENDEEL, force=force),
    }


if __name__ == "__main__":
    download_all()
