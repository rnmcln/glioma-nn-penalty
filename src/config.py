"""Central configuration: paths, seeds, prespecified analysis choices.

All analytical decisions that must be fixed before seeing results are declared
here so they are auditable and version-controlled (see PROTOCOL.md).
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DICTIONARY = DATA / "dictionary"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
MODEL_OBJECTS = RESULTS / "model_objects"
LOGS = RESULTS / "logs"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

for _d in (RAW, PROCESSED, DICTIONARY, METRICS, PREDICTIONS, MODEL_OBJECTS, LOGS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 20260628
# Defaults are the rigorous committed settings; both can be reduced for a quick
# run via environment variables (GNN_N_BOOTSTRAP, GNN_CV_REPEATS) without editing
# code. The manuscript reports results under the committed defaults.
N_BOOTSTRAP = int(os.environ.get("GNN_N_BOOTSTRAP", 2000))
CV_REPEATS = int(os.environ.get("GNN_CV_REPEATS", 10))
CV_FOLDS = int(os.environ.get("GNN_CV_FOLDS", 5))

# ---------------------------------------------------------------------------
# Endpoints (months). Primary = 2-year OS, censored-before-endpoint excluded.
# ---------------------------------------------------------------------------
ENDPOINTS = {
    "os_1y": 12,
    "os_2y": 24,   # PRIMARY
    "os_5y": 60,
}
PRIMARY_ENDPOINT = "os_2y"

# ---------------------------------------------------------------------------
# Predictor sets (concept names; harmonised columns produced in data_harmonise)
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "A_clinical": ["age", "sex", "grade", "histology"],
    "B_clinical_molecular": [
        "age", "sex", "grade", "histology",
        "idh_status", "codel_1p19q", "mgmt_status",
    ],
    # C is expression-derived; populated at runtime if expression is harmonised.
    "C_expression": [],
}
PRIMARY_FEATURE_SET = "B_clinical_molecular"

# Missingness rule: drop variables with > threshold missing unless essential.
MISSINGNESS_DROP_THRESHOLD = 0.40
ESSENTIAL_VARS = {"age", "grade"}

# Clinically meaningful improvement (prespecified decision rule).
MEANINGFUL_AUROC_DELTA = 0.03   # NN must beat best transparent model by >= this
# ...and must not worsen Brier score or calibration slope materially.

# Decision-curve thresholds (probability of death by endpoint).
DCA_THRESHOLDS = [round(x, 2) for x in [i / 100 for i in range(1, 100)]]
REPORT_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]

# Prespecified categorical levels (structural metadata, not outcome-derived).
# Fixing these makes the one-hot schema identical across folds and cohorts, which
# avoids fold-to-fold dimension drift. It uses no outcome information.
CATEGORICAL_LEVELS = {
    "sex": ["Female", "Male"],
    "histology": ["GBM", "Astrocytoma", "Oligodendroglioma", "Mixed/OA"],
    "idh_status": ["Mutant", "WT"],
    "codel_1p19q": ["Codeleted", "Non-codeleted"],
    "mgmt_status": ["Methylated", "Unmethylated"],
}

# Prespecified expression panel for feature set C (EXPLORATORY).
# Provenance: this is an a priori list of established glioma-relevant genes
# (e.g., EGFR, PDGFRA, PTEN, NF1, TP53, RB1, CDKN2A/B, IDH1/2, ATRX, CIC, MGMT,
# TERT, OLIG2, GFAP) drawn from the WHO CNS classification and recurrent glioma
# drivers, plus a small set of microenvironment/proliferation markers. Genes were
# NOT selected using survival outcomes. The analysis uses only panel genes
# present in ALL cohorts; expression is z-scored within each cohort. Because the
# rationale is biological/availability-based rather than data-driven, feature set
# C is reported as an exploratory analysis, not a primary claim. Nested
# outcome-guided selection (set D) is confined to within-fold training data only.
GENE_PANEL = [
    "EGFR", "PDGFRA", "PTEN", "NF1", "TP53", "RB1", "CDKN2A", "CDKN2B",
    "MDM2", "MDM4", "CDK4", "CDK6", "MET", "PIK3CA", "PIK3R1", "PIK3CB",
    "IDH1", "IDH2", "ATRX", "CIC", "FUBP1", "NOTCH1", "TERT", "MGMT",
    "OLIG2", "GFAP", "SOX2", "NES", "CD44", "CHI3L1", "VEGFA", "MKI67",
    "BCAN", "DLL3", "POSTN", "SERPINE1", "TIMP1", "LGALS1", "ANXA1", "CASP8",
]

# Data sources
CBIOPORTAL_STUDY = "lgggbm_tcga_pub"   # Merged LGG+GBM (TCGA, Cell 2016)
CBIOPORTAL_API = "https://www.cbioportal.org/api"
GEO_REMBRANDT = "GSE108476"            # REMBRANDT
GEO_GRAVENDEEL = "GSE16011"            # Gravendeel
