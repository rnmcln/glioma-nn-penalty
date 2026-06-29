"""Publication figures (matplotlib only), revised for editorial hierarchy.

Design choices follow the figure review: a clean monochrome study-design diagram;
external validation shown in the main performance figure; a dedicated figure for
the collapse of discrimination within molecular strata; calibration and
decision-curve analysis in the main external cohort with predicted-risk
distribution and a clinically plausible threshold band. All-model versions and
internal-only panels are supplementary. Low-ink Tufte-style styling, colour-blind
safe, colour redundant with position/grouping.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .calibration import calibration_curve_points
from .decision_curve import net_benefit

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
})

# colour-blind-safe, redundant with grouping
C_TRANS = "#0072B2"   # transparent models (blue)
C_ML = "#888888"      # standard ML (grey)
C_NN = "#D55E00"      # neural networks (vermilion)
LABELS = {
    "base_rate": "Base rate", "logistic_age_grade": "Age + grade LR",
    "penalised_logistic": "Penalised LR", "random_forest": "Random forest",
    "hist_gboost": "Gradient boosting", "tf_compact": "Compact MLP",
    "tf_medium": "Medium MLP", "tf_regularised": "Regularised MLP",
    "ft_transformer": "FT-Transformer",
}
ORDER = ["base_rate", "logistic_age_grade", "penalised_logistic", "random_forest",
         "hist_gboost", "tf_compact", "tf_medium", "tf_regularised", "ft_transformer"]
TRANSPARENT = {"base_rate", "logistic_age_grade", "penalised_logistic"}
STANDARD_ML = {"random_forest", "hist_gboost"}
TF = {"tf_compact", "tf_medium", "tf_regularised", "ft_transformer"}


def _colour(m):
    return C_TRANS if m in TRANSPARENT else (C_ML if m in STANDARD_ML else C_NN)


def _marker(m):
    return "o" if m in TRANSPARENT else ("s" if m in STANDARD_ML else "^")


def _linestyle(m):
    return "-" if m in TRANSPARENT else ("--" if m in STANDARD_ML else ":")


def _save(fig, stem, tiff=False):
    fig.savefig(config.FIGURES / f"{stem}.png", dpi=300)
    fig.savefig(config.FIGURES / f"{stem}.pdf")
    if tiff:
        try:
            fig.savefig(config.FIGURES / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        except Exception:
            pass
    plt.close(fig)


def _ordered(models):
    return [m for m in ORDER if m in models]


# ---------------------------------------------------------------------------
# Figure 1 — study design (monochrome, clean left-to-right flow)
# ---------------------------------------------------------------------------
def figure_study_design(stem="figure1_study_design"):
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.axis("off"); ax.grid(False)
    grey = "#444444"

    def box(x, y, w, h, title, lines):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=grey, lw=1.0))
        ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.215, "\n".join(lines), ha="center", va="top", fontsize=7.8, color="#222")

    box(0.005, 0.52, 0.225, 0.40, "Data sources",
        ["Development:", "TCGA LGG/GBM (n=667)", "", "External validation:", "Gravendeel (n=264)", "REMBRANDT (n=342)"])
    box(0.265, 0.34, 0.18, 0.34, "Endpoint",
        ["2-year overall", "survival", "", "1- and 5-year", "sensitivity"])
    box(0.475, 0.34, 0.21, 0.34, "Models",
        ["Base rate; logistic", "regression; penalised", "logistic; random forest;", "gradient boosting;", "multilayer perceptrons;", "FT-Transformer"])
    box(0.715, 0.34, 0.235, 0.34, "Evaluation",
        ["AUROC, AUPRC,", "Brier score,", "calibration,", "decision curves,", "paired bootstrap"])

    def arrow(x1, x2, y1=0.52, y2=0.51):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color=grey))
    arrow(0.23, 0.265, 0.62, 0.51); arrow(0.445, 0.475); arrow(0.685, 0.715)
    ax.set_xlim(0, 1); ax.set_ylim(0.25, 1.0)
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Figure 2 — AUROC forest across validation settings (internal + external)
# ---------------------------------------------------------------------------
def figure_forest_cohorts(rows, stem="figure2_performance_forest"):
    """rows: list of dicts {cohort, model, auroc, lo, hi}. Grouped by cohort."""
    cohorts = ["TCGA (internal)", "Gravendeel (external)", "REMBRANDT (external)"]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.6), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, coh in zip(axes, cohorts):
        for yi, m in zip(y, models):
            r = next((x for x in rows if x["cohort"] == coh and x["model"] == m), None)
            if not r:
                continue
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=_colour(m), lw=1.4, alpha=0.8)
            ax.plot(r["auroc"], yi, _marker(m), color=_colour(m), ms=5.5)
        ax.axvline(0.5, color="#bbbbbb", lw=0.8, ls="--")
        ax.set_title(coh, fontsize=9.5)
        ax.set_xlim(0.45, 0.98); ax.set_xlabel("AUROC (higher is better)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABELS[m] for m in models])
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker=mk, color="w", mfc=c, mec=c, ms=8, label=l)
         for c, l, mk in [(C_TRANS, "Transparent", "o"), (C_ML, "Standard ML", "s"), (C_NN, "Neural network", "^")]]
    axes[2].legend(handles=h, fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("Discrimination across internal and external validation", fontsize=10.5)
    fig.tight_layout()
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Figure 3 — discrimination collapses within molecular strata
# ---------------------------------------------------------------------------
def figure_strata(rows, stem="figure3_strata"):
    """rows: {stratum, model, auroc, lo, hi}. 2x2 grid for readable labels."""
    strata = ["Pooled (LGG+GBM)", "IDH-wildtype", "IDH-mutant", "Glioblastoma"]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    y = np.arange(len(models))[::-1]
    for ax, st in zip(axes.ravel(), strata):
        for yi, m in zip(y, models):
            r = next((x for x in rows if x["stratum"] == st and x["model"] == m), None)
            if not r:
                continue
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=_colour(m), lw=1.6, alpha=0.85)
            ax.plot(r["auroc"], yi, _marker(m), color=_colour(m), ms=6)
        ax.axvline(0.5, color="#bbbbbb", lw=0.8, ls="--")
        ax.set_title(st, fontsize=11, fontweight="bold")
        ax.set_xlim(0.45, 0.98)
        ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in models], fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("AUROC (higher is better)")
    fig.suptitle("Discrimination within molecular strata: pooled AUROC largely reflects molecular class",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Figure 4 — calibration + decision curve in the main external cohort
# ---------------------------------------------------------------------------
def _best(preds, pa, group):
    cand = [m for m in preds if m in group]
    return max(cand, key=lambda m: pa[m]) if cand else None


def figure_calib_dca(y_true, preds, pa, cohort_label, stem="figure4_calibration_dca"):
    show = [m for m in ["penalised_logistic", _best(preds, pa, STANDARD_ML), _best(preds, pa, TF)] if m]
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(10, 4.6))
    # calibration with predicted-risk rug
    axc.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    for m in show:
        mp, of, cnt = calibration_curve_points(y_true, preds[m], n_bins=10)
        sizes = 20 + 120 * (cnt / max(cnt.max(), 1))
        axc.plot(mp, of, _linestyle(m), lw=1.4, color=_colour(m), label=LABELS[m])
        axc.scatter(mp, of, s=sizes, marker=_marker(m), color=_colour(m), alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
    # rug of predicted risks (first shown model)
    p0 = preds[show[0]]
    axc.plot(np.clip(p0, 0, 1), np.full_like(p0, -0.02), "|", color="#888", ms=4, alpha=0.4)
    axc.set_xlim(0, 1); axc.set_ylim(-0.04, 1)
    axc.set_xlabel("Mean predicted probability of 2-year mortality")
    axc.set_ylabel("Observed frequency")
    axc.set_title(f"Calibration — {cohort_label}", fontsize=9.5)
    axc.legend(fontsize=7.5, loc="upper left", frameon=False)
    # decision curve with plausible band
    thr = np.array([t / 100 for t in range(5, 81)])
    base = net_benefit(y_true, preds[show[0]], thr)
    axd.axvspan(0.10, 0.50, color="#f3f3f3", zorder=0)
    axd.plot(thr, base["net_benefit_all"], color="#999", lw=1, ls=":", label="Treat all")
    axd.plot(thr, np.zeros_like(thr), color="black", lw=1, ls=":", label="Treat none")
    for m in show:
        nb = net_benefit(y_true, preds[m], thr)
        axd.plot(thr, nb["net_benefit_model"], _linestyle(m), lw=1.6, color=_colour(m), label=LABELS[m])
    axd.set_xlim(0.05, 0.80); axd.set_ylim(-0.05, max(0.45, float(np.nanmax(base["net_benefit_all"])) + 0.05))
    axd.set_xlabel("Threshold probability")
    axd.set_ylabel("Net benefit")
    axd.set_title(f"Decision-curve analysis — {cohort_label}", fontsize=9.5)
    axd.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Supplementary: internal stacked performance; all-model calib/DCA; learning curves
# ---------------------------------------------------------------------------
def figure_internal_stacked(perf, stem="figureS_internal_performance"):
    models = _ordered(perf.keys())
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharey=True)
    for ax, metric, title in zip(axes, ("auroc", "brier"),
                                 ("Discrimination (AUROC, higher is better)", "Accuracy (Brier score, lower is better)")):
        for yi, m in zip(y, models):
            d = perf[m][metric]
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=_colour(m), lw=1.3, alpha=0.8)
            ax.plot(d["point"], yi, _marker(m), color=_colour(m), ms=5.5)
        ax.set_xlabel(title)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABELS[m] for m in models])
    fig.suptitle("Internal validation performance (TCGA, set B, 2-year)", fontsize=10)
    fig.tight_layout()
    _save(fig, stem)


def figure_all_calibration(y_true, preds, stem="figureS_calibration_all"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    a1.plot([0, 1], [0, 1], "k--", lw=1); a2.plot([0, 1], [0, 1], "k--", lw=1)
    for m in _ordered(preds.keys()):
        mp, of, _ = calibration_curve_points(y_true, preds[m], 10)
        ax = a1 if m in (TRANSPARENT | STANDARD_ML) else a2
        ax.plot(mp, of, _linestyle(m), marker=_marker(m), ms=3.5, lw=1, color=_colour(m), label=LABELS[m])
    a1.set_title("Transparent and standard ML"); a2.set_title("Neural networks")
    for ax in (a1, a2):
        ax.set_xlabel("Mean predicted probability"); ax.legend(fontsize=7, frameon=False)
    a1.set_ylabel("Observed frequency")
    fig.suptitle("Calibration, all models (TCGA internal, 2-year)", fontsize=10)
    fig.tight_layout(); _save(fig, stem)


def figure_all_dca(y_true, preds, stem="figureS_decision_all"):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    thr = np.array([t / 100 for t in range(5, 81)])
    base = net_benefit(y_true, preds["penalised_logistic"], thr)
    ax.plot(thr, base["net_benefit_all"], color="#999", lw=1, ls=":", label="Treat all")
    ax.plot(thr, np.zeros_like(thr), "k:", lw=1, label="Treat none")
    for m in _ordered(preds.keys()):
        if m == "base_rate":
            continue
        nb = net_benefit(y_true, preds[m], thr)
        ax.plot(thr, nb["net_benefit_model"], _linestyle(m), lw=1.1, color=_colour(m), label=LABELS[m])
    ax.set_xlim(0.05, 0.80); ax.set_ylim(-0.05, 0.45)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis, all models (TCGA internal, 2-year)", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    _save(fig, stem)


def figure_learning_curves(histories, stem="figureS_learning_curves"):
    """histories: dict variant -> DataFrame with loss, val_loss."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    cols = {"compact": "#D55E00", "medium": "#E69F00", "regularised": "#CC79A7"}
    for name, h in histories.items():
        ax.plot(h["loss"], lw=1.2, color=cols.get(name, "#333"), label=f"{name} (train)")
        if "val_loss" in h:
            ax.plot(h["val_loss"], lw=1.2, ls="--", color=cols.get(name, "#333"), label=f"{name} (val)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("Neural-network training and validation loss (TCGA, set B)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, stem)
