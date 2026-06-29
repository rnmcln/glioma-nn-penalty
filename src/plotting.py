"""Publication figures (matplotlib only) with a single, consistent visual identity.

One palette and marker system is used across every main and supplementary figure:
transparent models in muted blue (circles), standard machine learning in neutral
grey (squares), neural networks in burnt orange (triangles); reference, ideal and
treat-all/treat-none lines are thin and visually secondary. Sentence-case, concise
panel titles; captions carry interpretation. Figures export to PNG, vector PDF and
SVG at final manuscript size; the four main figures also export 600-dpi TIFF.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

from . import config
from .calibration import calibration_curve_points
from .decision_curve import net_benefit

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#666666",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
    "savefig.bbox": "tight", "figure.dpi": 110,
})

# Single restrained, colour-blind-aware palette (redundant with marker shape).
C_TRANS = "#3B6FA0"   # transparent models  — muted blue
C_ML = "#7F7F7F"      # standard ML         — neutral grey
C_NN = "#C26A27"      # neural networks     — burnt orange
C_REF = "#9AA0A6"     # reference / treat-all — light grey
C_TEXT = "#2A2A2A"    # charcoal text
C_ACCENT = "#3B6FA0"  # single accent (Fig 1)

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

X_AUROC = (0.45, 0.98)   # shared AUROC axis limits across forest figures


def _colour(m):
    return C_TRANS if m in TRANSPARENT else (C_ML if m in STANDARD_ML else C_NN)


def _marker(m):
    return "o" if m in TRANSPARENT else ("s" if m in STANDARD_ML else "^")


def _linestyle(m):
    return "-" if m in TRANSPARENT else ("--" if m in STANDARD_ML else (0, (1, 1)))


def _ordered(models):
    return [m for m in ORDER if m in models]


def _class_legend(extra=None):
    h = [Line2D([0], [0], marker=mk, color="none", mfc=c, mec=c, ms=7, label=l)
         for c, l, mk in [(C_TRANS, "Transparent models", "o"),
                          (C_ML, "Standard ML", "s"),
                          (C_NN, "Neural networks", "^")]]
    if extra:
        h += extra
    return h


def _save(fig, stem, tiff=False):
    fig.savefig(config.FIGURES / f"{stem}.png", dpi=400)
    fig.savefig(config.FIGURES / f"{stem}.pdf")
    fig.savefig(config.FIGURES / f"{stem}.svg")
    if tiff:
        try:
            fig.savefig(config.FIGURES / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        except Exception:
            pass
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — AUROC forest across internal and external validation
# ---------------------------------------------------------------------------
def figure_forest_cohorts(rows, stem="figure1_performance_forest"):
    cohorts = ["TCGA (internal)", "Gravendeel (external)", "REMBRANDT (external)"]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, coh in zip(axes, cohorts):
        ax.axvline(0.5, color=C_REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for yi, m in zip(y, models):
            r = next((x for x in rows if x["cohort"] == coh and x["model"] == m), None)
            if not r:
                continue
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=_colour(m), lw=1.4, alpha=0.85, zorder=2)
            ax.plot(r["auroc"], yi, _marker(m), color=_colour(m), ms=5.5,
                    mec="white", mew=0.5, zorder=3)
        ax.set_title(coh, fontsize=10)
        ax.set_xlim(*X_AUROC)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.tick_params(length=2.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABELS[m] for m in models])
    fig.supxlabel("AUROC (higher is better)", fontsize=9.5, y=0.05)
    fig.legend(handles=_class_legend(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Figure 2 — discrimination within molecular strata (2x2, panel labels A-D)
# ---------------------------------------------------------------------------
def figure_strata(rows, stem="figure2_strata"):
    strata = ["Pooled cohort", "IDH-wildtype", "IDH-mutant", "Glioblastoma"]
    alias = {"Pooled cohort": "Pooled (LGG+GBM)"}
    letters = ["A", "B", "C", "D"]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, st, lab in zip(axes.ravel(), strata, letters):
        key = alias.get(st, st)
        ax.axvline(0.5, color=C_REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for yi, m in zip(y, models):
            r = next((x for x in rows if x["stratum"] in (st, key) and x["model"] == m), None)
            if not r:
                continue
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=_colour(m), lw=1.5, alpha=0.85, zorder=2)
            ax.plot(r["auroc"], yi, _marker(m), color=_colour(m), ms=5.8,
                    mec="white", mew=0.5, zorder=3)
        ax.set_title(f"{lab}  {st}", fontsize=10, loc="left")
        ax.set_xlim(*X_AUROC)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in models])
        ax.tick_params(length=2.5)
    for ax in axes[1, :]:
        ax.set_xlabel("AUROC (higher is better)")
    fig.legend(handles=_class_legend(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Discrimination within molecular strata", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Figure 3 — calibration + decision curve in the main external cohort
# ---------------------------------------------------------------------------
def figure_calib_dca(y_true, preds, pa, cohort_label, stem="figure3_calibration_dca"):
    show = [m for m in ["penalised_logistic", "random_forest", "tf_compact"] if m in preds]
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    # --- calibration ---
    axc.plot([0, 1], [0, 1], color=C_REF, lw=1.0, ls=(0, (4, 3)), label="Ideal", zorder=1)
    for m in show:
        mp, of, _ = calibration_curve_points(y_true, preds[m], n_bins=10)
        axc.plot(mp, of, color=_colour(m), lw=1.4, ls=_linestyle(m), zorder=2)
        axc.plot(mp, of, _marker(m), color=_colour(m), ms=4.2, mec="white", mew=0.4,
                 label=LABELS[m], zorder=3)
    p0 = np.clip(preds[show[0]], 0, 1)
    axc.plot(p0, np.full_like(p0, -0.025), "|", color="#bbbbbb", ms=3, alpha=0.35, zorder=1)
    axc.set_xlim(0, 1); axc.set_ylim(-0.045, 1.0)
    axc.set_xlabel("Mean predicted probability")
    axc.set_ylabel("Observed frequency")
    axc.set_title("Calibration", fontsize=10)
    axc.legend(loc="upper left", frameon=False, fontsize=8)

    # --- decision curve ---
    thr = np.array([t / 100 for t in range(5, 81)])
    base = net_benefit(y_true, preds[show[0]], thr)
    axd.axvspan(0.10, 0.50, color="#f1f1f1", zorder=0)
    axd.plot(thr, base["net_benefit_all"], color=C_REF, lw=0.9, ls=(0, (4, 3)), label="Treat all", zorder=1)
    axd.plot(thr, np.zeros_like(thr), color="#444444", lw=0.9, ls=(0, (1, 2)), label="Treat none", zorder=1)
    for m in show:
        nb = net_benefit(y_true, preds[m], thr)
        axd.plot(thr, nb["net_benefit_model"], color=_colour(m), lw=1.5, ls=_linestyle(m),
                 label=LABELS[m], zorder=2)
    axd.set_xlim(0.05, 0.80)
    axd.set_ylim(-0.05, max(0.45, float(np.nanmax(base["net_benefit_all"])) + 0.05))
    axd.set_xlabel("Threshold probability")
    axd.set_ylabel("Net benefit")
    axd.set_title("Decision-curve analysis", fontsize=10)
    axd.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, stem, tiff=True)


# ---------------------------------------------------------------------------
# Supplementary — internal stacked performance (AUROC, Brier)
# ---------------------------------------------------------------------------
def figure_internal_stacked(perf, stem="figureS_internal_performance"):
    models = _ordered(perf.keys())
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.4), sharey=True)
    for ax, metric, title in zip(axes, ("auroc", "brier"),
                                 ("Discrimination (AUROC, higher is better)",
                                  "Brier score (lower is better)")):
        for yi, m in zip(y, models):
            d = perf[m][metric]
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=_colour(m), lw=1.3, alpha=0.85, zorder=2)
            ax.plot(d["point"], yi, _marker(m), color=_colour(m), ms=5.2,
                    mec="white", mew=0.5, zorder=3)
        ax.set_xlabel(title)
        ax.tick_params(length=2.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABELS[m] for m in models])
    fig.legend(handles=_class_legend(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Supplementary — all-model calibration (two panels)
# ---------------------------------------------------------------------------
def figure_all_calibration(y_true, preds, stem="figureS_calibration_all"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True, sharey=True)
    for ax in (a1, a2):
        ax.plot([0, 1], [0, 1], color=C_REF, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for m in _ordered(preds.keys()):
        if m == "base_rate":   # constant prediction has no informative calibration curve
            continue
        mp, of, _ = calibration_curve_points(y_true, preds[m], 10)
        ax = a1 if m in (TRANSPARENT | STANDARD_ML) else a2
        ax.plot(mp, of, color=_colour(m), lw=1.1, ls=_linestyle(m), zorder=2)
        ax.plot(mp, of, _marker(m), color=_colour(m), ms=3.4, mec="white", mew=0.3,
                label=LABELS[m], zorder=3)
    a1.set_title("Transparent and standard ML", fontsize=10)
    a2.set_title("Neural networks", fontsize=10)
    for ax in (a1, a2):
        ax.set_xlabel("Mean predicted probability")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax.set_xlim(0, 1)
    a1.set_ylabel("Observed frequency")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Supplementary — all-model decision curves (two panels)
# ---------------------------------------------------------------------------
def figure_all_dca(y_true, preds, stem="figureS_decision_all"):
    thr = np.array([t / 100 for t in range(5, 81)])
    base = net_benefit(y_true, preds["penalised_logistic"], thr)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True, sharey=True)
    groups = [("Transparent and standard ML", TRANSPARENT | STANDARD_ML),
              ("Neural networks", TF)]
    for ax, (title, grp) in zip((a1, a2), groups):
        ax.plot(thr, base["net_benefit_all"], color=C_REF, lw=0.9, ls=(0, (4, 3)), label="Treat all", zorder=1)
        ax.plot(thr, np.zeros_like(thr), color="#444444", lw=0.9, ls=(0, (1, 2)), label="Treat none", zorder=1)
        for m in _ordered(preds.keys()):
            if m == "base_rate" or m not in grp:
                continue
            nb = net_benefit(y_true, preds[m], thr)
            ax.plot(thr, nb["net_benefit_model"], color=_colour(m), lw=1.3, ls=_linestyle(m),
                    label=LABELS[m], zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.05, 0.80); ax.set_ylim(-0.05, 0.45)
        ax.set_xlabel("Threshold probability")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    a1.set_ylabel("Net benefit")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Supplementary — neural-network learning curves
# ---------------------------------------------------------------------------
def figure_learning_curves(histories, stem="figureS_learning_curves"):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cols = {"compact": C_TRANS, "medium": C_NN, "regularised": "#6E6E6E"}
    names = {"compact": "Compact MLP", "medium": "Medium MLP", "regularised": "Regularised MLP"}
    vmins, tops = [], []
    for name, h in histories.items():
        c = cols.get(name, "#444444")
        val = np.asarray(h.get("val_loss", []), dtype=float)
        tr = np.asarray(h.get("loss", []), dtype=float)
        if tr.size:
            ax.plot(tr, color=c, lw=1.0, ls="--", alpha=0.5)
        if val.size:
            ax.plot(val, color=c, lw=1.6, ls="-", label=names.get(name, name))
            vmins.append(np.nanmin(val))
        # informative range: ignore the first epoch's spike when setting the top
        tail = np.concatenate([a[1:] for a in (val, tr) if a.size > 1]) if (val.size > 1 or tr.size > 1) else np.array([])
        if tail.size:
            tops.append(np.nanpercentile(tail, 98))
    if vmins and tops:
        ax.set_ylim(max(0.0, min(vmins) - 0.03), min(1.0, max(tops) + 0.03))
    style = [Line2D([0], [0], color="#444444", lw=1.6, ls="-", label="Validation"),
             Line2D([0], [0], color="#444444", lw=1.0, ls="--", alpha=0.6, label="Training")]
    leg1 = ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=style, loc="lower left", frameon=False, fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy loss")
    ax.tick_params(length=2.5)
    _save(fig, stem)
