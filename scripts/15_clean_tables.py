"""Step 15 - build reader-facing, internally consistent tables for the manuscript.

Produces, from the verified result CSVs and manuscript/labels.json:
  - tableS_settings.csv : the 18 prespecified settings with n, events, event rate
  - table5_decision_clean.csv : best NN vs best transparent for the 18 settings,
    with reader-facing names, REMBRANDT relabelled to its available features, and
    the four components of the prespecified criterion.
Asserts the canonical setting count and that referenced settings exist.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config

ROOT = pathlib.Path(__file__).resolve().parents[1]
labels = json.load(open(ROOT / "manuscript" / "labels.json"))
MODELS = labels["models"]
SETTINGS = labels["settings"]
assert len(SETTINGS) == 18, f"expected 18 settings, got {len(SETTINGS)}"

t3 = pd.read_csv(config.TABLES / "table3_internal_performance.csv")
t4 = pd.read_csv(config.TABLES / "table4_external_performance.csv")
t5 = pd.read_csv(config.TABLES / "table5_nn_vs_transparent.csv")
perf = pd.concat([t3, t4], ignore_index=True)


def n_event(code):
    sub = perf[perf.setting == code]
    if len(sub) == 0:
        return ("", "")
    r = sub.iloc[0]
    return (int(r["n"]), f"{float(r['event_rate']):.2f}")


# settings table (numbered 1..18 to make the prespecification auditable)
srows = []
for i, s in enumerate(SETTINGS, 1):
    n, rate = n_event(s["code"])
    srows.append({"no": i, "group": s["group"], "setting": s["name"], "n": n, "event_rate": rate})
pd.DataFrame(srows).to_csv(config.TABLES / "tableS_settings.csv", index=False)

# clean decision table
name = {s["code"]: s["name"] for s in SETTINGS}
drows = []
for i, s in enumerate(SETTINGS, 1):
    r = t5[t5.setting == s["code"]]
    if len(r) == 0:
        raise SystemExit(f"missing decision row for {s['code']}")
    r = r.iloc[0]
    drows.append({
        "no": i,
        "setting": s["name"],
        "best_nn": MODELS.get(r["best_tf_variant"], r["best_tf_variant"]),
        "best_transparent": MODELS.get(r["best_transparent"], r["best_transparent"]),
        "auroc_nn": r["auroc_nn"], "auroc_transparent": r["auroc_transparent"],
        "delta_auroc_ci": f"{r['delta_auroc']} ({r['delta_lo']}, {r['delta_hi']})",
        "auroc_margin_met": r["passes_auroc_margin"],
        "brier_not_worse": r["not_worse_brier"],
        "calibration_ok": r["calibration_acceptable"],
        "criterion_met": r["clinically_meaningful"],
    })
clean = pd.DataFrame(drows)
clean.to_csv(config.TABLES / "table5_decision_clean.csv", index=False)
print(f"settings: {len(SETTINGS)} | decision rows: {len(clean)} | criterion met: {int((clean.criterion_met==True).sum())}")
print("AUROC numerically higher for NN in:",
      int((pd.to_numeric(clean.auroc_nn) > pd.to_numeric(clean.auroc_transparent)).sum()), "of 18 settings")
