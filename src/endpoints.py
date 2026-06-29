"""Binary overall-survival endpoint construction with explicit censoring rules.

Primary strategy (per PROTOCOL.md): for a horizon H months, label
    y = 1 (death by H)  if event occurred and time <= H
    y = 0 (alive at H)  if time >= H (regardless of event)
Patients censored before H (event == 0 and time < H) are EXCLUDED from the
binary endpoint, because their status at H is unknown. The number excluded is
recorded so it can be reported and their characteristics compared.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_binary_endpoint(df: pd.DataFrame, horizon_months: int,
                         time_col: str = "os_months", event_col: str = "os_event"):
    """Return (labelled_df, info) where labelled_df has column 'y' in {0,1}.

    info documents inclusion/exclusion counts for transparent reporting.
    """
    t = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.to_numeric(df[event_col], errors="coerce")
    valid = t.notna() & e.notna()

    death_by_h = valid & (e == 1) & (t <= horizon_months)
    alive_at_h = valid & (t >= horizon_months)
    censored_before = valid & (e == 0) & (t < horizon_months)

    y = pd.Series(np.nan, index=df.index)
    y[death_by_h] = 1.0
    y[alive_at_h & ~death_by_h] = 0.0

    keep = y.notna()
    out = df.loc[keep].copy()
    out["y"] = y[keep].astype(int)

    info = {
        "horizon_months": horizon_months,
        "n_input": int(len(df)),
        "n_missing_time_or_status": int((~valid).sum()),
        "n_censored_before_horizon_excluded": int(censored_before.sum()),
        "n_labelled": int(keep.sum()),
        "n_events_death": int((out["y"] == 1).sum()),
        "n_nonevents_alive": int((out["y"] == 0).sum()),
        "event_rate": float(out["y"].mean()) if len(out) else float("nan"),
    }
    return out, info
