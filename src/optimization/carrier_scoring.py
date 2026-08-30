from __future__ import annotations
import numpy as np
import pandas as pd
from ..metrics import enrich_orders_with_costs

def _benefit_norm(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(100.0, index=s.index)
    return 100 * (s - lo) / (hi - lo)

def _cost_norm(s):
    return 100 - _benefit_norm(s)

def score_carriers(data: dict, cost_weight=.20, speed_weight=.35, reliability_weight=.45):
    df = enrich_orders_with_costs(data)
    g = df.groupby("Carrier", dropna=False).agg(
        Orders=("Carrier", "size"),
        Cost_Per_Order=("Total Cost", "mean"),
        Avg_TPT=("TPT", "mean") if "TPT" in df.columns else ("Carrier", "size"),
        On_Time_Rate=("On Time", "mean"),
    ).reset_index()
    g["On_Time_Rate"] *= 100
    g["Cost_Score"] = _cost_norm(g["Cost_Per_Order"])
    g["Speed_Score"] = _cost_norm(g["Avg_TPT"])
    g["Reliability_Score"] = _benefit_norm(g["On_Time_Rate"])
    total_w = max(cost_weight + speed_weight + reliability_weight, 1e-9)
    cw, sw, rw = cost_weight/total_w, speed_weight/total_w, reliability_weight/total_w
    g["Carrier_Score"] = (
        cw*g["Cost_Score"] + sw*g["Speed_Score"] + rw*g["Reliability_Score"]
    )
    return g.sort_values("Carrier_Score", ascending=False).reset_index(drop=True)
