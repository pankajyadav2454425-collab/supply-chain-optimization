from __future__ import annotations
import numpy as np
import pandas as pd
from .data_loader import clean_orders

def _col(df, candidates):
    norm = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace("_", " ")
        if key in norm:
            return norm[key]
    return None

def enrich_orders_with_costs(data: dict, late_penalty_per_unit_day: float = 5.0) -> pd.DataFrame:
    orders = clean_orders(data["orders"])
    out = orders.copy()
    out["Freight Cost"] = np.nan

    fr = data.get("freight_rates", pd.DataFrame()).copy()
    if not fr.empty:
        c_car = _col(fr, ["Carrier"])
        c_org = _col(fr, ["orig_port_cd", "Origin Port"])
        c_dst = _col(fr, ["dest_port_cd", "Destination Port"])
        c_minw = _col(fr, ["minm_wgh_qty", "min weight"])
        c_maxw = _col(fr, ["max_wgh_qty", "max weight"])
        c_min = _col(fr, ["minimum cost", "minimum_cost"])
        c_rate = _col(fr, ["rate"])
        if all([c_car, c_org, c_dst, c_rate]) and all(
            x in out.columns for x in ["Carrier", "Origin Port", "Destination Port", "Weight"]
        ):
            fr["_carrier"] = fr[c_car].astype(str)
            fr["_org"] = fr[c_org].astype(str)
            fr["_dst"] = fr[c_dst].astype(str)
            fr["_rate"] = pd.to_numeric(fr[c_rate], errors="coerce")
            fr["_min"] = pd.to_numeric(fr[c_min], errors="coerce").fillna(0) if c_min else 0
            fr["_minw"] = pd.to_numeric(fr[c_minw], errors="coerce").fillna(-np.inf) if c_minw else -np.inf
            fr["_maxw"] = pd.to_numeric(fr[c_maxw], errors="coerce").fillna(np.inf) if c_maxw else np.inf

            lookup = {}
            for key, g in fr.groupby(["_carrier", "_org", "_dst"], dropna=False):
                lookup[key] = g

            costs = []
            for _, r in out.iterrows():
                key = (str(r["Carrier"]), str(r["Origin Port"]), str(r["Destination Port"]))
                g = lookup.get(key)
                weight = float(r.get("Weight", 0) or 0)
                cost = np.nan
                if g is not None and len(g):
                    band = g[(g["_minw"] <= weight) & (weight <= g["_maxw"])]
                    if band.empty:
                        band = g.iloc[[0]]
                    rr = band.iloc[0]
                    cost = max(float(rr["_min"]), float(rr["_rate"]) * weight)
                costs.append(cost)
            out["Freight Cost"] = costs

    # Fallback cost if freight match is unavailable.
    fallback = (
        out.get("Weight", pd.Series(0, index=out.index)).fillna(0) *
        out.get("TPT", pd.Series(1, index=out.index)).replace(0, 1).fillna(1) *
        50.0
    )
    out["Freight Cost"] = out["Freight Cost"].fillna(fallback)

    wc = data.get("warehouse_costs", pd.DataFrame()).copy()
    out["Warehouse Cost"] = 0.0
    if not wc.empty and "Plant Code" in out.columns:
        c_plant = _col(wc, ["WH", "Plant Code", "Warehouse"])
        c_cost = _col(wc, ["Cost/unit", "Cost per unit", "Cost"])
        if c_plant and c_cost:
            mp = dict(zip(wc[c_plant].astype(str), pd.to_numeric(wc[c_cost], errors="coerce").fillna(0)))
            out["Warehouse Cost"] = (
                out["Plant Code"].astype(str).map(mp).fillna(0)
                * out.get("Unit quantity", pd.Series(0, index=out.index)).fillna(0)
            )

    late_days = out.get("Ship Late Day count", pd.Series(0, index=out.index)).fillna(0).clip(lower=0)
    qty = out.get("Unit quantity", pd.Series(0, index=out.index)).fillna(0)
    out["Late Penalty"] = late_days * qty * float(late_penalty_per_unit_day)
    out["Total Cost"] = out["Freight Cost"] + out["Warehouse Cost"] + out["Late Penalty"]
    out["On Time"] = late_days <= 0
    return out

def calculate_baseline_metrics(data: dict, late_penalty_per_unit_day: float = 5.0) -> dict:
    df = enrich_orders_with_costs(data, late_penalty_per_unit_day)
    total_orders = len(df)
    total_cost = float(df["Total Cost"].sum())
    return {
        "total_orders": total_orders,
        "total_units": float(df.get("Unit quantity", pd.Series(dtype=float)).sum()),
        "total_weight": float(df.get("Weight", pd.Series(dtype=float)).sum()),
        "freight_cost": float(df["Freight Cost"].sum()),
        "warehouse_cost": float(df["Warehouse Cost"].sum()),
        "late_penalty": float(df["Late Penalty"].sum()),
        "total_cost": total_cost,
        "late_orders": int((~df["On Time"]).sum()),
        "on_time_rate": float(df["On Time"].mean() * 100) if total_orders else 0.0,
        "avg_tpt": float(df.get("TPT", pd.Series(dtype=float)).mean()) if total_orders else 0.0,
        "cost_per_order": total_cost / total_orders if total_orders else 0.0,
        "active_plants": int(df["Plant Code"].nunique()) if "Plant Code" in df else 0,
        "active_carriers": int(df["Carrier"].nunique()) if "Carrier" in df else 0,
    }
