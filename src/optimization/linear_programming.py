from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from ..metrics import enrich_orders_with_costs

def _capacity_map(data, orders, demand_total, capacity_multiplier=1.0, disabled_plant=None):
    plants = sorted(orders["Plant Code"].astype(str).unique())
    caps = {p: np.inf for p in plants}
    wc = data.get("warehouse_capacities", pd.DataFrame()).copy()
    if not wc.empty:
        lower = {str(c).strip().lower(): c for c in wc.columns}
        pcol = next((lower[k] for k in lower if k in {"plant id", "plant code", "wh", "warehouse"}), None)
        ccol = next((lower[k] for k in lower if "capacity" in k), None)
        if pcol and ccol:
            days = 1
            if "Order Date" in orders and orders["Order Date"].notna().any():
                dates = pd.to_datetime(orders["Order Date"], errors="coerce")
                days = max(1, int((dates.max() - dates.min()).days) + 1)
            for _, r in wc.iterrows():
                plant = str(r[pcol]).strip()

                # Only use capacity for plants that actually exist in OrderList
                if plant in caps:
                    value = pd.to_numeric(r[ccol], errors="coerce")
            
                    if pd.notna(value) and float(value) > 0:
                        caps[plant] = float(value) * days
    # Fallback capacity from historical order counts with 20% headroom.
    hist = orders.groupby("Plant Code").size().to_dict()
    for p in plants:
        if not np.isfinite(caps[p]) or caps[p] <= 0:
            caps[p] = max(1.0, float(hist.get(p, 0)) * 1.2)
        caps[p] *= float(capacity_multiplier)
    if disabled_plant is not None and str(disabled_plant) in caps:
        caps[str(disabled_plant)] = 0.0
    if sum(caps.values()) < demand_total:
        scale = demand_total / max(sum(caps.values()), 1e-9)
        # keep infeasibility visible if a plant was deliberately disabled
        if disabled_plant is None:
            caps = {k: v * scale * 1.01 for k, v in caps.items()}
    return caps

def solve_transport_lp(
    data: dict,
    demand_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
    disabled_plant: str | None = None,
):
    df = enrich_orders_with_costs(data)
    if "Plant Code" not in df or "Customer" not in df:
        raise ValueError("Orders need Plant Code and Customer columns.")
    df["Plant Code"] = df["Plant Code"].astype(str)
    df["Customer"] = df["Customer"].astype(str)
    plants = sorted(df["Plant Code"].unique())
    customers = sorted(df["Customer"].unique())

    demand_hist = df.groupby("Customer").size().astype(float)
    demand = {c: max(0.0, float(demand_hist.get(c, 0)) * float(demand_multiplier)) for c in customers}
    total_demand = sum(demand.values())
    caps = _capacity_map(data, df, total_demand, capacity_multiplier, disabled_plant)

    pc = df.groupby(["Plant Code", "Customer"])["Total Cost"].mean()
    plant_avg = df.groupby("Plant Code")["Total Cost"].mean()
    global_avg = float(df["Total Cost"].mean()) if len(df) else 1.0

    var_keys = [(p, c) for p in plants for c in customers]
    costs = []
    for p, c in var_keys:
        v = pc.get((p, c), np.nan)
        if pd.isna(v):
            v = plant_avg.get(p, global_avg) * 1.15
        costs.append(float(v) * float(cost_multiplier))

    # Equality: each customer's demand must be met.
    A_eq, b_eq = [], []
    for c in customers:
        row = [1.0 if cc == c else 0.0 for p, cc in var_keys]
        A_eq.append(row); b_eq.append(demand[c])

    # Inequality: plant capacity.
    A_ub, b_ub = [], []
    for p in plants:
        row = [1.0 if pp == p else 0.0 for pp, c in var_keys]
        A_ub.append(row); b_ub.append(caps[p])

    res = linprog(
        c=np.array(costs),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=(0, None),
        method="highs",
    )
    alloc = []
    if res.success:
        for (p, c), x, unit_cost in zip(var_keys, res.x, costs):
            if x > 1e-7:
                alloc.append({
                    "Plant": p, "Customer": c,
                    "Allocated Orders": float(x),
                    "Unit Cost": float(unit_cost),
                    "Total Cost": float(x * unit_cost),
                })
    return {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "objective": float(res.fun) if res.success else np.nan,
        "allocations": pd.DataFrame(alloc),
        "capacities": pd.DataFrame({"Plant": list(caps), "Capacity": list(caps.values())}),
        "demand": pd.DataFrame({"Customer": list(demand), "Demand": list(demand.values())}),
    }
