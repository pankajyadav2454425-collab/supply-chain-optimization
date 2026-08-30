from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from ..metrics import enrich_orders_with_costs
from .linear_programming import _capacity_map

def solve_plant_activation_milp(
    data: dict,
    demand_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
    disabled_plant: str | None = None,
    fixed_cost_fraction: float = 0.05,
):
    df = enrich_orders_with_costs(data)
    df["Plant Code"] = df["Plant Code"].astype(str)
    df["Customer"] = df["Customer"].astype(str)
    plants = sorted(df["Plant Code"].unique())
    customers = sorted(df["Customer"].unique())
    demand_hist = df.groupby("Customer").size().astype(float)
    demand = {c: float(demand_hist.get(c, 0)) * float(demand_multiplier) for c in customers}
    total_demand = sum(demand.values())
    caps = _capacity_map(data, df, total_demand, capacity_multiplier, disabled_plant)

    pc = df.groupby(["Plant Code", "Customer"])["Total Cost"].mean()
    plant_avg = df.groupby("Plant Code")["Total Cost"].mean()
    global_avg = float(df["Total Cost"].mean()) if len(df) else 1.0
    var_x = [(p, c) for p in plants for c in customers]
    n_x = len(var_x)
    n_y = len(plants)

    c = np.zeros(n_x + n_y)
    for k, (p, cust) in enumerate(var_x):
        unit = pc.get((p, cust), np.nan)
        if pd.isna(unit):
            unit = plant_avg.get(p, global_avg) * 1.15
        c[k] = float(unit) * float(cost_multiplier)

    hist_cost = df.groupby("Plant Code")["Total Cost"].sum()
    for i, p in enumerate(plants):
        c[n_x + i] = float(hist_cost.get(p, global_avg)) * float(fixed_cost_fraction)

    # Constraints:
    # 1) customer demand equality
    # 2) sum_c x[p,c] - cap[p]*y[p] <= 0
    rows = len(customers) + len(plants)
    A = lil_matrix((rows, n_x + n_y), dtype=float)
    lb = np.full(rows, -np.inf)
    ub = np.full(rows, np.inf)

    r = 0
    for cust in customers:
        for k, (p, cst) in enumerate(var_x):
            if cst == cust:
                A[r, k] = 1
        lb[r] = demand[cust]; ub[r] = demand[cust]
        r += 1

    for i, p in enumerate(plants):
        for k, (pp, cust) in enumerate(var_x):
            if pp == p:
                A[r, k] = 1
        A[r, n_x + i] = -caps[p]
        ub[r] = 0
        r += 1

    integrality = np.zeros(n_x + n_y, dtype=int)
    integrality[n_x:] = 1
    lower = np.zeros(n_x + n_y)
    upper = np.full(n_x + n_y, np.inf)
    upper[n_x:] = 1
    if disabled_plant is not None and str(disabled_plant) in plants:
        idx = plants.index(str(disabled_plant))
        upper[n_x + idx] = 0

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options={"disp": False},
    )

    alloc, activation = [], []
    if res.success:
        x = res.x[:n_x]
        y = res.x[n_x:]
        for (p, cust), qty, unit_cost in zip(var_x, x, c[:n_x]):
            if qty > 1e-7:
                alloc.append({
                    "Plant": p, "Customer": cust,
                    "Allocated Orders": float(qty),
                    "Unit Cost": float(unit_cost),
                    "Total Cost": float(qty * unit_cost),
                })
        for p, yy, fc in zip(plants, y, c[n_x:]):
            activation.append({
                "Plant": p,
                "Activate": int(round(float(yy))),
                "Fixed Cost": float(fc),
                "Capacity": float(caps[p]),
            })
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "objective": float(res.fun) if res.success else np.nan,
        "allocations": pd.DataFrame(alloc),
        "activation": pd.DataFrame(activation),
    }
