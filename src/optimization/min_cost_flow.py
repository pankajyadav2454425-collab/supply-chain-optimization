from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from ..metrics import enrich_orders_with_costs
from .linear_programming import _capacity_map

def solve_min_cost_flow(data: dict, demand_multiplier=1.0, cost_multiplier=1.0,
                        capacity_multiplier=1.0, disabled_plant=None):
    df = enrich_orders_with_costs(data)
    df["Plant Code"] = df["Plant Code"].astype(str)
    df["Customer"] = df["Customer"].astype(str)
    plants = sorted(df["Plant Code"].unique())
    customers = sorted(df["Customer"].unique())
    demand_hist = df.groupby("Customer").size()
    demand = {c: int(round(float(demand_hist.get(c, 0))*demand_multiplier)) for c in customers}
    total = sum(demand.values())
    caps_float = _capacity_map(data, df, total, capacity_multiplier, disabled_plant)
    caps = {p: int(np.floor(v)) for p, v in caps_float.items()}

    pc = df.groupby(["Plant Code", "Customer"])["Total Cost"].mean()
    plant_avg = df.groupby("Plant Code")["Total Cost"].mean()
    global_avg = float(df["Total Cost"].mean()) if len(df) else 1.0

    G = nx.DiGraph()
    G.add_node("SOURCE", demand=-total)
    for p in plants:
        G.add_node(f"P::{p}", demand=0)
        G.add_edge("SOURCE", f"P::{p}", capacity=max(caps[p], 0), weight=0)
    for c in customers:
        G.add_node(f"C::{c}", demand=demand[c])
    scale = 100
    for p in plants:
        for c in customers:
            unit = pc.get((p, c), np.nan)
            if pd.isna(unit):
                unit = plant_avg.get(p, global_avg) * 1.15
            G.add_edge(
                f"P::{p}", f"C::{c}",
                capacity=max(caps[p], total),
                weight=max(0, int(round(float(unit)*cost_multiplier*scale)))
            )
    try:
        cost, flow = nx.network_simplex(G)
        rows = []
        for p in plants:
            pdata = flow.get(f"P::{p}", {})
            for node, qty in pdata.items():
                if node.startswith("C::") and qty > 0:
                    rows.append({
                        "Plant": p,
                        "Customer": node.replace("C::", "", 1),
                        "Flow Orders": qty,
                    })
        return {
            "success": True,
            "objective": cost/scale,
            "flows": pd.DataFrame(rows),
            "message": "Optimal min-cost flow found.",
        }
    except Exception as e:
        return {
            "success": False,
            "objective": np.nan,
            "flows": pd.DataFrame(),
            "message": str(e),
        }
