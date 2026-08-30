from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from ..metrics import enrich_orders_with_costs

def build_route_graph(data: dict, objective="balanced"):
    df = enrich_orders_with_costs(data)
    need = ["Plant Code", "Origin Port", "Carrier", "Destination Port", "Customer"]
    if not all(c in df.columns for c in need):
        raise ValueError(f"Orders need columns: {need}")

    # Normalize components for a stable composite route weight.
    tmp = df.copy()
    tmp["Route Cost"] = tmp["Total Cost"].clip(lower=0)
    tmp["Route Time"] = tmp.get("TPT", pd.Series(0, index=tmp.index)).clip(lower=0)
    tmp["Route Risk"] = tmp.get("Ship Late Day count", pd.Series(0, index=tmp.index)).clip(lower=0)

    def norm(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.0, index=s.index)
        return (s-lo)/(hi-lo)

    c, t, r = norm(tmp["Route Cost"]), norm(tmp["Route Time"]), norm(tmp["Route Risk"])
    if objective == "lowest_cost":
        tmp["_w"] = c
    elif objective == "fastest":
        tmp["_w"] = t
    elif objective == "most_reliable":
        tmp["_w"] = r
    else:
        tmp["_w"] = .40*c + .35*t + .25*r

    G = nx.DiGraph()
    grouped = tmp.groupby(need, dropna=False)["_w"].mean().reset_index()
    for _, row in grouped.iterrows():
        p = f"Plant::{row['Plant Code']}"
        o = f"Origin::{row['Origin Port']}"
        car = f"Carrier::{row['Carrier']}"
        d = f"Dest::{row['Destination Port']}"
        cust = f"Customer::{row['Customer']}"
        w = float(row["_w"]) + 1e-6
        for a, b in [(p,o),(o,car),(car,d),(d,cust)]:
            prev = G.get_edge_data(a,b,{}).get("weight")
            if prev is None or w < prev:
                G.add_edge(a,b,weight=w)
    return G

def shortest_route(data: dict, plant: str, customer: str, objective="balanced"):
    G = build_route_graph(data, objective)
    src = f"Plant::{plant}"
    dst = f"Customer::{customer}"
    try:
        path = nx.shortest_path(G, src, dst, weight="weight", method="dijkstra")
        dist = nx.shortest_path_length(G, src, dst, weight="weight", method="dijkstra")
        return {"success": True, "path": path, "score": float(dist), "message": "Route found."}
    except Exception as e:
        return {"success": False, "path": [], "score": np.nan, "message": str(e)}
