from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

SHEET_ALIASES = {
    "orders": ["orderlist", "orders", "order list"],
    "freight_rates": ["freightrates", "freight rates"],
    "plant_ports": ["plantports", "plant ports"],
    "products_per_plant": ["productsperplant", "products per plant"],
    "vmi_customers": ["vmicustomers", "vmi customers"],
    "warehouse_capacities": ["whcapacities", "warehouse capacities", "wh capacities"],
    "warehouse_costs": ["whcosts", "warehouse costs", "wh costs"],
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _find_sheet(all_sheets, aliases):
    norm_map = {_norm(k): k for k in all_sheets}
    for alias in aliases:
        a = _norm(alias)
        if a in norm_map:
            return norm_map[a]
    return None

def load_supply_chain_data(filepath: str | Path) -> dict[str, pd.DataFrame]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    sheets = pd.read_excel(filepath, sheet_name=None)
    out = {}
    missing = []
    for canonical, aliases in SHEET_ALIASES.items():
        found = _find_sheet(sheets, aliases)
        if found is None:
            missing.append(canonical)
            out[canonical] = pd.DataFrame()
        else:
            out[canonical] = sheets[found].copy()
    if out["orders"].empty:
        raise ValueError(
            "OrderList sheet was not found. Available sheets: "
            + ", ".join(sheets.keys())
        )
    out["_sheet_names"] = pd.DataFrame({"sheet": list(sheets.keys())})
    return out

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    numeric_candidates = [
        "TPT", "Ship ahead day count", "Ship Late Day count",
        "Unit quantity", "Weight"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df

def validate_data(data: dict[str, pd.DataFrame]) -> dict:
    orders = clean_orders(data["orders"])
    required = [
        "Order ID", "Plant Code", "Customer", "Carrier",
        "Unit quantity", "Weight"
    ]
    missing_cols = [c for c in required if c not in orders.columns]
    duplicate_orders = int(orders["Order ID"].duplicated().sum()) if "Order ID" in orders else 0
    negative_qty = int((orders.get("Unit quantity", pd.Series(dtype=float)) < 0).sum())
    negative_weight = int((orders.get("Weight", pd.Series(dtype=float)) < 0).sum())
    missing_cells = int(orders.isna().sum().sum())
    return {
        "rows": len(orders),
        "missing_required_columns": missing_cols,
        "duplicate_order_ids": duplicate_orders,
        "negative_quantity_rows": negative_qty,
        "negative_weight_rows": negative_weight,
        "missing_cells": missing_cells,
        "is_valid": not missing_cols and negative_qty == 0 and negative_weight == 0,
    }
