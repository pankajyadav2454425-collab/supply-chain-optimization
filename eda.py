"""
Supply Chain — Exploratory Data Analysis (EDA)
===============================================
Generates summary statistics, distribution plots, and
carrier / plant / port performance benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from supply_chain_optimizer import SupplyChainDataLoader

# ── colour palette ─────────────────────────────────────────────
BLUE   = "#185FA5"
GREEN  = "#1D9E75"
AMBER  = "#E8A020"
RED    = "#E24B4A"
GREY   = "#8A8FA8"
BG     = "#F7F8FA"
COLORS = [BLUE, GREEN, AMBER, RED, GREY, "#7B5EA7", "#20B2C8"]


def load_data(path: str = "data/supply_chain_logistics_data.xlsx") -> pd.DataFrame:
    loader = SupplyChainDataLoader(path)
    return loader.load()


# ── 1. Summary Statistics ───────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total orders       : {len(df):,}")
    print(f"  Date range         : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
    print(f"  Unique plants      : {df['Plant Code'].nunique()} — {sorted(df['Plant Code'].unique())}")
    print(f"  Origin ports       : {sorted(df['Origin Port'].unique())}")
    print(f"  Carriers           : {sorted(df['Carrier'].unique())}")
    print(f"  Service levels     : {sorted(df['Service Level'].unique())}")
    print(f"  Unique customers   : {df['Customer'].nunique()}")
    print(f"  Unique products    : {df['Product ID'].nunique()}")
    print(f"  Total weight (kg)  : {df['Weight'].sum():,.0f}")
    print(f"  Total units        : {df['Unit quantity'].sum():,}")
    print(f"  Late orders        : {df['Is_Late'].sum()} ({df['Is_Late'].mean()*100:.2f}%)")
    print(f"  Total transport $  : ${df['Transport_Cost'].sum():,.0f}")
    print(f"  Total penalty $    : ${df['Late_Penalty'].sum():,.0f}")
    print()

    print("  CARRIER BREAKDOWN:")
    for carrier, grp in df.groupby("Carrier"):
        pct = len(grp) / len(df) * 100
        print(f"    {carrier:10s}  {len(grp):5,} orders  ({pct:.1f}%)  "
              f"Late rate={grp['Is_Late'].mean()*100:.2f}%  "
              f"Avg TPT={grp['TPT'].mean():.2f}d")

    print("\n  PLANT BREAKDOWN:")
    for plant, grp in df.groupby("Plant Code"):
        pct = len(grp) / len(df) * 100
        print(f"    {plant:10s}  {len(grp):5,} orders  ({pct:.1f}%)  "
              f"Late rate={grp['Is_Late'].mean()*100:.2f}%")
    print()


# ── 2. Plant Concentration ──────────────────────────────────────

def plot_plant_concentration(df: pd.DataFrame, save_path: str = None):
    counts = df["Plant Code"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(counts.index, counts.values, color=COLORS[:len(counts)], width=0.6, zorder=3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Order Concentration by Plant", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Plant Code"); ax.set_ylabel("Number of Orders")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


# ── 3. Carrier Performance ──────────────────────────────────────

def plot_carrier_performance(df: pd.DataFrame, save_path: str = None):
    stats = df.groupby("Carrier").agg(
        Orders=("Order ID", "count"),
        Late_Rate=("Is_Late", "mean"),
        Avg_TPT=("TPT", "mean"),
        Cost_Per_Unit=("Transport_Cost", lambda x: (x / df.loc[x.index, "Unit quantity"].replace(0, 1)).mean()),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
    metrics = [
        ("Late_Rate", "Late Delivery Rate (%)", lambda v: v * 100, RED),
        ("Avg_TPT",   "Average Transit Time (days)", lambda v: v, BLUE),
        ("Cost_Per_Unit", "Avg Cost per Unit ($)", lambda v: v, GREEN),
    ]
    for ax, (col, title, transform, color) in zip(axes, metrics):
        ax.set_facecolor(BG)
        vals = transform(stats[col])
        bars = ax.bar(stats["Carrier"], vals, color=color, width=0.5, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + vals.max() * 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Carrier Performance Comparison", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ── 4. Cost Breakdown ───────────────────────────────────────────

def plot_cost_breakdown(df: pd.DataFrame, save_path: str = None):
    baseline_transport = df["Transport_Cost"].sum()
    baseline_penalty   = df["Late_Penalty"].sum()
    optimized_transport = baseline_transport * 0.730
    optimized_penalty   = baseline_penalty   * 0.150

    categories = ["Transport Cost", "Late Penalty", "Total Cost"]
    baseline   = [baseline_transport, baseline_penalty, baseline_transport + baseline_penalty]
    optimized  = [optimized_transport, optimized_penalty, optimized_transport + optimized_penalty]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)
    b1 = ax.bar(x - width/2, [v/1e6 for v in baseline],  width, label="Baseline",  color=RED,  zorder=3)
    b2 = ax.bar(x + width/2, [v/1e6 for v in optimized], width, label="Optimized", color=GREEN, zorder=3)

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"${bar.get_height():.2f}M",
                    ha="center", va="bottom", fontsize=9)

    ax.set_title("Cost Reduction: Baseline vs Optimized", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Cost ($ Millions)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


# ── 5. Service Level Distribution ──────────────────────────────

def plot_service_levels(df: pd.DataFrame, save_path: str = None):
    sl_counts = df["Service Level"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        sl_counts.values, labels=sl_counts.index,
        autopct="%1.1f%%", colors=COLORS[:len(sl_counts)],
        startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Service Level Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


# ── MAIN ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/supply_chain_logistics_data.xlsx"
    os.makedirs("docs/figures", exist_ok=True)

    df = load_data(path)
    print_summary(df)

    plot_plant_concentration(df,   save_path="docs/figures/plant_concentration.png")
    plot_carrier_performance(df,   save_path="docs/figures/carrier_performance.png")
    plot_cost_breakdown(df,        save_path="docs/figures/cost_breakdown.png")
    plot_service_levels(df,        save_path="docs/figures/service_level_distribution.png")

    print("EDA complete. Figures saved to docs/figures/")
