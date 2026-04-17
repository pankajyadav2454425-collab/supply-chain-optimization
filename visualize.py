"""
Supply Chain — Results Visualization & Report Generator
=======================================================
Generates all charts for the optimization report and saves
them to docs/figures/.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from supply_chain_optimizer import (
    SupplyChainDataLoader, WeightedScoringModel,
    PlantActivationMILP, ScenarioAnalyzer
)

BLUE   = "#185FA5"
GREEN  = "#1D9E75"
AMBER  = "#E8A020"
RED    = "#E24B4A"
PURPLE = "#7B5EA7"
BG     = "#F7F8FA"
os.makedirs("docs/figures", exist_ok=True)


def load(path: str) -> pd.DataFrame:
    return SupplyChainDataLoader(path).load()


# ── Figure 1: Carrier Reallocation ─────────────────────────────

def fig_carrier_reallocation(save: bool = True):
    carriers = ["V444_0", "V444_1", "V44_3"]
    before   = [68.0, 22.8, 9.3]
    after    = [40.0, 35.0, 25.0]

    x = np.arange(len(carriers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    b1 = ax.bar(x - w/2, before, w, label="Before", color=RED,   zorder=3)
    b2 = ax.bar(x + w/2, after,  w, label="After",  color=GREEN, zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(carriers)
    ax.set_ylabel("Share of Orders (%)")
    ax.set_title("Carrier Reallocation: Before vs After Optimization", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("docs/figures/carrier_reallocation.png", dpi=150)
    plt.show()


# ── Figure 2: Plant Rebalancing ─────────────────────────────────

def fig_plant_rebalancing(save: bool = True):
    plants  = ["PLANT03", "PLANT08", "PLANT09", "PLANT12", "PLANT13", "PLANT16"]
    before  = [92.7, 1.1, 0.1, 3.3, 0.9, 1.9]
    after   = [74.0, 4.0, 8.0, 8.0, 5.0, 1.0]

    x = np.arange(len(plants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)
    b1 = ax.bar(x - w/2, before, w, label="Before", color=BLUE,  zorder=3)
    b2 = ax.bar(x + w/2, after,  w, label="After",  color=GREEN, zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(plants)
    ax.set_ylabel("Order Share (%)")
    ax.set_title("Plant Load Rebalancing: Before vs After", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("docs/figures/plant_rebalancing.png", dpi=150)
    plt.show()


# ── Figure 3: Scenario Risk Matrix ─────────────────────────────

def fig_scenario_analysis(baseline_cost: float, save: bool = True):
    analyzer = ScenarioAnalyzer(baseline_cost)
    df = analyzer.run_all()

    risk_colors = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": RED,
                   "CRITICAL": "#8B0000", "BASELINE": BLUE}
    colors = [risk_colors[r] for r in df["Risk_Level"]]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(df["Scenario"], df["Cost_Change_pct"], color=colors, zorder=3)
    for bar, val in zip(bars, df["Cost_Change_pct"]):
        xpos = val + 0.3 if val >= 0 else val - 0.3
        ha   = "left"    if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cost Change vs Baseline (%)")
    ax.set_title("Scenario Analysis — Cost Impact", fontsize=13, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for l, c in risk_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("docs/figures/scenario_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Figure 4: WSM Carrier Scorecard ────────────────────────────

def fig_wsm_scorecard(df: pd.DataFrame, save: bool = True):
    wsm = WeightedScoringModel(df)
    scores = wsm.score_carriers()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    colors = [GREEN, BLUE, RED][:len(scores)]

    # Composite score bar
    ax = axes[0]; ax.set_facecolor(BG)
    bars = ax.bar(scores["Carrier"], scores["Composite_Score"], color=colors, width=0.5, zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("WSM Composite Score (higher = better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score"); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    # Late rate comparison
    ax2 = axes[1]; ax2.set_facecolor(BG)
    bars2 = ax2.bar(scores["Carrier"], scores["Late_Rate_pct"], color=colors, width=0.5, zorder=3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10)
    ax2.set_title("Late Delivery Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Late Rate (%)"); ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines[["top","right"]].set_visible(False)

    plt.suptitle("Carrier Scorecard — Weighted Scoring Model", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig("docs/figures/wsm_carrier_scorecard.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Figure 5: KPI Summary Waterfall ────────────────────────────

def fig_kpi_waterfall(save: bool = True):
    labels  = ["Baseline Cost", "LP Savings", "Carrier Switch", "Plant Rebalance", "Optimized Cost"]
    values  = [20_326_227, -4_882_889, -1_000_000, -905_237, 13_538_101]
    running = [20_326_227, 15_443_338, 14_443_338, 13_538_101, 13_538_101]
    colors  = [BLUE, GREEN, GREEN, GREEN, BLUE]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)
    bottoms = [0, running[0], running[1], running[2], 0]
    for i, (label, val, bottom, color) in enumerate(zip(labels, values, bottoms, colors)):
        ax.bar(i, abs(val), bottom=bottom if val > 0 else running[i-1] + val,
               color=color, width=0.5, zorder=3, alpha=0.9)
        ypos = (bottom + abs(val) / 2) if val > 0 else (running[i-1] + val + abs(val) / 2)
        ax.text(i, ypos, f"${abs(val)/1e6:.2f}M", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Cost ($ Millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.set_title("Cost Reduction Waterfall", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("docs/figures/cost_waterfall.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── MAIN ────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/supply_chain_logistics_data.xlsx"
    df = load(path)

    print("Generating all visualization figures...")
    fig_carrier_reallocation()
    fig_plant_rebalancing()
    fig_scenario_analysis(df["Total_Cost"].sum())
    fig_wsm_scorecard(df)
    fig_kpi_waterfall()
    print("All figures saved to docs/figures/")
