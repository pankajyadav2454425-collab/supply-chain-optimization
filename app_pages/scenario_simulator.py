import streamlit as st
import pandas as pd
import plotly.express as px
from src.optimization.scenario_analysis import run_scenario

def render(ctx):
    st.header("⚡ Scenario Simulator")
    df = ctx["orders"]
    c1,c2,c3 = st.columns(3)
    demand = c1.slider("Demand Change %", -20, 50, 0)
    freight = c2.slider("Freight Cost Change %", -20, 100, 0)
    capacity = c3.slider("Capacity Change %", -100, 50, 0)
    plants = ["None"] + sorted(df["Plant Code"].astype(str).unique())
    disabled = st.selectbox("Plant Shutdown", plants)
    disabled = None if disabled == "None" else disabled

    if st.button("Run Scenario", type="primary"):
        with st.spinner("Re-optimizing network..."):
            out = run_scenario(ctx["data"], demand, freight, capacity, disabled)
        lp = out["lp"]
        if not lp["success"]:
            st.error(f"Scenario infeasible or failed: {lp['message']}")
            return
        baseline = ctx["baseline"]["total_cost"]
        scenario = lp["objective"]
        delta = scenario - baseline
        a,b,c = st.columns(3)
        a.metric("Baseline Cost", f"${baseline:,.0f}")
        b.metric("Scenario Cost", f"${scenario:,.0f}", delta=f"${delta:,.0f}")
        c.metric("Cost Change %", f"{delta/baseline*100:.1f}%")
        st.dataframe(lp["allocations"], use_container_width=True, hide_index=True)
