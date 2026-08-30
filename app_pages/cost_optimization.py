import streamlit as st
import pandas as pd
import plotly.express as px

def render(ctx):
    st.header("💰 Cost Optimization")
    b, lp = ctx["baseline"], ctx["lp"]
    if not lp["success"]:
        st.error(f"LP optimization failed: {lp['message']}")
        return
    opt = lp["objective"]
    sav = b["total_cost"] - opt
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Baseline Cost", f"${b['total_cost']:,.0f}")
    c2.metric("Optimized Cost", f"${opt:,.0f}")
    c3.metric("Savings", f"${sav:,.0f}")
    c4.metric("Savings %", f"{sav/b['total_cost']*100:.1f}%")

    comp = pd.DataFrame({"Scenario":["Baseline","Optimized"],"Cost":[b["total_cost"],opt]})
    st.plotly_chart(px.bar(comp, x="Scenario", y="Cost", title="Baseline vs Optimized Cost"), use_container_width=True)
    st.subheader("Optimized Allocation")
    st.dataframe(lp["allocations"], use_container_width=True, hide_index=True)
    csv = lp["allocations"].to_csv(index=False).encode()
    st.download_button("Download Optimization Results", csv, "optimized_allocations.csv", "text/csv")
