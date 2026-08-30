import streamlit as st
import pandas as pd
import plotly.express as px

def render(ctx):
    b, df, lp = ctx["baseline"], ctx["orders"], ctx["lp"]
    st.title("🚚 Supply Chain Optimization Control Tower")
    st.caption("Executive view of logistics cost, service performance, capacity and optimization opportunities.")

    opt_cost = lp["objective"] if lp["success"] else float("nan")
    savings = b["total_cost"] - opt_cost if lp["success"] else float("nan")
    savings_pct = savings / b["total_cost"] * 100 if lp["success"] and b["total_cost"] else float("nan")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Orders", f"{b['total_orders']:,}")
    c2.metric("Baseline Cost", f"${b['total_cost']:,.0f}")
    c3.metric("Optimized Cost", f"${opt_cost:,.0f}" if lp["success"] else "N/A")
    c4.metric("Savings", f"${savings:,.0f}" if lp["success"] else "N/A")
    c5.metric("Savings %", f"{savings_pct:.1f}%" if lp["success"] else "N/A")
    c6.metric("On-Time", f"{b['on_time_rate']:.1f}%")

    left, right = st.columns(2)
    with left:
        cost_df = pd.DataFrame({
            "Component": ["Freight", "Warehouse", "Late Penalty"],
            "Cost": [b["freight_cost"], b["warehouse_cost"], b["late_penalty"]],
        })
        st.plotly_chart(px.bar(cost_df, x="Component", y="Cost", title="Baseline Cost Breakdown"), use_container_width=True)
    with right:
        if "Plant Code" in df:
            p = df.groupby("Plant Code").size().reset_index(name="Orders")
            st.plotly_chart(px.bar(p, x="Plant Code", y="Orders", title="Orders by Plant"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "Carrier" in df:
            car = df.groupby("Carrier").agg(Orders=("Carrier","size"), On_Time=("On Time","mean")).reset_index()
            car["On_Time"] *= 100
            st.plotly_chart(px.bar(car, x="Carrier", y="Orders", color="On_Time", title="Carrier Volume & On-Time %"), use_container_width=True)
    with right:
        if "Customer" in df:
            cust = df.groupby("Customer").size().nlargest(10).reset_index(name="Orders")
            st.plotly_chart(px.bar(cust, x="Orders", y="Customer", orientation="h", title="Top 10 Customers by Orders"), use_container_width=True)
