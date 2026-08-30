import streamlit as st
from src.optimization.dijkstra import shortest_route

def render(ctx):
    st.header("🗺️ Route Optimization")
    df = ctx["orders"]
    plants = sorted(df["Plant Code"].astype(str).unique())
    customers = sorted(df["Customer"].astype(str).unique())
    p = st.selectbox("Plant", plants)
    c = st.selectbox("Customer", customers)
    obj = st.selectbox("Objective", ["balanced","lowest_cost","fastest","most_reliable"])
    if st.button("Find Best Route"):
        out = shortest_route(ctx["data"], p, c, obj)
        if out["success"]:
            st.success(" → ".join(out["path"]))
            st.metric("Route Score", f"{out['score']:.4f}")
        else:
            st.warning(out["message"])
