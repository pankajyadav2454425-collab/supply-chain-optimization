import streamlit as st

def render(ctx):
    st.header("🔎 Data Explorer")
    labels = {
        "orders":"Orders",
        "freight_rates":"Freight Rates",
        "plant_ports":"Plant-Port Links",
        "products_per_plant":"Products per Plant",
        "warehouse_capacities":"Warehouse Capacities",
        "warehouse_costs":"Warehouse Costs",
        "vmi_customers":"VMI Customers",
    }
    key = st.selectbox("Table", list(labels), format_func=lambda x: labels[x])
    df = ctx["data"][key]
    st.caption(f"{len(df):,} rows × {len(df.columns):,} columns")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{key}.csv", "text/csv")
