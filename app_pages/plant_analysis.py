import streamlit as st
import pandas as pd
import plotly.express as px

def render(ctx):
    st.header("🏭 Plants & Capacity")
    df = ctx["orders"]
    plant = df.groupby("Plant Code").agg(
        Orders=("Plant Code","size"),
        Units=("Unit quantity","sum"),
        Total_Cost=("Total Cost","sum"),
        On_Time=("On Time","mean"),
    ).reset_index()
    plant["On_Time"] *= 100
    caps = ctx["lp"]["capacities"]
    plant = plant.merge(caps, left_on="Plant Code", right_on="Plant", how="left")
    plant["Utilization_%"] = (plant["Orders"] / plant["Capacity"].replace(0, pd.NA) * 100).fillna(0)
    plant["Risk"] = pd.cut(
        plant["Utilization_%"],
        [-1,50,75,90,10**9],
        labels=["Available","Moderate","High","Critical"]
    )
    st.dataframe(plant, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(plant, x="Plant Code", y="Utilization_%", color="Risk", title="Plant Utilization"), use_container_width=True)
    st.plotly_chart(px.bar(plant, x="Plant Code", y="Total_Cost", title="Cost by Plant"), use_container_width=True)
