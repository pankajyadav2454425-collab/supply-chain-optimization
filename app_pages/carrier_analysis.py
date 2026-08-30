import streamlit as st
import plotly.express as px
from src.optimization.carrier_scoring import score_carriers

def render(ctx):
    st.header("🚚 Carrier Performance")
    c1,c2,c3 = st.columns(3)
    cw = c1.slider("Cost Weight", 0, 100, 20)
    sw = c2.slider("Speed Weight", 0, 100, 35)
    rw = c3.slider("Reliability Weight", 0, 100, 45)
    scores = score_carriers(ctx["data"], cw, sw, rw)
    if len(scores):
        st.metric("Recommended Carrier", str(scores.iloc[0]["Carrier"]))
    st.dataframe(scores, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(scores, x="Carrier", y="Carrier_Score", title="Carrier Ranking"), use_container_width=True)
