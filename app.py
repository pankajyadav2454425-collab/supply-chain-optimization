from pathlib import Path
import streamlit as st

from src.pipeline import run_pipeline
from app_pages import (
    executive_overview,
    cost_optimization,
    plant_analysis,
    carrier_analysis,
    route_network,
    scenario_simulator,
    data_explorer,
)

st.set_page_config(
    page_title="Supply Chain Optimization Control Tower",
    page_icon="🚚",
    layout="wide",
)

DATA_PATH = Path("data/raw/supply_chain_logistics_data.xlsx")

@st.cache_data(show_spinner=False)
def load_context(path_str):
    return run_pipeline(path_str)

st.sidebar.title("🚚 Control Tower")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Cost Optimization",
        "Plants & Capacity",
        "Carrier Performance",
        "Route Optimization",
        "Scenario Simulator",
        "Data Explorer",
    ],
)

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Put your Excel file at "
        "`data/raw/supply_chain_logistics_data.xlsx`."
    )
    st.stop()

with st.spinner("Loading data and solving baseline optimization..."):
    ctx = load_context(str(DATA_PATH))

validation = ctx["validation"]
if not validation["is_valid"]:
    st.warning(f"Data validation issues: {validation}")

pages = {
    "Executive Overview": executive_overview.render,
    "Cost Optimization": cost_optimization.render,
    "Plants & Capacity": plant_analysis.render,
    "Carrier Performance": carrier_analysis.render,
    "Route Optimization": route_network.render,
    "Scenario Simulator": scenario_simulator.render,
    "Data Explorer": data_explorer.render,
}
pages[page](ctx)
