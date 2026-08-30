# 🚚 Supply Chain Optimization Control Tower

A portfolio-ready Operations Research + Supply Chain Analytics project built with **Python, SciPy, NetworkX, Plotly, and Streamlit**.

The project turns the Supply Chain Logistics Problem dataset into a live decision-support application for:

- baseline cost and service analysis
- transportation LP optimization
- plant-activation MILP
- min-cost network flow
- Dijkstra-based route selection
- multi-criteria carrier scoring
- plant capacity and risk analysis
- interactive disruption scenario simulation
- data exploration and downloadable optimization outputs

## Architecture

```text
Excel workbook
    ↓
Data Loader + Validation
    ↓
Cost / KPI Engine
    ↓
LP + MILP + Min-Cost Flow + Routing + Carrier Scoring
    ↓
Optimization Pipeline
    ↓
Streamlit Control Tower
```

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The workbook must be named:

```text
data/raw/supply_chain_logistics_data.xlsx
```

## Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Dashboard Pages

1. Executive Overview
2. Cost Optimization
3. Plants & Capacity
4. Carrier Performance
5. Route Optimization
6. Scenario Simulator
7. Data Explorer

## Important modeling note

The source workbook contains operational data but does not provide every managerial parameter required for all possible optimization decisions. Any modeled assumptions (for example, a late-delivery penalty or facility fixed-cost proxy) are explicitly isolated in the code and should be replaced with organization-specific values in production.

## Streamlit deployment

Push this repository to GitHub, then deploy `app.py` from the `main` branch on Streamlit Community Cloud.

## Project structure

```text
.
├── app.py
├── app_pages/
├── src/
│   └── optimization/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── tests/
├── docs/
├── .streamlit/
├── .github/workflows/
├── requirements.txt
└── README.md
```
