# 🚚 Supply Chain Logistics Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Orders](https://img.shields.io/badge/Dataset-9%2C215%20Orders-orange)
![Savings](https://img.shields.io/badge/Cost%20Savings-33.4%25-success)

**Multi-algorithm optimization engine that reduces total supply chain cost by 33.4% and late deliveries by 80.2%.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Optimization Algorithms](#optimization-algorithms)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Scenario Analysis](#scenario-analysis)
- [Implementation Roadmap](#implementation-roadmap)
- [Contributing](#contributing)

---

## Overview

This project applies **five optimization algorithms** to a real-world supply chain logistics dataset of **9,215 shipment orders** across 7 manufacturing plants, 3 origin ports, 3 freight carriers, and 46 customers. The goal is to **minimize total supply chain cost** (transportation + late-delivery penalties) while meeting all customer demand and plant capacity constraints.

### Business Problem

> *Given 9,215 shipment orders across 7 plants, 3 ports, 3 carriers, and 3 service levels — determine the optimal assignment of orders to plants, carriers, and routes to minimize total cost while maximizing on-time delivery.*

---

## Key Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total Transportation Cost | $18.08M | $13.20M | **−27.0%** |
| Late Delivery Penalty | $2.24M | $0.34M | **−85.0%** |
| **Total Cost** | **$20.32M** | **$13.54M** | **−33.4% (−$6.78M)** |
| On-Time Delivery Rate | 97.92% | 99.60% | +1.68 pp |
| Late Orders | 192 | 38 | **−80.2%** |
| PLANT03 Concentration | 92.7% | 74.0% | Risk reduced |

### Carrier Scorecard (Weighted Scoring Model)

| Rank | Carrier | Avg TPT | Late Rate | Cost/Unit | Score |
|------|---------|---------|-----------|-----------|-------|
| 🥇 1st | V44_3 | 1.29 days | **0.00%** | $0.537 | 144,363 |
| 🥈 2nd | V444_1 | 1.05 days | 1.29% | $0.810 | 51,230 |
| 🥉 3rd | V444_0 | 2.00 days | 5.44% | $0.608 | 12,782 |

---

## Repository Structure

```
supply-chain-optimization/
│
├── 📂 src/                              # Core Python modules
│   ├── supply_chain_optimizer.py        # Main optimization engine (5 algorithms)
│   ├── eda.py                           # Exploratory Data Analysis
│   └── visualize.py                     # Charts & report figures
│
├── 📂 data/
│   └── supply_chain_logistics_data.xlsx # Raw dataset (9,215 orders)
│
├── 📂 dashboard/
│   ├── supply_chain_dashboard.html      # Interactive HTML dashboard v1
│   └── supply_chain_dashboard_v2.html   # Interactive HTML dashboard v2
│
├── 📂 reports/
│   └── supply_chain_optimization_report.docx  # Full Word report
│
├── 📂 notebooks/
│   └── supply_chain_analysis.ipynb      # Jupyter walkthrough notebook
│
├── 📂 tests/
│   └── test_optimizer.py                # Pytest unit & integration tests
│
├── 📂 docs/
│   └── figures/                         # Auto-generated charts (PNG)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Optimization Algorithms

### 1. 📐 Linear Programming (LP) — Cost Minimization
Minimizes total transportation cost subject to demand satisfaction and plant capacity constraints.

```
Minimize:  Σ C_ij · x_ij  +  Σ P_j · d_j
Subject to:
  Σ_i x_ij  ≥  D_j      (demand satisfaction)
  Σ_j x_ij  ≤  U_i      (plant capacity)
  x_ij ≥ 0
```
**Result:** Reduces transportation cost from $18.08M → $13.20M **(−27%)**

---

### 2. 🌐 Network Flow Optimization (Min-Cost Flow)
Models the supply chain as a directed capacitated network `G(V, E)`:
```
V = {Plants} ∪ {Origin Ports} ∪ {Dest Port} ∪ {Customers}
Flow balance: Σ_j f_ij  −  Σ_k f_ki  =  b_i   ∀ node i
```
Finds optimal multi-echelon routing: **Plant → Origin Port → Destination Port → Customer**

---

### 3. 🗺️ Dijkstra's Shortest Path — Route Prediction
Computes minimum-cost routes using a generalized edge weight:
```
w(i,j) = α·Cost_ij  +  β·TPT_ij·C_time  +  γ·LateRate_ij·Penalty

Weights: α=0.40 (cost), β=0.35 (time), γ=0.25 (reliability)
```
**Best predicted route:** `PLANT03 → PORT04 → V44_3 → CRF → PORT09 → Customer`  
Performance: TPT=1.34d | Late=0.00% | Cost/Unit=$0.18

---

### 4. ⚖️ Weighted Scoring Model (WSM) — Carrier Selection
Multi-criteria carrier ranking:
```
Score(carrier) = 0.35·(1/TPT) + 0.45·(1/(LateDays+ε)) + 0.20·(1/CostPerUnit)
```
→ **V44_3** wins with score **144,363** (zero late deliveries, lowest cost/unit)

---

### 5. 🔢 MILP — Plant Activation Decisions
Binary optimization for plant activation:
```
Minimize:  Σ F_i·y_i  +  Σ C_ij·x_ij
Subject to: Σ_j x_ij ≤ U_i·y_i,   y_i ∈ {0,1}
```
**Recommendation:** Activate PLANT09 + scale PLANT12/13 → reduce PLANT03 dependency 92.7% → 74%

---

## Dataset

**File:** `data/supply_chain_logistics_data.xlsx`  
**Sheet:** `OrderList` — 9,215 rows × 14 columns

| Column | Type | Description |
|--------|------|-------------|
| Order ID | Numeric | Unique order identifier |
| Order Date | DateTime | May 2013 |
| Origin Port | Categorical | PORT04, PORT05, PORT09 |
| Carrier | Categorical | V444_0, V444_1, V44_3 |
| TPT | Integer | Transit time in days (0–4) |
| Service Level | Categorical | CRF, DTP, DTD |
| Ship Late Day count | Integer | Days delivered late |
| Customer | Categorical | 46 unique customers |
| Plant Code | Categorical | 7 plants |
| Unit quantity | Integer | Units shipped |
| Weight | Float | Shipment weight (kg) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/pankajyadav2454425-collab/supply-chain-optimization.git
cd supply-chain-optimization

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
from src.supply_chain_optimizer import run_optimization_pipeline

results = run_optimization_pipeline("data/supply_chain_logistics_data.xlsx")

print(f"LP Savings:   ${results['lp']['savings']:,.0f}  ({results['lp']['savings_pct']}%)")
print(f"Best carrier: {results['carrier_scores'][0]['Carrier']}")
print(f"Best route:   {results['routes'][0]['path']}")
```

---

## Usage

### Run the full optimization pipeline
```bash
python src/supply_chain_optimizer.py data/supply_chain_logistics_data.xlsx
```

### Run EDA & generate summary statistics
```bash
python src/eda.py data/supply_chain_logistics_data.xlsx
```

### Generate all visualization figures
```bash
python src/visualize.py data/supply_chain_logistics_data.xlsx
# Figures saved to docs/figures/
```

### Run unit tests
```bash
pytest tests/test_optimizer.py -v
pytest tests/test_optimizer.py -v --cov=src --cov-report=term-missing
```

### Open interactive dashboard
Open either HTML file directly in your browser:
```
dashboard/supply_chain_dashboard.html
dashboard/supply_chain_dashboard_v2.html
```

---

## Scenario Analysis

The model was stress-tested under six disruption scenarios:

| Scenario | Cost Impact | On-Time Rate | Risk |
|----------|-------------|--------------|------|
| Demand Surge +20% | +$2.1M (+10.3%) | 95.2% | 🟡 MEDIUM |
| **PLANT03 Failure** | **+$8.4M (+41%)** | **31.0%** | 🔴 **CRITICAL** |
| V444_0 Unavailable | +$1.2M (+5.9%) | 99.4% | 🟢 LOW |
| Transport Cost +30% | +$4.2M (+20.7%) | 96.1% | 🟡 MEDIUM |
| PORT04 Congestion 40% | +$2.8M (+13.8%) | 84.0% | 🟠 HIGH |
| **Optimized Model** | **−$6.75M (−33%)** | **99.6%** | ✅ BASELINE |

> ⚠️ **Critical risk:** PLANT03 handles 92.7% of all orders. A failure would raise costs by 41% and drop on-time delivery to 31%. Plant diversification is the #1 priority.

---

## Implementation Roadmap

| Phase | Timeline | Actions | Est. Savings |
|-------|----------|---------|--------------|
| **Phase 1** | Week 1–2 | Shift 15% V444_0 → V44_3; route CRF via PLANT03→PORT04→V44_3 | ~$650K/month |
| **Phase 2** | Month 1–2 | Activate PLANT09; scale PLANT12 & PLANT13; weekly LP reoptimization | ~$1.5M/month |
| **Phase 3** | Month 3+ | Deploy Dijkstra real-time route engine; demand forecasting (SARIMA/XGBoost) | ~$3M/month |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with 🐍 Python · scipy · pandas · matplotlib
</div>
