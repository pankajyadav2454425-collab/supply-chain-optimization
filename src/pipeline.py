from __future__ import annotations
from .data_loader import load_supply_chain_data, validate_data
from .metrics import calculate_baseline_metrics, enrich_orders_with_costs
from .optimization.linear_programming import solve_transport_lp
from .optimization.plant_milp import solve_plant_activation_milp
from .optimization.min_cost_flow import solve_min_cost_flow
from .optimization.carrier_scoring import score_carriers

def run_pipeline(filepath):
    data = load_supply_chain_data(filepath)
    validation = validate_data(data)
    baseline = calculate_baseline_metrics(data)
    lp = solve_transport_lp(data)
    milp = solve_plant_activation_milp(data)
    flow = solve_min_cost_flow(data)
    carriers = score_carriers(data)
    enriched_orders = enrich_orders_with_costs(data)
    return {
        "data": data,
        "validation": validation,
        "baseline": baseline,
        "lp": lp,
        "milp": milp,
        "flow": flow,
        "carriers": carriers,
        "orders": enriched_orders,
    }
