from __future__ import annotations
from .linear_programming import solve_transport_lp
from .plant_milp import solve_plant_activation_milp
from .min_cost_flow import solve_min_cost_flow

def run_scenario(
    data: dict,
    demand_change_pct=0.0,
    freight_change_pct=0.0,
    capacity_change_pct=0.0,
    disabled_plant=None,
):
    dm = 1 + demand_change_pct/100
    cm = 1 + freight_change_pct/100
    capm = max(0, 1 + capacity_change_pct/100)

    lp = solve_transport_lp(
        data, demand_multiplier=dm, cost_multiplier=cm,
        capacity_multiplier=capm, disabled_plant=disabled_plant
    )
    milp = solve_plant_activation_milp(
        data, demand_multiplier=dm, cost_multiplier=cm,
        capacity_multiplier=capm, disabled_plant=disabled_plant
    )
    flow = solve_min_cost_flow(
        data, demand_multiplier=dm, cost_multiplier=cm,
        capacity_multiplier=capm, disabled_plant=disabled_plant
    )
    return {"lp": lp, "milp": milp, "flow": flow}
