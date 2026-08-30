from tests.test_core import synthetic_data
from src.optimization.linear_programming import solve_transport_lp
from src.optimization.plant_milp import solve_plant_activation_milp
from src.optimization.min_cost_flow import solve_min_cost_flow

def test_lp_success():
    out = solve_transport_lp(synthetic_data())
    assert out["success"]
    assert out["objective"] >= 0
    assert abs(out["allocations"]["Allocated Orders"].sum() - 4) < 1e-6

def test_milp_success():
    out = solve_plant_activation_milp(synthetic_data())
    assert out["success"]
    assert set(out["activation"]["Activate"].unique()).issubset({0,1})

def test_flow_success():
    out = solve_min_cost_flow(synthetic_data())
    assert out["success"]
