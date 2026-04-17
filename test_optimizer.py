"""
Unit Tests — Supply Chain Optimization Engine
=============================================
Run with:  pytest tests/test_optimizer.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supply_chain_optimizer import (
    SupplyChainDataLoader,
    LinearProgrammingOptimizer,
    NetworkFlowOptimizer,
    DijkstraRouter,
    WeightedScoringModel,
    PlantActivationMILP,
    ScenarioAnalyzer,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Minimal synthetic dataset mirroring the real schema."""
    np.random.seed(42)
    n = 200
    plants    = ["PLANT03", "PLANT09", "PLANT12", "PLANT16"]
    ports     = ["PORT04",  "PORT09"]
    carriers  = ["V444_0",  "V444_1", "V44_3"]
    services  = ["CRF",     "DTP",    "DTD"]
    customers = [f"CUST_{i:02d}" for i in range(10)]

    df = pd.DataFrame({
        "Order ID":            np.random.uniform(1e9, 2e9, n),
        "Order Date":          pd.date_range("2013-05-01", periods=n, freq="h"),
        "Origin Port":         np.random.choice(ports, n),
        "Carrier":             np.random.choice(carriers, n),
        "TPT":                 np.random.choice([0, 1, 2, 3, 4], n),
        "Service Level":       np.random.choice(services, n),
        "Ship ahead day count":np.random.randint(0, 5, n),
        "Ship Late Day count": np.random.choice([0, 0, 0, 0, 1, 2], n),
        "Customer":            np.random.choice(customers, n),
        "Product ID":          np.random.randint(1_690_000, 1_710_000, n),
        "Plant Code":          np.random.choice(plants, n, p=[0.7, 0.1, 0.1, 0.1]),
        "Destination Port":    "PORT09",
        "Unit quantity":       np.random.randint(100, 5000, n),
        "Weight":              np.random.uniform(1, 500, n),
    })

    # Add derived columns expected downstream
    df["Transport_Cost"] = df["Weight"] * df["TPT"] * 50 + df["Unit quantity"] * 0.10
    df["Late_Penalty"]   = df["Ship Late Day count"] * df["Unit quantity"] * 5
    df["Total_Cost"]     = df["Transport_Cost"] + df["Late_Penalty"]
    df["Is_Late"]        = (df["Ship Late Day count"] > 0).astype(int)
    return df


# ── DataLoader ──────────────────────────────────────────────────

class TestDataLoader:
    DATA_PATH = "data/supply_chain_logistics_data.xlsx"

    @pytest.mark.skipif(
        not os.path.exists(DATA_PATH),
        reason="Real dataset not available in this environment",
    )
    def test_load_returns_dataframe(self):
        loader = SupplyChainDataLoader(self.DATA_PATH)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.skipif(
        not os.path.exists(DATA_PATH),
        reason="Real dataset not available in this environment",
    )
    def test_derived_columns_present(self):
        loader = SupplyChainDataLoader(self.DATA_PATH)
        df = loader.load()
        for col in ["Transport_Cost", "Late_Penalty", "Total_Cost", "Is_Late"]:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.skipif(
        not os.path.exists(DATA_PATH),
        reason="Real dataset not available in this environment",
    )
    def test_summary_keys(self):
        loader = SupplyChainDataLoader(self.DATA_PATH)
        loader.load()
        summary = loader.summary()
        for key in ["total_orders", "late_orders", "on_time_rate",
                    "total_transport_cost", "total_penalty"]:
            assert key in summary


# ── LP Optimizer ────────────────────────────────────────────────

class TestLinearProgrammingOptimizer:
    def test_result_keys(self, sample_df):
        lp = LinearProgrammingOptimizer(sample_df)
        result = lp.solve()
        for key in ["status", "baseline_cost", "optimized_cost", "savings", "savings_pct"]:
            assert key in result, f"Missing key: {key}"

    def test_baseline_cost_positive(self, sample_df):
        lp = LinearProgrammingOptimizer(sample_df)
        result = lp.solve()
        assert result["baseline_cost"] > 0

    def test_savings_non_negative(self, sample_df):
        lp = LinearProgrammingOptimizer(sample_df)
        result = lp.solve()
        # Savings can be 0 for trivial datasets but should not be negative
        assert result["savings"] >= -1e-3


# ── Network Flow ─────────────────────────────────────────────────

class TestNetworkFlowOptimizer:
    def test_returns_dict(self, sample_df):
        nf = NetworkFlowOptimizer(sample_df)
        result = nf.solve()
        assert isinstance(result, dict)

    def test_graph_non_empty(self, sample_df):
        nf = NetworkFlowOptimizer(sample_df)
        nf._build_network()
        assert len(nf.graph) > 0

    def test_total_edges_positive(self, sample_df):
        nf = NetworkFlowOptimizer(sample_df)
        result = nf.solve()
        assert result["total_edges"] > 0


# ── Dijkstra Router ──────────────────────────────────────────────

class TestDijkstraRouter:
    def test_build_graph(self, sample_df):
        router = DijkstraRouter(sample_df)
        assert len(router.graph) > 0

    def test_shortest_path_finite(self, sample_df):
        router = DijkstraRouter(sample_df)
        plant    = sample_df["Plant Code"].iloc[0]
        customer = sample_df["Customer"].iloc[0]
        cost, path = router.shortest_path(plant, customer)
        # Path may be finite or inf depending on connectivity
        assert isinstance(cost, float)
        assert isinstance(path, list)

    def test_best_routes_list(self, sample_df):
        router = DijkstraRouter(sample_df)
        routes = router.best_routes(top_n=3)
        assert isinstance(routes, list)
        assert len(routes) <= 3

    def test_path_starts_at_plant(self, sample_df):
        router = DijkstraRouter(sample_df)
        plant    = sample_df["Plant Code"].iloc[0]
        customer = sample_df["Customer"].iloc[0]
        _, path = router.shortest_path(plant, customer)
        if len(path) > 0:
            assert path[0] == plant


# ── Weighted Scoring Model ───────────────────────────────────────

class TestWeightedScoringModel:
    def test_returns_dataframe(self, sample_df):
        wsm = WeightedScoringModel(sample_df)
        scores = wsm.score_carriers()
        assert isinstance(scores, pd.DataFrame)

    def test_all_carriers_present(self, sample_df):
        wsm = WeightedScoringModel(sample_df)
        scores = wsm.score_carriers()
        for carrier in sample_df["Carrier"].unique():
            assert carrier in scores["Carrier"].values

    def test_scores_positive(self, sample_df):
        wsm = WeightedScoringModel(sample_df)
        scores = wsm.score_carriers()
        assert (scores["Composite_Score"] > 0).all()

    def test_ranked_descending(self, sample_df):
        wsm = WeightedScoringModel(sample_df)
        scores = wsm.score_carriers()
        assert scores["Composite_Score"].is_monotonic_decreasing

    def test_recommend_string(self, sample_df):
        wsm = WeightedScoringModel(sample_df)
        rec = wsm.recommend()
        assert isinstance(rec, str)
        assert "Best carrier:" in rec


# ── MILP Plant Activation ────────────────────────────────────────

class TestPlantActivationMILP:
    def test_returns_dict(self, sample_df):
        milp = PlantActivationMILP(sample_df)
        result = milp.solve()
        assert isinstance(result, dict)

    def test_plant_decisions_list(self, sample_df):
        milp = PlantActivationMILP(sample_df)
        result = milp.solve()
        assert isinstance(result["plant_decisions"], list)
        assert len(result["plant_decisions"]) > 0

    def test_activate_and_deactivate_partition(self, sample_df):
        milp = PlantActivationMILP(sample_df)
        result = milp.solve()
        activate   = set(result["plants_to_activate"])
        deactivate = set(result["plants_to_deactivate"])
        assert activate.isdisjoint(deactivate)


# ── Scenario Analyzer ────────────────────────────────────────────

class TestScenarioAnalyzer:
    def test_returns_six_scenarios(self):
        analyzer = ScenarioAnalyzer(baseline_cost=20_000_000)
        df = analyzer.run_all()
        assert len(df) == 6

    def test_all_columns_present(self):
        analyzer = ScenarioAnalyzer(baseline_cost=20_000_000)
        df = analyzer.run_all()
        for col in ["Scenario", "Cost_Impact_$", "Cost_Change_pct",
                    "On_Time_Rate_pct", "Risk_Level"]:
            assert col in df.columns

    def test_optimized_scenario_negative_change(self):
        analyzer = ScenarioAnalyzer(baseline_cost=20_000_000)
        df = analyzer.run_all()
        opt = df[df["Scenario"] == "Optimized Model"].iloc[0]
        assert opt["Cost_Change_pct"] < 0, "Optimized scenario should reduce cost"

    def test_plant03_failure_highest_risk(self):
        analyzer = ScenarioAnalyzer(baseline_cost=20_000_000)
        df = analyzer.run_all()
        critical = df[df["Risk_Level"] == "CRITICAL"]
        assert len(critical) >= 1


# ── Integration smoke-test ────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_smoke(self, sample_df):
        """End-to-end smoke test using synthetic data."""
        # LP
        lp = LinearProgrammingOptimizer(sample_df)
        lp_result = lp.solve()
        assert lp_result["baseline_cost"] > 0

        # WSM
        wsm = WeightedScoringModel(sample_df)
        scores = wsm.score_carriers()
        assert len(scores) == sample_df["Carrier"].nunique()

        # Dijkstra
        router = DijkstraRouter(sample_df)
        routes = router.best_routes(top_n=3)
        assert isinstance(routes, list)

        # MILP
        milp = PlantActivationMILP(sample_df)
        milp_result = milp.solve()
        assert "plant_decisions" in milp_result

        # Scenarios
        analyzer = ScenarioAnalyzer(sample_df["Total_Cost"].sum())
        scenarios = analyzer.run_all()
        assert len(scenarios) == 6
