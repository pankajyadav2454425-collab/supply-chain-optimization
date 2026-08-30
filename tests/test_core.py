import pandas as pd
from src.data_loader import clean_orders
from src.optimization.carrier_scoring import score_carriers

def synthetic_data():
    orders = pd.DataFrame({
        "Order ID":["O1","O2","O3","O4"],
        "Order Date":pd.to_datetime(["2024-01-01"]*4),
        "Plant Code":["P1","P1","P2","P2"],
        "Customer":["C1","C2","C1","C2"],
        "Carrier":["A","A","B","B"],
        "Origin Port":["OP1","OP1","OP2","OP2"],
        "Destination Port":["DP1","DP2","DP1","DP2"],
        "Unit quantity":[10,20,10,20],
        "Weight":[1,2,1,2],
        "TPT":[2,3,4,5],
        "Ship Late Day count":[0,0,1,2],
    })
    return {
        "orders": orders,
        "freight_rates": pd.DataFrame(),
        "plant_ports": pd.DataFrame(),
        "products_per_plant": pd.DataFrame(),
        "vmi_customers": pd.DataFrame(),
        "warehouse_capacities": pd.DataFrame(),
        "warehouse_costs": pd.DataFrame(),
    }

def test_clean_orders():
    df = clean_orders(synthetic_data()["orders"])
    assert len(df) == 4
    assert df["Weight"].sum() == 6

def test_carrier_score_range():
    scores = score_carriers(synthetic_data())
    assert len(scores) == 2
    assert scores["Carrier_Score"].between(0,100).all()
