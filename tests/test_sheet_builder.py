import numpy as np
import pandas as pd

from portfolio_risk import sheet_builder


def test_workbook_has_live_formulas_over_synthetic_prices():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    prices = pd.DataFrame(100 * np.exp(rng.normal(0, 0.02, (60, 3)).cumsum(0)), index=idx, columns=["VOO", "A", "GC=F"])
    wb = sheet_builder.build_workbook({"A": 60.0, "GC=F": 40.0}, prices, {"A": "Alpha"})
    assert wb.sheetnames == ["Inputs", "Prices", "Returns", "Covariance", "Correlation", "Assets", "Portfolio", "Data quality", "Optimiser"]
    assert wb["Prices"].max_row == 61
    assert wb["Returns"]["C2"].value.startswith("=Prices!")
    assert wb["Inputs"]["D10"].value == "=C10/SUM($C$10:$C$11)"
    assert wb["Data quality"]["B3"].value == '=RIGHT(A3,2)="=F"'
