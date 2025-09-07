import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from labels.fixed_horizon import add_fixed_horizon_labels

def test_forward_return_alignment():
    dt = pd.date_range("2025-06-01", periods=5, freq="1min", tz="UTC")
    close = [100, 101, 102, 103, 104]
    df = pd.DataFrame({
        "datetime": dt,
        "symbol": ["BTCUSDT"]*5,
        "open": close, "high": close, "low": close, "close": close, "volume": 1
    })
    out = add_fixed_horizon_labels(df, horizon_minutes=1, threshold_bps=0)
    assert abs(out["r_fwd"].iloc[0] - (np.log(101) - np.log(100))) < 1e-9
    assert len(out) == 4
