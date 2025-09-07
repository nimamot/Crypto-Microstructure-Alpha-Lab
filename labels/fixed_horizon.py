import pandas as pd
import numpy as np

def add_fixed_horizon_labels(df: pd.DataFrame, horizon_minutes: int, threshold_bps: float):
    """
    df columns required: datetime (UTC), open, high, low, close, volume, symbol
    Returns a copy with:
      r_fwd  - log return over H minutes
      y_cls  - classification: 1 up, -1 down, 0 flat based on threshold
    """
    df = df.sort_values(["symbol","datetime"]).copy()
    # one row per minute, so horizon in rows equals minutes
    H = int(horizon_minutes)
    df["r_fwd"] = np.log(df["close"].shift(-H)) - np.log(df["close"])
    tau = threshold_bps * 1e-4
    cond_up = df["r_fwd"] >  tau
    cond_dn = df["r_fwd"] < -tau
    df["y_cls"] = 0
    df.loc[cond_up, "y_cls"] = 1
    df.loc[cond_dn, "y_cls"] = -1
    # drop tail where future is unknown
    df = df.iloc[:-H] if H > 0 else df
    return df
