# features/basic_bar_features.py
import numpy as np
import pandas as pd

def _zscore(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std(ddof=0)
    return (s - m) / sd

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_dn = dn.rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def add_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns required: datetime, symbol, open, high, low, close, volume
    Returns df with feature columns, aligned so row t uses info available by end of bar t.
    """
    df = df.sort_values(["symbol", "datetime"]).copy()

    # Ensure features output does not carry labels if present
    for c in ("r_fwd", "y_cls"):
        if c in df.columns:
            df = df.drop(columns=[c])

    # Returns
    df["ret_1"] = np.log(df["close"]).diff(1)
    df["ret_5"] = np.log(df["close"]).diff(5)
    df["ret_15"] = np.log(df["close"]).diff(15)

    # Rolling volatility on 1 min returns
    df["vol_5"] = df["ret_1"].rolling(5, min_periods=5).std(ddof=0)
    df["vol_20"] = df["ret_1"].rolling(20, min_periods=20).std(ddof=0)

    # Price position vs moving averages
    df["ma_5"] = df["close"].rolling(5, min_periods=5).mean()
    df["ma_20"] = df["close"].rolling(20, min_periods=20).mean()
    df["px_ma5_diff"] = (df["close"] - df["ma_5"]) / df["ma_5"]
    df["px_ma20_diff"] = (df["close"] - df["ma_20"]) / df["ma_20"]

    # Range and momentum proxies
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_change"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    # Volume stats
    df["vol_z_20"] = _zscore(df["volume"].astype(float), 20)

    # RSI
    df["rsi_14"] = _rsi(df["close"].astype(float), 14)

    # Time features
    dt = df["datetime"].dt
    df["dow"] = dt.dayofweek.astype("int8")
    df["minute"] = (dt.hour * 60 + dt.minute).astype("int16")

    # Drop rows with any NaN in key features
    feature_cols = [
        "ret_1","ret_5","ret_15","vol_5","vol_20",
        "px_ma5_diff","px_ma20_diff","hl_range","oc_change",
        "vol_z_20","rsi_14","dow","minute"
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df
