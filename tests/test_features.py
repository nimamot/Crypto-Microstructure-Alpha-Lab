import numpy as np
import pandas as pd
from features.basic_bar_features import add_bar_features
from labels.fixed_horizon import add_fixed_horizon_labels

def _toy_df(n=60, start="2025-06-01T00:00:00Z"):
    dt = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    # synthetic walk with noise
    px = 100 * np.exp(np.cumsum(np.random.normal(0, 0.0005, size=n)))
    high = px * (1 + np.random.uniform(0, 0.001, size=n))
    low = px * (1 - np.random.uniform(0, 0.001, size=n))
    open_ = px * (1 + np.random.uniform(-0.0005, 0.0005, size=n))
    vol = np.random.randint(10, 1000, size=n)
    return pd.DataFrame({
        "datetime": dt,
        "symbol": ["BTCUSDT"]*n,
        "open": open_, "high": high, "low": low, "close": px, "volume": vol
    })

def test_feature_shapes_and_nans():
    df = _toy_df(120)
    df_feat = add_bar_features(df)
    assert {"datetime","symbol"}.issubset(df_feat.columns)
    key_feats = ["ret_1","ret_5","ret_15","vol_5","vol_20","px_ma5_diff","px_ma20_diff",
                 "hl_range","oc_change","vol_z_20","rsi_14","dow","minute"]
    for c in key_feats:
        assert c in df_feat.columns
        assert df_feat[c].isna().sum() == 0

def test_no_lookahead_in_features():
    df = _toy_df(60)
    df_feat = add_bar_features(df)
    # Features must not use future info, so the earliest index after rolling windows should be >= the largest window size
    earliest = df_feat["datetime"].min()
    # For window 20 features we expect at least first 19 rows dropped
    assert earliest >= df["datetime"].iloc[19]

def test_labels_alignment_with_features():
    df = _toy_df(120)
    df = add_fixed_horizon_labels(df, horizon_minutes=1, threshold_bps=0)
    df_feat = add_bar_features(df)
    merged = df_feat.merge(df[["datetime","symbol","r_fwd","y_cls"]], on=["datetime","symbol"])
    assert len(merged) > 0
    assert merged["r_fwd"].isna().sum() == 0
    assert merged["y_cls"].isin([-1,0,1]).all()
