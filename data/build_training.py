# data/build_training.py
import os
import yaml
import pandas as pd
from features.basic_bar_features import add_bar_features
from labels.fixed_horizon import add_fixed_horizon_labels

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    tf = cfg["timeframe"]
    sym = cfg["symbols"][0].replace("/", "")
    labeled_path = f"data/processed/{sym}_{tf}_labeled.parquet"

    # Build labeled dataset if missing
    if not os.path.exists(labeled_path):
        from data.build_dataset import main as build_dataset_main
        build_dataset_main()

    # Load labeled bars
    df = pd.read_parquet(labeled_path)

    # Ensure labels exist for current config
    H = int(cfg["label"]["horizon_minutes"])
    tau = float(cfg["label"]["threshold_bps"])
    if ("r_fwd" not in df.columns) or ("y_cls" not in df.columns):
        df = add_fixed_horizon_labels(df, H, tau)

    # Compute features on bar data
    df_feat = add_bar_features(df)

    # Define the feature set we want to keep from df_feat
    feature_cols = [
        "ret_1","ret_5","ret_15","vol_5","vol_20",
        "px_ma5_diff","px_ma20_diff","hl_range","oc_change",
        "vol_z_20","rsi_14","dow","minute"
    ]

    # Keys for joining
    keys = ["datetime", "symbol"]

    # Left side: only the features to avoid column overlap with labels
    lhs = df_feat.set_index(keys)[feature_cols]

    # Right side: labels only
    rhs = df.set_index(keys)[["r_fwd", "y_cls"]]

    # Inner join on index, then reset
    df_all = lhs.join(rhs, how="inner").reset_index()

    # Final selection and NA drop
    keep_cols = ["datetime","symbol"] + feature_cols + ["r_fwd","y_cls"]
    df_all = df_all.dropna(subset=feature_cols + ["r_fwd","y_cls"])[keep_cols]

    # Save
    out_dir = "data/processed/train"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{sym}_{tf}_train.parquet"
    df_all.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(df_all)} rows and {len(feature_cols)} features")

if __name__ == "__main__":
    main()
