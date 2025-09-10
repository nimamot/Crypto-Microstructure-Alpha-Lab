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

    # If labeled set does not exist, build it from raw first
    if not os.path.exists(labeled_path):
        from data.build_dataset import main as build_dataset_main
        build_dataset_main()

    df = pd.read_parquet(labeled_path)

    # Recompute labels from raw if needed or you want a different threshold
    H = int(cfg["label"]["horizon_minutes"])
    tau = float(cfg["label"]["threshold_bps"])
    if "r_fwd" not in df.columns or "y_cls" not in df.columns:
        df = add_fixed_horizon_labels(df, H, tau)

    # Add features
    df_feat = add_bar_features(df)

    # Keep rows that still have labels after horizon drop
    df_all = df_feat.merge(
        df[["datetime","symbol","r_fwd","y_cls"]],
        on=["datetime","symbol"],
        how="inner"
    ).dropna(subset=["r_fwd","y_cls"])

    # Final column order
    feature_cols = [
        "ret_1","ret_5","ret_15","vol_5","vol_20",
        "px_ma5_diff","px_ma20_diff","hl_range","oc_change",
        "vol_z_20","rsi_14","dow","minute"
    ]
    cols = ["datetime","symbol"] + feature_cols + ["r_fwd","y_cls"]

    out_dir = "data/processed/train"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{sym}_{tf}_train.parquet"
    df_all[cols].to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(df_all)} rows and {len(feature_cols)} features")

if __name__ == "__main__":
    main()
