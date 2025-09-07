import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


import glob, os
import pandas as pd
import yaml
from labels.fixed_horizon import add_fixed_horizon_labels

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_raw_parquet(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    dfs = [pd.read_parquet(f) for f in sorted(files)]
    return pd.concat(dfs, ignore_index=True)

def main():
    cfg = load_cfg()
    tf = cfg["timeframe"]
    sym = cfg["symbols"][0].replace("/", "")
    raw_pattern = f"data/raw/crypto/{tf}/{sym}/date=*.parquet"
    df = read_raw_parquet([raw_pattern])
    df = add_fixed_horizon_labels(df, cfg["label"]["horizon_minutes"], cfg["label"]["threshold_bps"])
    out_path = f"data/processed/{sym}_{tf}_labeled.parquet"
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print("Wrote", out_path, len(df), "rows")

if __name__ == "__main__":
    main()
