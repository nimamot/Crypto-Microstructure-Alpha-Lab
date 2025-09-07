import os, math, time, pathlib, datetime as dt, json
import pandas as pd
import ccxt
import yaml

def rate_sleep(ex):
    if isinstance(ex, ccxt.NetworkError):
        time.sleep(2)
    else:
        time.sleep(0.5)

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

def iso_to_ms(s):
    return int(pd.Timestamp(s).value // 1_000_000)

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1000):
    all_rows = []
    ms = since_ms
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=ms, limit=limit)
        except Exception as e:
            rate_sleep(e)
            continue
        if not batch:
            break
        all_rows += batch
        last_ts = batch[-1][0]
        # advance one candle to avoid duplicates
        ms = last_ts + exchange.parse_timeframe(timeframe) * 1000
        if last_ts >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)
    return all_rows

def main():
    cfg = load_cfg()
    data_dir = cfg["data_dir"]
    symbols = cfg["symbols"]
    timeframe = cfg["timeframe"]
    start_ms = iso_to_ms(cfg["start"])
    end_ms = iso_to_ms(cfg["end"])

    ex = ccxt.binance({"enableRateLimit": True})
    for sym in symbols:
        rows = fetch_ohlcv_range(ex, sym, timeframe, start_ms, end_ms)
        if not rows:
            print(f"No data for {sym}")
            continue
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["symbol"] = sym.replace("/", "")
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["datetime","symbol","open","high","low","close","volume"]]
        # partition by date
        out_base = f"{data_dir}/raw/crypto/{timeframe}/{df['symbol'].iloc[0]}"
        for day, d in df.groupby(df["datetime"].dt.date):
            day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
            out_path = f"{out_base}/date={day_str}.parquet"
            ensure_dir(out_path)
            d.to_parquet(out_path, index=False)
            print("Wrote", out_path, len(d))

if __name__ == "__main__":
    main()
