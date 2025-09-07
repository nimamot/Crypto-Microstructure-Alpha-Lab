# Crypto Microstructure Alpha Lab

A research and demo stack for short horizon price prediction on crypto markets using limit order book data. You will:
1) ingest exchange data
2) create forward return labels
3) train baseline models
4) backtest with realistic costs
5) surface live signals via small services and a dashboard

This project is designed to showcase both quantitative research and ML engineering skills. Everything is reproducible and config driven.

---

## Features

- Minute bars ingestion from Binance via CCXT
- Configurable labeler for H minute forward log returns with optional classification threshold
- Parquet storage partitioned by date for fast reads
- Clean repo structure for data, labels, models, backtests, and services
- Tests for label alignment
- Plots and quick EDA notebooks

---

## Project Structure

