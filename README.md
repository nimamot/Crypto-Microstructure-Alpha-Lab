# Crypto Microstructure Alpha Research Lab

A comprehensive research framework for high-frequency crypto market microstructure analysis and short-horizon price prediction. This project demonstrates advanced quantitative research methodologies, feature engineering, and machine learning techniques applied to cryptocurrency markets.

## 🎯 Research Objectives

This lab explores the predictability of cryptocurrency price movements at minute-level granularity, focusing on:

- **Market Microstructure Analysis**: Understanding price formation mechanisms in crypto markets
- **Feature Engineering**: Developing robust technical indicators for high-frequency trading
- **Label Engineering**: Creating forward-looking targets for supervised learning
- **Model Development**: Building and validating predictive models for short-horizon returns
- **Risk Management**: Incorporating realistic transaction costs and market impact

## 🔬 Research Methodology

### Data Pipeline
1. **Data Ingestion**: High-frequency OHLCV data from Binance via CCXT
2. **Label Engineering**: Forward return labeling with configurable horizons and thresholds
3. **Feature Engineering**: 13 technical indicators capturing market microstructure
4. **Training Data Preparation**: Aligned features and labels for supervised learning
5. **Model Training**: Baseline models for regression and classification tasks

### Feature Engineering
Our research employs a comprehensive set of technical indicators:

- **Return Features**: 1, 5, and 15-minute log returns
- **Volatility Measures**: Rolling standard deviation on multiple timeframes
- **Trend Indicators**: Price position relative to moving averages (5, 20 periods)
- **Momentum Proxies**: High-low range and open-close changes
- **Volume Analysis**: Z-scored volume statistics
- **Technical Indicators**: RSI (14-period) for momentum
- **Time Features**: Day-of-week and intraday minute effects

### Label Engineering
- **Forward Returns**: Log returns over configurable horizons (1+ minutes)
- **Classification Targets**: Binary/ternary classification based on return thresholds
- **Look-ahead Bias Prevention**: Strict temporal alignment between features and labels

## 📊 Dataset

- **Symbol**: BTC/USDT
- **Timeframe**: 1-minute bars
- **Period**: June-August 2025 (3 months)
- **Features**: 13 technical indicators
- **Labels**: Forward returns and classification targets
- **Size**: ~16MB processed training dataset

## 🏗️ Project Architecture

```
├── data/                    # Data processing pipeline
│   ├── download_ohlcv.py   # Exchange data ingestion
│   ├── build_dataset.py    # Labeled dataset creation
│   └── build_training.py   # ML training data preparation
├── features/               # Feature engineering
│   └── basic_bar_features.py  # Technical indicators
├── labels/                 # Label engineering
│   └── fixed_horizon.py    # Forward return labeling
├── models/                 # Model development (future)
├── backtest/              # Backtesting framework (future)
├── tests/                 # Validation and testing
│   ├── test_labels.py     # Label alignment tests
│   └── test_features.py   # Feature validation tests
└── config.yaml           # Research configuration
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Pipeline
```bash
# 1. Download raw OHLCV data
python data/download_ohlcv.py

# 2. Create labeled dataset
python data/build_dataset.py

# 3. Generate training features
python data/build_training.py
```

### Running Tests
```bash
# Validate label alignment
python -m pytest tests/test_labels.py

# Validate feature engineering
python -m pytest tests/test_features.py
```

## 🔧 Configuration

Research parameters are configurable via `config.yaml`:

```yaml
data_dir: "data"
symbols: ["BTC/USDT"]
exchange: "binance"
timeframe: "1m"
start: "2025-06-01T00:00:00Z"
end: "2025-08-31T23:59:00Z"
label:
  horizon_minutes: 1      # Forward return horizon
  threshold_bps: 2        # Classification threshold
```

## 📈 Research Results

### Feature Engineering Validation
- ✅ **No Look-ahead Bias**: Features use only past information
- ✅ **Robust Indicators**: 13 technical features with proper handling of edge cases
- ✅ **Temporal Alignment**: Features and labels properly aligned in time

### Data Quality
- ✅ **Complete Coverage**: 3 months of continuous 1-minute data
- ✅ **Clean Labels**: Forward returns with proper handling of missing values
- ✅ **Feature Completeness**: All technical indicators computed without gaps

## 🧪 Testing Framework

Our research includes comprehensive validation:

- **Label Alignment Tests**: Ensures forward returns are correctly computed
- **Feature Validation**: Verifies technical indicators are properly calculated
- **Temporal Consistency**: Prevents look-ahead bias in feature engineering
- **Data Integrity**: Validates data quality and completeness

## 🔮 Future Research Directions

- **Model Development**: Implement baseline ML models (Random Forest, XGBoost, Neural Networks)
- **Feature Expansion**: Add order book features, sentiment analysis, cross-asset signals
- **Backtesting**: Develop realistic transaction cost models
- **Live Trading**: Deploy signals via API services
- **Risk Management**: Implement position sizing and portfolio optimization

## 📚 Research Applications

This framework is suitable for:
- **Academic Research**: Market microstructure studies
- **Quantitative Finance**: Algorithmic trading strategy development
- **Machine Learning**: Time series prediction research
- **Data Science**: High-frequency data analysis techniques

## 🤝 Contributing

This is a research project focused on demonstrating quantitative finance and ML engineering skills. Contributions are welcome for:
- Additional feature engineering techniques
- Model implementations
- Backtesting frameworks
- Documentation improvements

## 📄 License

This project is for educational and research purposes. Please ensure compliance with exchange terms of service when using live data.

