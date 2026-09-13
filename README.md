# 🌍 Global Multi-Asset Market Advisor

> An interactive Data Science platform that analyzes **40+ global assets** across equities, indices, metals, currencies, commodities, crypto and market-risk indicators using technical signals, historical backtesting and market-network analysis.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

## 📌 Overview

The **Global Multi-Asset Market Advisor** is a research-oriented market analytics project designed to turn raw financial time-series data into interpretable signals and portfolio insights.

The system combines:

- Technical-indicator-based asset scoring
- Momentum strategy backtesting
- Interactive market visualization
- Cross-asset correlation and contagion analysis
- Portfolio allocation presets
- Graph-based market analysis / GNN experimentation

The project is intended for **educational and research purposes**. It is not financial advice.

---

## 🎯 Problem Statement

Global markets contain thousands of assets whose prices, volatility and relationships change over time. Manually comparing multiple markets can be difficult and inconsistent.

This project explores how a reproducible data pipeline can:

1. Collect historical market data from multiple asset classes.
2. Transform price data into technical indicators.
3. Combine multiple signals into an interpretable 0–100 score.
4. Evaluate a simple momentum strategy using historical data.
5. Visualize market relationships and potential risk transmission.
6. Present the analysis through an interactive dashboard.

---

## ✨ Key Features

### 📊 Multi-Asset Advisor

Scores 40+ assets using five technical components:

| Signal | Weight | Description |
|---|---:|---|
| Trend | 22% | Price and EMA relationship |
| RSI / Momentum | 22% | Relative-strength and momentum condition |
| MACD | 20% | MACD vs signal-line relationship |
| Volatility | 18% | Current volatility relative to historical regime |
| Price Position | 18% | Position inside the 52-week range |

Scores are mapped to:

- **75–100:** Strong Buy
- **60–74:** Buy
- **40–59:** Hold
- **25–39:** Sell
- **0–24:** Strong Sell

### 📈 Price & Technical Analysis

Interactive charts provide:

- Price history
- EMA 20 / 50 / 200
- RSI
- MACD and signal line
- Configurable historical viewing windows

### 📜 Backtesting

A monthly momentum strategy ranks assets using their previous 12-month momentum and holds the top 25% for the following month.

The implementation is designed to avoid look-ahead bias by making each month's selection using information available before the subsequent holding period.

> **Important:** Historical backtest performance does not guarantee future results.

### 💼 Portfolio Builder

Provides risk-appetite presets such as:

- Conservative
- Balanced
- Aggressive

The allocation logic also considers diversification across asset categories.

### 🧭 Market Context

The project also explores cross-asset relationships using:

- Rolling correlations
- Network analysis
- Centrality / community analysis
- Diebold–Yilmaz spillover analysis
- Granger-causality analysis
- Graph Neural Network experimentation

---

## 🌐 Asset Universe

The current universe covers multiple categories:

- 🇺🇸🇮🇳🇪🇺🇯🇵 Global indices
- 📈 US and Indian equities
- 🥇 Precious metals
- 💱 Foreign exchange
- 🛢️ Commodities
- ₿ Crypto assets
- 😨 Market-risk indicators such as VIX

Market data is retrieved using **Yahoo Finance through `yfinance`**.

---

## 🏗️ Project Architecture

```text
Yahoo Finance / yfinance
          │
          ▼
   Historical Price Data
          │
          ▼
   Data Cleaning & Alignment
          │
          ├───────────────┐
          ▼               ▼
 Technical Indicators   Market Returns
          │               │
          ▼               ▼
    Signal Engine      Backtesting
          │               │
          └───────┬───────┘
                  ▼
        CSV Research Outputs
                  │
          ┌───────┴────────┐
          ▼                ▼
   Streamlit Advisor   Market Network
       Dashboard          Analysis
          │                │
          ▼                ▼
   Interactive UI       GNN / Spillover
```

---

## 📁 Project Structure

```text
├── advisor_dashboard.py          # Main Streamlit advisor dashboard
├── multi_asset_advisor.py        # Data pipeline, scoring and backtesting
├── dashboard.py                  # Market correlation / risk dashboard
├── global_market_network.py      # Core market-network pipeline
├── advanced_analysis.py          # Correlation, communities and spillover analysis
├── gnn_predict.py                # Graph neural network experimentation
├── advisor_signals.csv           # Generated asset signals
├── advisor_backtest.csv          # Generated backtest results
├── advisor_prices.csv            # Generated historical price data
├── requirements.txt              # Python dependencies
└── .streamlit/                   # Streamlit configuration
```

---

## 🧰 Tech Stack

**Programming & Data**

- Python
- Pandas
- NumPy
- yfinance

**Visualization & Dashboard**

- Streamlit
- Plotly
- Matplotlib

**Statistics & Network Analysis**

- Statsmodels
- NetworkX

**Machine Learning**

- PyTorch
- PyTorch Geometric
- GATv2-based graph experimentation

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Salman082119/global-market-network.git
cd global-market-network
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate market data and signals

```bash
python multi_asset_advisor.py
```

This generates the research CSV files used by the dashboard.

### 4. Launch the dashboard

```bash
streamlit run advisor_dashboard.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 📊 Methodology

### Step 1 — Data Collection

Historical market prices are collected for a diversified global universe using `yfinance`.

### Step 2 — Data Preparation

Price series are aligned, sorted, cleaned and forward-filled where appropriate before downstream calculations.

### Step 3 — Feature / Signal Engineering

The system calculates trend, RSI, MACD, volatility and 52-week price-position features.

### Step 4 — Composite Scoring

The individual components are combined into a weighted 0–100 score and mapped to an interpretable signal label.

### Step 5 — Backtesting

Assets are re-ranked monthly using 12-month momentum. The highest-ranked quarter is held for the following month and compared with an equal-weight benchmark.

### Step 6 — Market Relationship Analysis

The broader project investigates how assets move together using correlation networks, spillover statistics and graph-based machine learning.

---

## 📈 Backtest Interpretation

The backtest is included to evaluate the research methodology rather than to claim a guaranteed trading advantage.

When interpreting results, consider:

- Transaction costs
- Slippage
- Liquidity
- Taxes
- Survivorship bias
- Parameter sensitivity
- Regime changes
- Data-source limitations

**Backtested performance should not be treated as a prediction of future returns.**

---

## 🖥️ Screenshots

> Add dashboard screenshots here after deployment. Recommended screenshots:
>
> 1. Advisor overview / ranked signals
> 2. Price + technical indicators
> 3. Backtest results
> 4. Market correlation / contagion view
> 5. Portfolio builder

```text
screenshots/
├── advisor.png
├── technicals.png
├── backtest.png
├── market-context.png
└── portfolio.png
```

---

## 🔮 Future Improvements

Planned directions include:

- 🤖 AI-powered market explanation layer
- 🧠 Improved GNN-based contagion prediction
- 📊 Walk-forward and multi-period validation
- 🧪 Hyperparameter / weight sensitivity analysis
- 📉 Risk-adjusted metrics such as Sharpe and Sortino ratios
- 💾 Automated data-refresh pipeline
- 🌐 Production deployment with a public live demo
- 🔔 Optional alerts for major signal changes
- 🧠 Natural-language portfolio and market summaries

### AI Recommendation Layer — Planned

A future version can add an AI layer that explains **why** an asset received a particular signal instead of simply displaying Buy / Hold / Sell.

For example:

```text
User: Why is NVIDIA currently rated Buy?

AI:
- Trend score is strong because price is above EMA50 and EMA200.
- MACD is positive.
- RSI indicates momentum without being deeply oversold.
- Current volatility is elevated, which reduces the overall score.
- Overall composite score: 68/100 → Buy.
```

This would make the project more useful as an **AI + Data Science research application** while keeping the underlying quantitative calculations transparent.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The signals, backtests and portfolio suggestions are generated from quantitative rules and historical data. They may be incorrect, incomplete or unsuitable for real-world investment decisions.

**This project does not provide financial advice.**

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Salman Shaikh**

B.Tech Computer Science — Data Science & Artificial Intelligence

GitHub: https://github.com/Salman082119
