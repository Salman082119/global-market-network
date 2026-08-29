# Global Multi-Asset Market Advisor

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

Interactive dashboard that scores **40+ global assets** (US & India & Europe & Japan indices/stocks, precious metals, FX, commodities, crypto, VIX) on a 0–100 scale and predicts **Buy / Sell / Hold** signals — with an honest backtest and a correlation/contagion "Market Context" view.

## Live Demo

🔗 **[Launch Dashboard](https://yourname.streamlit.app)**

## Features

- **📊 Buy/Sell Advisor** — 42 assets scored 0–100 by 5 technical signals (trend · RSI · MACD · volatility · price position), Top Buys / Top Sells, category & score filters
- **📈 Price & Technicals** — interactive per-asset chart with EMA20/50/200, RSI and MACD overlay
- **📜 Backtest & Honesty** — monthly momentum re-rank vs equal-weight benchmark, compounding curve, no look-ahead, clear disclaimers
- **📈 Performance** — 1M/3M/6M/1Y/YTD returns heatmap + deepest historical drawdown crash alerts
- **💼 Portfolio Builder** — risk-appetite weighted allocation (Conservative / Balanced / Aggressive) with category diversification (max 3 per category)
- **🎯 Strategy Presets** — Momentum / Balanced / Swing / Long-term / Defensive re-weighting of the 5 signals
- **🏅 Compare Mode** — rebased price overlay + correlation between any two assets
- **🧭 Market Context** — correlation & contagion view of major indices (rolling spillover, risk index live in the companion dashboard)
- **Dark theme**, data freshness indicator, CSV exports throughout

## Signal Engine (score 0–100)

| Component | Weight | What it measures |
|---|---|---|
| Trend | 25% | EMA50 vs EMA200 position |
| RSI (momentum) | 25% | 14-period RSI (30/70 band) |
| MACD | 20% | MACD line vs signal line |
| Volatility | 15% | current vol vs long-run vol (inverted) |
| Price position | 15% | where price sits in 52-week range |

**Labels:** Strong Buy 75–100 · Buy 60–74 · Hold 40–59 · Sell 25–39 · Strong Sell 0–24.
VIX is inverted (higher fear → lower score) since it is a risk asset.

## Honest Backtest

Monthly momentum strategy (re-rank all assets by 12-month momentum, hold top 25%, no look-ahead) vs equal-weight buy-and-hold:

| Metric | Value |
|---|---|
| Strategy final multiplier | ~20x |
| Equal-weight benchmark | ~10x |
| Backtest window | ~185 monthly re-ranks (2009–2024) |

> ⚠️ Not financial advice. Signals are purely technical, can be wrong, and past
> performance does not guarantee future returns.

## Quick Start

```bash
git clone https://github.com/yourname/global-market-network.git
cd global-market-network
pip install -r requirements.txt
python multi_asset_advisor.py     # fetch universe + generate signals/backtest (~3 min)
streamlit run advisor_dashboard.py  # launch the advisor dashboard
```

## Project Structure

```
├── advisor_dashboard.py          # Multi-asset advisor dashboard (main)
├── multi_asset_advisor.py        # Universe fetch + signal engine + backtest
├── dashboard.py                  # Original contagion/correlation dashboard
├── global_market_network.py      # Data pipeline + baseline visuals
├── advanced_analysis.py          # Rolling corr, communities, spillover, Granger
├── gnn_predict.py                # GATv2 graph neural network
├── advisor_signals.csv           # 42 assets, scores + signals
├── advisor_backtest.csv          # Monthly strategy vs benchmark returns
├── advisor_prices.csv            # Full price history for all assets
├── requirements.txt              # Python dependencies
└── .streamlit/config.toml        # Dark theme config
```

## Tech Stack

- **Data**: Yahoo Finance (yfinance), pandas, numpy
- **Indicators**: EMA, RSI, MACD, realized volatility (pandas)
- **Backtest**: monthly momentum re-ranking, compounding
- **Dashboard**: Streamlit, Plotly
- **Analysis**: networkx (Louvain, centrality), statsmodels (Granger, DY spillover)
- **ML**: PyTorch + PyTorch Geometric (GATv2Conv)

## Methodology

1. **Universe**: 40+ assets across 7 categories — indices, US/India/Europe/Japan stocks, precious metals, FX, commodities, crypto
2. **Scoring**: each asset scored 0–100 from 5 weighted technical components
3. **Signals**: mapped to Strong Buy / Buy / Hold / Sell / Strong Sell
4. **Backtest**: monthly re-rank by 12-month momentum, hold top 25%, no look-ahead
5. **Market Context**: rolling correlation, Diebold-Yilmaz spillover, GNN contagion prediction

## License

MIT
