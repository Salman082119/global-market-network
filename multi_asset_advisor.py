"""
Multi-Asset Global Market Advisor (Phase A - Core)
===================================================
Fetches a diversified global universe (indices, global stocks, precious metals,
currencies, commodities, crypto/risk) and scores every asset 0-100 using five
technical signals (trend, momentum/RSI, MACD, volatility regime, price position).

Score -> label:
    75-100  Strong Buy
    60-74   Buy
    40-59   Hold
    25-39   Sell
    0-24    Strong Sell

Also runs an honest backtest: re-ranks each month and measures how a
momentum-following strategy would have done vs everything (buy last).
No guarantees - signals are technical, not financial advice.

Outputs:
    advisor_signals.csv         (per-asset current score, label, signal breakdown)
    advisor_backtest.csv        (monthly strategy vs benchmark returns)
    advisor_prices.csv          (merged close prices for the universe)

Usage:
    python multi_asset_advisor.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

START = "2007-01-01"      # after 2006 data so indicators have warm-up
END = "2025-01-01"

# ------------------------------------------------------------------ universe
UNIVERSE = {
    # ---- Global indices / country ETFs ----
    "🇺🇸 S&P500": "^GSPC", "🇺🇸 NASDAQ100": "^NDX",
    "🇺🇸 Dow": "^DJI", "🇮🇳 NIFTY50": "^NSEI", "🇮🇳 SENSEX": "^BSESN",
    "🇬🇧 FTSE100": "^FTSE", "🇩🇪 DAX": "^GDAXI", "🇫🇷 CAC40": "^FCHI",
    "🇯🇵 Nikkei225": "^N225", "🇭🇰 HangSeng": "^HSI",
    "🇨🇳 Shanghai": "000001.SS", "🇧🇷 Bovespa": "^BVSP",
    "🇨🇦 TSX": "^GSPTSE", "🇦🇺 ASX200": "^AXJO",

    # ---- Global individual stocks ----
    "🍎 Apple": "AAPL", "🚀 Tesla": "TSLA", "🖥️ Microsoft": "MSFT",
    "🔍 Google": "GOOGL", "🛒 Amazon": "AMZN", "📱 Meta": "META",
    "🧠 NVIDIA": "NVDA", "🏦 JP Morgan": "JPM", "🥤 Coca-Cola": "KO",
    "⚡ Reliance": "RELIANCE.NS", "💻 TCS": "TCS.NS", "🏦 HDFC Bank": "HDFCBANK.NS",
    "🚗 Maruti": "MARUTI.NS", "☕ Asian Paints": "ASIANPAINT.NS",

    # ---- Precious metals ----
    "🥇 Gold": "GC=F", "🥈 Silver": "SI=F", "🗻 Platinum": "PL=F",

    # ---- Currencies (FX) ----
    "💱 EUR/USD": "EURUSD=X", "💱 GBP/USD": "GBPUSD=X",
    "💱 USD/JPY": "JPY=X", "💱 USD/INR": "INR=X", "💱 DXY Dollar": "DX-Y.NYB",

    # ---- Commodities ----
    "🛢️ Crude WTI": "CL=F", "🛢️ Brent": "BZ=F", "🟠 Copper": "HG=F",

    # ---- Risk / crypto ----
    "😨 VIX": "^VIX", "₿ Bitcoin": "BTC-USD", "Ξ Ethereum": "ETH-USD",
}

CATEGORY = {
    "🇺🇸 S&P500": "Indices", "🇺🇸 NASDAQ100": "Indices", "🇺🇸 Dow": "Indices",
    "🇮🇳 NIFTY50": "Indices", "🇮🇳 SENSEX": "Indices", "🇬🇧 FTSE100": "Indices",
    "🇩🇪 DAX": "Indices", "🇫🇷 CAC40": "Indices", "🇯🇵 Nikkei225": "Indices",
    "🇭🇰 HangSeng": "Indices", "🇨🇳 Shanghai": "Indices", "🇧🇷 Bovespa": "Indices",
    "🇨🇦 TSX": "Indices", "🇦🇺 ASX200": "Indices",
    "🍎 Apple": "Stocks", "🚀 Tesla": "Stocks", "🖥️ Microsoft": "Stocks",
    "🔍 Google": "Stocks", "🛒 Amazon": "Stocks", "📱 Meta": "Stocks",
    "🧠 NVIDIA": "Stocks", "🏦 JP Morgan": "Stocks", "🥤 Coca-Cola": "Stocks",
    "⚡ Reliance": "Stocks", "💻 TCS": "Stocks", "🏦 HDFC Bank": "Stocks",
    "🚗 Maruti": "Stocks", "☕ Asian Paints": "Stocks",
    "🥇 Gold": "Metals", "🥈 Silver": "Metals", "🗻 Platinum": "Metals",
    "💱 EUR/USD": "Currency", "💱 GBP/USD": "Currency", "💱 USD/JPY": "Currency",
    "💱 USD/INR": "Currency", "💱 DXY Dollar": "Currency",
    "🛢️ Crude WTI": "Commodity", "🛢️ Brent": "Commodity", "🟠 Copper": "Commodity",
    "😨 VIX": "Risk", "₿ Bitcoin": "Crypto", "Ξ Ethereum": "Crypto",
}

# VIX is inverted (higher = fear = bad for risk)
INVERTED = {"😨 VIX"}

# ---------------------------------------------------------------- data fetch
def fetch_prices():
    data = {}
    for name, ticker in UNIVERSE.items():
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            col = col if col in df.columns else df.columns[0]
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.squeeze("columns")
            s.index = pd.to_datetime(s.index).tz_localize(None)
            data[name] = s.sort_index()
        except Exception as e:
            print(f"  skip {name}: {e}")
    prices = pd.DataFrame(data).ffill().dropna(how="all")
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    return prices


def daily_returns(prices):
    return prices.pct_change().dropna(how="all")


# ---------------------------------------------------------------- indicators
def score_asset(close: pd.Series) -> dict:
    """Return signal breakdown + 0-100 composite score for one asset."""
    c = close.dropna()
    if len(c) < 60:
        return {"trend": 0, "rsi": 0, "macd": 0, "vol": 0, "pos": 0, "score": 0,
                "label": "Hold", "why": "insufficient data"}

    price = float(c.iloc[-1])

    # 1. Trend - EMA50 vs EMA200
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = c.ewm(span=200, adjust=False).mean().iloc[-1] if len(c) > 200 else ema50
    trend = 100 if price > ema50 > ema200 else (75 if price > ema50 else
            (40 if price > ema200 else 0))

    # 2. Momentum / RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss and loss > 0 else 100
    rsi = 100 - 100 / (1 + rs)
    if np.isnan(rsi):
        rsi = 50
    # RSI: below 30 oversold (buy zone), above 70 overbought (caution/sell zone)
    if rsi < 30:
        rsi_score = 85
    elif rsi < 45:
        rsi_score = 65
    elif rsi <= 60:
        rsi_score = 50
    elif rsi <= 70:
        rsi_score = 35
    else:
        rsi_score = 20

    # 3. MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = (macd.iloc[-1] - signal.iloc[-1])
    macd_score = 90 if macd_hist > 0 and macd_hist > macd.diff().iloc[-1] else (
                 65 if macd_hist > 0 else 35)

    # 4. Volatility regime (vs own history)
    ret = c.pct_change().dropna()
    vol = ret.rolling(21).std().iloc[-1]
    vol_avg = ret.rolling(250).std().mean() if len(ret) > 250 else vol
    vol_ratio = vol / vol_avg if vol_avg else 1
    # rising volatility is a caution flag for "buy" (uncertainty):
    vol_score = 60 if vol_ratio < 1.2 else (45 if vol_ratio < 1.6 else 25)

    # 5. Price position (distance from 52-week high/low)
    hi = c.tail(252).max()
    lo = c.tail(252).min()
    pos = (price - lo) / (hi - lo) if (hi - lo) else 0.5
    pos_score = pos * 100

    # Composite (equal weights) - flip for inverted assets like VIX
    raw = 0.22 * trend + 0.22 * rsi_score + 0.20 * macd_score + \
          0.18 * vol_score + 0.18 * pos_score
    score = raw

    label = ("Strong Buy" if score >= 75 else "Buy" if score >= 60 else
             "Hold" if score >= 40 else "Sell" if score >= 25 else "Strong Sell")
    return {"trend": trend, "rsi": rsi_score, "macd": macd_score,
            "vol": vol_score, "pos": pos_score, "score": round(float(score), 1),
            "label": label,
            "price": round(price, 4),
            "rsi_raw": round(float(rsi), 1),
            "vol_ratio": round(float(vol_ratio), 2)}


# ---------------------------------------------------------------- backtest
def backtest(prices, months_back=60):
    """Monthly re-rank momentum strategy vs benchmark (equal-weight all)."""
    monthly = prices.resample("ME").last().dropna(how="all")
    cols = monthly.columns
    strat, bench = [], []
    for i in range(30, len(monthly) - 1):
        mom = monthly.iloc[i] / monthly.iloc[i - 12] - 1  # 12m momentum
        n = max(3, len(cols)//4)
        top = mom.nlargest(n).index
        ret_next = monthly.iloc[i + 1] / monthly.iloc[i] - 1
        strat.append(ret_next[top].mean())
        bench.append(ret_next.mean())
    return pd.DataFrame({"date": monthly.index[31:len(monthly)],
                         "strategy": strat, "benchmark": bench})


# ---------------------------------------------------------------- main
def main():
    print("Fetching universe...")
    prices = fetch_prices()
    prices.to_csv("advisor_prices.csv")
    print(f"Universe: {prices.shape[1]} assets, {prices.shape[0]} days\n")

    rows = []
    for name in prices.columns:
        b = score_asset(prices[name])
        rows.append({"asset": name, "category": CATEGORY.get(name, "Other"),
                     **b})
    sig = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    sig.to_csv("advisor_signals.csv", index=False)
    print(pd.DataFrame({
        "asset": [a.split(" ", 1)[1] if " " in a else a for a in sig["asset"]],
        "cat": sig["category"], "score": sig["score"], "signal": sig["label"]}).
        to_string(index=False))
    print(f"\nSaved advisor_signals.csv ({len(sig)} assets)")

    bt = backtest(prices)
    bt.to_csv("advisor_backtest.csv", index=False)
    if len(bt):
        grow = (1 + bt["strategy"]).cumprod()
        gbench = (1 + bt["benchmark"]).cumprod()
        print(f"\nBacktest ({len(bt)} monthly re-ranks):")
        print(f"  Momentum-strategy final multiplier: {grow.iloc[-1]:.2f}x")
        print(f"  Equal-weight benchmark multiplier:  {gbench.iloc[-1]:.2f}x")
    print("\nDone.")


if __name__ == "__main__":
    main()
