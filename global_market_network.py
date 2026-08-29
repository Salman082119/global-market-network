"""
Global Stock Market Correlation & Contagion Network - Starter Script
======================================================================
Fetches historical data for major world indices, computes daily returns,
builds a correlation matrix, and visualizes it as a network graph.

Requirements:
    pip install yfinance pandas numpy matplotlib seaborn networkx

Usage:
    python global_market_network.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ----------------------------------------------------------------------
# 1. CONFIG: Major global indices (Yahoo Finance tickers)
# ----------------------------------------------------------------------
INDICES = {
    "S&P500 (US)": "^GSPC",
    "Nasdaq (US)": "^IXIC",
    "FTSE100 (UK)": "^FTSE",
    "DAX (Germany)": "^GDAXI",
    "CAC40 (France)": "^FCHI",
    "Nikkei225 (Japan)": "^N225",
    "HangSeng (HK)": "^HSI",
    "Shanghai (China)": "000001.SS",
    "Nifty50 (India)": "^NSEI",
    "Bovespa (Brazil)": "^BVSP",
    "TSX (Canada)": "^GSPTSE",
    "ASX200 (Australia)": "^AXJO",
}

START_DATE = "2005-01-01"
END_DATE = "2025-01-01"

# ----------------------------------------------------------------------
# 2. DATA COLLECTION
# ----------------------------------------------------------------------
def fetch_data(indices: dict, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all indices into one DataFrame."""
    all_data = {}
    for name, ticker in indices.items():
        print(f"Fetching {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Flatten MultiIndex columns (yfinance >= 0.2.x behavior)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                series = df[close_col]
                if isinstance(series, pd.DataFrame):
                    series = series.squeeze("columns")
                series.index = pd.to_datetime(series.index).tz_localize(None)
                all_data[name] = series.sort_index()
            else:
                print(f"  -> No data returned for {name}")
        except Exception as e:
            print(f"  -> Failed to fetch {name}: {e}")

    prices = pd.DataFrame(all_data)
    return prices


# ----------------------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------------------
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to daily log returns, forward-filling missing days
    (different markets have different holidays)."""
    prices = prices.ffill().dropna(how="all")
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")


# ----------------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# ----------------------------------------------------------------------
def rolling_correlation(returns: pd.DataFrame, market_a: str, market_b: str, window: int = 90):
    """Rolling correlation between two specific markets."""
    return returns[market_a].rolling(window).corr(returns[market_b])


def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Global Market Correlation Matrix"):
    corr = returns.corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("Saved correlation_heatmap.png")
    return corr


# ----------------------------------------------------------------------
# 5. NETWORK GRAPH
# ----------------------------------------------------------------------
def build_correlation_network(corr: pd.DataFrame, threshold: float = 0.4):
    """Build a graph where an edge exists if |correlation| > threshold."""
    G = nx.Graph()
    G.add_nodes_from(corr.columns)

    for i, market_a in enumerate(corr.columns):
        for market_b in corr.columns[i + 1:]:
            weight = corr.loc[market_a, market_b]
            if abs(weight) >= threshold:
                G.add_edge(market_a, market_b, weight=round(weight, 2))
    return G


def plot_network(G: nx.Graph, title: str = "Global Market Contagion Network"):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.8)

    # Node size by degree centrality (how "connected" a market is)
    centrality = nx.degree_centrality(G)
    node_sizes = [3000 * centrality[n] + 500 for n in G.nodes()]

    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("market_network.png", dpi=150)
    print("Saved market_network.png")

    # Print top 5 most "central" (systemically important) markets
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 most connected (systemically important) markets:")
    for name, score in ranked[:5]:
        print(f"  {name}: {score:.3f}")


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Step 1: Fetching data...")
    prices = fetch_data(INDICES, START_DATE, END_DATE)
    prices.to_csv("raw_prices.csv")
    print(f"Data shape: {prices.shape}\n")

    print("Step 2: Computing log returns...")
    returns = compute_log_returns(prices)
    returns.to_csv("log_returns.csv")

    print("\nStep 3: Building correlation matrix + heatmap...")
    corr = plot_correlation_heatmap(returns)

    print("\nStep 4: Building contagion network graph...")
    G = build_correlation_network(corr, threshold=0.4)
    plot_network(G)

    print("\nDone! Check raw_prices.csv, log_returns.csv, correlation_heatmap.png, market_network.png")
