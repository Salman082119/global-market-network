"""
Global Market Network - Optional Deepenings (Phase 7-9)
========================================================
Optional analyses mentioned in the roadmap:

  A) Extended Universe   - add FX (DXY, EUR, JPY, GBP), commodities
                           (WTI oil, Brent, gold, silver, copper), and
                           risk assets (VIX, Bitcoin) on top of the 12 indices.
  B) Tail-Dependence Network - empirical lower/upper tail dependence per pair
                           plus an EVT (peaks-over-threshold) tail index for
                           each market, visualised as a directed stress network.

Reads log_returns.csv (12 indices). Extended assets are fetched live from
Yahoo Finance (yfinance) then merged onto the same calendar.

Outputs:
    phaseB_tail_dependence.csv        (symmetric tail-dependence matrix)
    phaseB_tail_dependence_network.png
    phaseB_evt_tail_index.csv         (per-market tail index xi)
    extended_assets_returns.csv       (optional; extended universe returns)

Usage:
    python optional_analysis.py
"""

import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

CRISIS_WINDOWS = {
    "GFC (Sep08-Mar09)": ("2008-09-01", "2009-03-31"),
    "China deval. (Aug15-Feb16)": ("2015-08-10", "2016-02-11"),
    "COVID (Feb19-Mar23 '20)": ("2020-02-19", "2020-03-23"),
    "Rate-hike bear (Jan-Oct22)": ("2022-01-03", "2022-10-12"),
}
SPAND_COLORS = ["red", "purple", "orange", "brown"]

EXTENDED_TICKERS = {
    "US Dollar (DXY)": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "WTI Crude": "CL=F",
    "Brent Crude": "BZ=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "VIX Fear": "^VIX",
    "Bitcoin": "BTC-USD",
}

FULL_UNIVERSE_IDX = {
    "S&P500 (US)": "^GSPC", "Nasdaq (US)": "^IXIC", "FTSE100 (UK)": "^FTSE",
    "DAX (Germany)": "^GDAXI", "CAC40 (France)": "^FCHI", "Nikkei225 (Japan)": "^N225",
    "HangSeng (HK)": "^HSI", "Shanghai (China)": "000001.SS",
    "Nifty50 (India)": "^NSEI", "Bovespa (Brazil)": "^BVSP",
    "TSX (Canada)": "^GSPTSE", "ASX200 (Australia)": "^AXJO",
}

# ---------------------------------------------------------------- helpers
def load_returns(path="log_returns.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def to_pct(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return s.rename(series.name)


# ---------------------------------------------------------------- SECTION A
def fetch_extended(start="2005-01-01", end="2025-01-01"):
    import yfinance as yf
    frames = {}
    for name, ticker in EXTENDED_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                close_col = close_col if close_col in df.columns else df.columns[0]
                s = df[close_col]
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze("columns")
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[name] = s.sort_index()
        except Exception as e:
            print(f"  -> failed {name}: {e}")
    return pd.DataFrame(frames)


def section_a_extended_universe(base_returns, extended_prices):
    """Merge extended assets, compute returns, report correlation to core."""
    print("\n=== SECTION A: Extended Universe ===")
    if extended_prices is None or extended_prices.empty:
        print("No extended data fetched; skipping.")
        return None

    ext_ret = np.log(extended_prices / extended_prices.shift(1)).dropna(how="all")
    combined = pd.concat([base_returns, ext_ret], axis=1)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    # Keep only the extended columns for a compact correlation vs S&P500
    core = base_returns
    sp = core["S&P500 (US)"]
    mask = ~core.index.isin([])
    ext_corr = pd.DataFrame({
        "corr_vs_S&P500": ext_ret.apply(lambda c: sp.reindex(core.index).corr(c))
    })
    ext_corr = ext_corr.sort_values("corr_vs_S&P500", ascending=False)
    print("Extended assets correlation vs S&P500 (log returns):")
    print(ext_corr.round(3).to_string())

    # Correlation matrix across the whole merged universe (core + extended)
    full_corr = combined.corr()
    full_corr.to_csv("phaseA_extended_universe_corr.csv")
    print("\nSaved phaseA_extended_universe_corr.csv "
          f"({full_corr.shape[0]}x{full_corr.shape[1]})")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(full_corr, annot=False, cmap="RdBu_r", center=0,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title("Correlation matrix: 12 indices + FX / commodities / risk assets")
    plt.tight_layout()
    plt.savefig("phaseA_extended_universe_heatmap.png", dpi=150)
    print("Saved phaseA_extended_universe_heatmap.png")
    combined.to_csv("extended_assets_returns.csv")
    return ext_corr


# ---------------------------------------------------------------- SECTION B
def kendall_tau(x, y):
    """Kendall's tau with numpy (no scipy dependency)."""
    n = len(x)
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx * sy > 0:
                concord += 1
            elif sx * sy < 0:
                discord += 1
    total = n * (n - 1) / 2
    return (concord - discord) / total


def tail_dependence(x, y, q=0.1, upper=True):
    """Empirical lower/upper tail dependence coefficient (lambda)."""
    n = len(x)
    if upper:
        qx = np.quantile(x, 1 - q)
        qy = np.quantile(y, 1 - q)
        both = np.sum((x > qx) & (y > qy))
    else:
        qx = np.quantile(x, q)
        qy = np.quantile(y, q)
        both = np.sum((x < qx) & (y < qy))
    return both / (n * q)


def evt_tail_index(x, threshold_q=0.9):
    """Tail index xi via Hill estimator on excesses above threshold."""
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < 100:
        return np.nan
    thr = np.quantile(x, threshold_q)
    excess = x[x > thr] - thr
    if len(excess) < 1 or np.any(excess <= 0):
        return np.nan
    return float(np.mean(np.log(excess / excess.min()))) if excess.min() > 0 else np.nan


def reciprocal_tail_dep_matrix(returns, q=0.1, upper=True):
    cols = returns.columns
    k = len(cols)
    M = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            x = returns[cols[i]].to_numpy()
            y = returns[cols[j]].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 100:
                continue
            M[i, j] = M[j, i] = tail_dependence(x[mask], y[mask], q=q, upper=upper)
    return pd.DataFrame(M, index=cols, columns=cols)


def section_b_tail_dependence(returns):
    """Compute tail-dependence + EVT tail index, build a stress network."""
    print("\n=== SECTION B: Tail-Dependence & EVT ===")

    lo = reciprocal_tail_dep_matrix(returns, q=0.10, upper=False)
    up = reciprocal_tail_dep_matrix(returns, q=0.10, upper=True)

    lo.to_csv("phaseB_lower_tail_dependence.csv")
    up.to_csv("phaseB_upper_tail_dependence.csv")
    print("Saved lower/upper tail-dependence matrices.")

    # EVT tail index per market (on right tail of |returns|, i.e. volatility)
    evt = {}
    for col in returns.columns:
        x = np.abs(returns[col].to_numpy())
        evt[col] = evt_tail_index(x)
    evt_df = pd.DataFrame(list(evt.items()), columns=["market", "tail_index_xi"])
    evt_df.to_csv("phaseB_evt_tail_index.csv", index=False)
    print(evt_df.round(3).to_string(index=False))

    # Combined 'stress' network using upper tail dependence as edge weight
    thr = np.nanquantile(up.values[np.triu_indices(up.shape[0], k=1)], 0.5)
    thr = 0.35 if np.isnan(thr) else thr
    G = nx.Graph()
    G.add_nodes_from(up.columns)
    for i, a in enumerate(up.columns):
        for b in up.columns[i + 1:]:
            w = up.loc[a, b]
            if np.isfinite(w) and w >= thr:
                G.add_edge(a, b, weight=round(w, 3))

    fig = plt.figure(figsize=(13, 11))
    pos = nx.spring_layout(G, seed=42, k=1.1, weight="weight")
    node_col = [evt.get(n, np.nan) for n in G.nodes()]
    vmin, vmax = np.nanmin(node_col), np.nanmax(node_col)
    sc = nx.draw_networkx_nodes(G, pos, node_color=node_col,
                                node_size=[900 + 2500 * G.degree(n) for n in G.nodes()],
                                cmap="viridis", vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"] * 6 for u, v in G.edges()],
                           alpha=0.6, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    plt.colorbar(sc, label="EVT tail index ξ (higher = fatter right tail)")
    plt.title("Tail-dependence network (upper tail, q=0.10) — node colour = EVT tail index")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("phaseB_tail_dependence_network.png", dpi=150)
    print("Saved phaseB_tail_dependence_network.png")

    # Summary: strongest co-crash pairs (upper tail)
    pairs = []
    for i, a in enumerate(up.columns):
        for b in up.columns[i + 1:]:
            w = up.loc[a, b]
            if np.isfinite(w):
                pairs.append((a, b, w))
    pairs.sort(key=lambda t: -t[2])
    print("\nTop 8 strongest co-crash (upper tail) pairs:")
    for a, b, w in pairs[:8]:
        print(f"  {a.split(' (')[0]:<16} <-> {b.split(' (')[0]:<16}  {w:.3f}")

    # Compare: calm-period vs crisis-period tail dependence (COVID)
    pre = returns.loc[:PRE_COVID_END]
    cov = returns.loc["2020-01-01":"2020-12-31"]
    print("\nUpper-tail dependence: calm (<=Feb'20) vs COVID-2020")
    for i, a in enumerate(up.columns):
        for b in up.columns[i + 1:]:
            xp, yp = pre[a].to_numpy(), pre[b].to_numpy()
            xc, yc = cov[a].to_numpy(), cov[b].to_numpy()
            mp = np.isfinite(xp) & np.isfinite(yp)
            mc = np.isfinite(xc) & np.isfinite(yc)
            if mp.sum() < 80 or mc.sum() < 80:
                continue
            tp = tail_dependence(xp[mp], yp[mp], upper=True)
            tc = tail_dependence(xc[mc], yc[mc], upper=True)
            if tc - tp > 0.15 and tc > 0.35:
                print(f"  {a.split(' (')[0]:<14} <-> {b.split(' (')[0]:<14} "
                      f"calm {tp:.2f} -> COVID {tc:.2f}")
    return up, evt_df


PRE_COVID_END = "2020-02-18"


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    returns = load_returns()

    print("Fetching extended assets (FX / commodities / risk)...")
    extended_prices = None
    try:
        extended_prices = fetch_extended()
        if extended_prices is not None and not extended_prices.empty:
            section_a_extended_universe(returns, extended_prices)
        else:
            print("Extended fetch returned nothing; skipping Section A.")
    except Exception as e:
        print(f"Section A skipped: {e}")

    section_b_tail_dependence(returns)
    print("\nDone. Files: phaseA_*.csv/png, phaseB_*.csv/png, extended_assets_returns.csv")
