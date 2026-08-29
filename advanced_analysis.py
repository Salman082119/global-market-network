"""
Global Market Network - Phases 3, 4 & 5
========================================
Phase 3: Rolling correlations + crisis-period (static vs dynamic) analysis
Phase 4: Community detection (Louvain) + centrality measures
Phase 5: Granger causality + Diebold-Yilmaz spillover (VAR/FEVD) + COVID event study

Reads raw_prices.csv produced by global_market_network.py.

Usage:
    python advanced_analysis.py
"""

import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

CRISIS_WINDOWS = {
    "GFC (Sep08-Mar09)": ("2008-09-01", "2009-03-31"),
    "China deval. (Aug15-Feb16)": ("2015-08-10", "2016-02-11"),
    "COVID (Feb19-Mar23 '20)": ("2020-02-19", "2020-03-23"),
    "Rate-hike bear (Jan-Oct22)": ("2022-01-03", "2022-10-12"),
}

SPAND_COLORS = ["red", "purple", "orange", "brown"]

PRE_COVID_END = "2020-02-18"


def load_returns(path="raw_prices.csv") -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True).ffill()
    return np.log(prices / prices.shift(1)).dropna(how="any")


# ----------------------------------------------------------------------
# PHASE 3 - ROLLING CORRELATION & CRISIS SPIKES
# ----------------------------------------------------------------------
def average_pairwise_correlation(returns: pd.DataFrame, window: int = 90) -> pd.Series:
    rc = returns.rolling(window).corr()
    n = returns.shape[1]
    triu = np.triu_indices(n, k=1)
    means, dates = [], []
    for date, block in rc.groupby(level=0):
        m = block.to_numpy()
        c = m[triu]
        c = c[~np.isnan(c)]
        means.append(c.mean() if len(c) else np.nan)
        dates.append(date)
    return pd.Series(means, index=pd.DatetimeIndex(dates))


def plot_rolling_correlation(avg_corr: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(avg_corr.index, avg_corr.values, color="navy", lw=1.2,
            label="90-day avg pairwise correlation")
    for i, (label, (start, end)) in enumerate(CRISIS_WINDOWS.items()):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=SPAND_COLORS[i % len(SPAND_COLORS)], alpha=0.15, label=label)
    ax.set_title("Average Pairwise Correlation Across Global Markets (90-day rolling)", fontsize=14)
    ax.set_ylabel("Avg pairwise corr")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig("phase3_rolling_correlation.png", dpi=150)
    print("Saved phase3_rolling_correlation.png")


def mean_upper(corr_df: pd.DataFrame) -> float:
    a = corr_df.to_numpy()
    return a[np.triu_indices(a.shape[0], k=1)].mean()


def compare_period_heatmaps(returns: pd.DataFrame):
    periods = {"Full period": returns} | {
        label: returns.loc[(returns.index >= start) & (returns.index <= end)]
        for label, (start, end) in CRISIS_WINDOWS.items()
    }
    ncols = 3
    nrows = int(np.ceil(len(periods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, r) in zip(axes, periods.items()):
        sns.heatmap(r.corr(), ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    square=True, cbar=False)
        ax.set_title(f"{label}\navg corr = {mean_upper(r.corr()):.3f}", fontsize=11)
    for ax in axes[len(periods):]:
        ax.axis("off")
    plt.suptitle("Static vs Crisis Correlation Structure", fontsize=14)
    plt.tight_layout()
    plt.savefig("phase3_crisis_heatmaps.png", dpi=150, bbox_inches="tight")
    print("Saved phase3_crisis_heatmaps.png")


# ----------------------------------------------------------------------
# PHASE 4 - LOUVAIN COMMUNITIES & CENTRALITY
# ----------------------------------------------------------------------
def community_analysis(corr: pd.DataFrame, threshold: float = 0.4):
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            w = corr.loc[a, b]
            if abs(w) >= threshold:
                G.add_edge(a, b, weight=round(w, 3))

    communities = nx.community.louvain_communities(G, seed=42, weight="weight")

    plt.figure(figsize=(13, 10))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    palette = plt.cm.tab10.colors
    comm_map = {node: ci for ci, comm in enumerate(communities) for node in comm}
    node_colors = [palette[comm_map[n] % len(palette)] for n in G.nodes()]
    sizes = [3000 * nx.degree_centrality(G)[n] + 500 for n in G.nodes()]
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.4, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    plt.title("Market Communities (Louvain, edge = |corr| > 0.4)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("phase4_communities_network.png", dpi=150)
    print("Saved phase4_communities_network.png\n")

    print("Communities detected:")
    for ci, comm in enumerate(communities):
        print(f"  Cluster {ci}: {sorted(comm)}")

    cent = pd.DataFrame({
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight", max_iter=1000),
    })
    cent["avg_rank"] = cent.rank(ascending=False).mean(axis=1)
    cent = cent.sort_values("avg_rank")
    cent.round(4).to_csv("phase4_centrality.csv")
    print("\nCentrality measures (saved phase4_centrality.csv):")
    print(cent.round(3))
    return G


# ----------------------------------------------------------------------
# PHASE 5A - GRANGER CAUSALITY
# ----------------------------------------------------------------------
def granger_analysis(returns: pd.DataFrame, maxlag: int = 5, alpha: float = 0.05):
    cols = list(returns.columns)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols)

    for cause in cols:
        for effect in cols:
            if cause == effect:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = grangercausalitytests(returns[[effect, cause]], maxlag=maxlag)
                pmat.loc[cause, effect] = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
            except Exception:
                pass

    pmat.to_csv("phase5_granger_pvalues.csv")
    n_tests = pmat.notna().sum().sum()
    sig = pmat.stack()[lambda s: s < alpha]
    bonf = alpha / n_tests
    sig_bonf = pmat.stack()[lambda s: s < bonf]

    print(f"\nADF stationarity check (log returns should be stationary):")
    adf_p = {c: adfuller(returns[c].dropna())[1] for c in cols[:3]}
    for c, p in adf_p.items():
        print(f"  {c}: p={p:.2e} -> {'stationary' if p < 0.05 else 'NON-stationary'}")

    print(f"\nGranger causality ({n_tests} directed pairs, lags 1-{maxlag}):")
    print(f"  Significant at {alpha:.0%}: {len(sig)} links "
          f"(Bonferroni-corrected p < {bonf:.5f}: {len(sig_bonf)} links)")
    print("\nStrongest 10 causal links (cause -> effect):")
    for (cause, effect), p in sig.sort_values().head(10).items():
        print(f"  {cause} -> {effect}: p={p:.2e}")
    return pmat


# ----------------------------------------------------------------------
# PHASE 5B - DIEBOLD-YILMAZ SPILLOVER (VAR + GENERALIZED FEVD)
# ----------------------------------------------------------------------
def dy_spillover(returns: pd.DataFrame, horizon: int = 10, maxlags: int = 10):
    model = VAR(returns)
    sel = model.select_order(maxlags=maxlags)
    lag = max(int(sel.aic), 1)
    print(f"\nVAR lag selected by AIC: {lag}; FEVD horizon: {horizon}")

    res = model.fit(lag)
    sigma = np.asarray(res.sigma_u)
    phi = res.ma_rep(horizon)
    m = phi @ sigma

    k = returns.shape[1]
    theta = np.zeros((k, k))
    for i in range(k):
        denom = np.sum(m[:, i, i])
        for j in range(k):
            theta[i, j] = np.sum(m[:, i, j] ** 2) / sigma[j, j] / denom
    S = theta / theta.sum(axis=1, keepdims=True)

    names = list(returns.columns)
    Smat = pd.DataFrame(S, index=[f"FROM {n}" for n in names],
                        columns=[f"TO {n}" for n in names])

    to_others = S.sum(axis=0) - np.diag(S)
    from_others = S.sum(axis=1) - np.diag(S)
    total = S[~np.eye(k, dtype=bool)].sum() / k * 100

    summary = pd.DataFrame({
        "contributes_TO_others (%)": to_others * 100,
        "is_affected_FROM_others (%)": from_others * 100,
        "net_spillover (%)": (to_others - from_others) * 100,
    }, index=names).sort_values("net_spillover (%)", ascending=False)

    Smat.to_csv("phase5_spillover_matrix.csv")
    print(summary.round(2))
    print(f"\nTotal spillover index: {total:.2f}% "
          f"(share of forecast-error variance coming from cross-market spillovers)")

    plt.figure(figsize=(10, 8))
    sns.heatmap(Smat, cmap="viridis", annot=True, fmt=".2f", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title(f"Generalized FEVD Spillover Matrix (H={horizon})\nTotal spillover index = {total:.1f}%",
              fontsize=13)
    plt.tight_layout()
    plt.savefig("phase5_spillover_heatmap.png", dpi=150)
    print("Saved phase5_spillover_heatmap.png")
    return summary


# ----------------------------------------------------------------------
# PHASE 5C - COVID CRASH EVENT STUDY
# ----------------------------------------------------------------------
def covid_event_study(returns: pd.DataFrame, prices_path="raw_prices.csv"):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).ffill()
    win = prices.loc["2020-02-19":"2020-04-30"]
    us_name = [c for c in prices.columns if "S&P500" in c][0]
    us_trough = win[us_name].idxmin()

    rows = []
    for mkt in win.columns:
        cummax = win[mkt].cummax()
        dd_series = win[mkt] / cummax - 1
        trough_date = dd_series.idxmin()
        roll20 = returns[us_name].rolling(20).corr(returns[mkt])
        pre = roll20.loc[:"2020-02-18"].tail(250).mean()
        crash = roll20.loc["2020-02-19":"2020-04-15"].mean()
        rows.append({
            "market": mkt,
            "max_drawdown_%": dd_series.min() * 100,
            "trough_date": trough_date.date(),
            "days_vs_US_trough": (trough_date - us_trough).days,
            "pre_covid_corr_with_US": pre,
            "crash_corr_with_US": crash,
            "delta_corr": crash - pre,
        })

    tbl = pd.DataFrame(rows).sort_values("days_vs_US_trough")
    tbl.to_csv("phase5_covid_event_study.csv", index=False)
    print(f"\nCOVID event study (US trough: {us_trough.date()}):\n")
    print(tbl.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rebased = win / win.iloc[0] * 100
    for mkt in rebase_order(rebased, us_name):
        lw = 2.5 if mkt == us_name else 1.2
        axes[0].plot(rebased.index, rebased[mkt], lw=lw)
    axes[0].axvline(us_trough, color="black", ls="--", alpha=0.6)
    axes[0].set_title("Rebased prices (2020-02-19 = 100)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    d = tbl.sort_values("delta_corr")
    axes[1].barh(d["market"], d["delta_corr"], color=np.where(d["delta_corr"] > 0, "crimson", "steelblue"))
    axes[1].set_title("Change in 20-day correlation with US\n(pre-COVID vs crash)")
    axes[1].tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig("phase5_covid_event_study.png", dpi=150)
    print("Saved phase5_covid_event_study.png")


def rebase_order(rebased: pd.DataFrame, us_name: str):
    cols = [c for c in rebased.columns if c != us_name]
    return cols + [us_name]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Loading data...")
    returns = load_returns()
    print(f"Log returns: {returns.shape}, {returns.index.min().date()} -> {returns.index.max().date()}")

    print("\n" + "=" * 70)
    print("PHASE 3: ROLLING CORRELATION & CRISIS ANALYSIS")
    print("=" * 70)
    avg_corr = average_pairwise_correlation(returns, window=90)
    normal_avg = avg_corr.mean()
    for label, (start, end) in CRISIS_WINDOWS.items():
        seg = avg_corr.loc[start:end]
        if seg.empty:
            print(f"  {label}: outside data range, skipped")
            continue
        print(f"  Avg pairwise corr during {label}: {seg.mean():.3f} "
              f"(vs overall mean {normal_avg:.3f})")
    plot_rolling_correlation(avg_corr)
    compare_period_heatmaps(returns)

    print("\n" + "=" * 70)
    print("PHASE 4: COMMUNITIES & CENTRALITY")
    print("=" * 70)
    corr = returns.corr()
    G = community_analysis(corr, threshold=0.4)

    print("\n" + "=" * 70)
    print("PHASE 5: CONTAGION MODELING")
    print("=" * 70)
    granger_analysis(returns)
    dy_spillover(returns)
    covid_event_study(returns)

    print("\nDone! New files:")
    for f in ["phase3_rolling_correlation.png", "phase3_crisis_heatmaps.png",
              "phase4_communities_network.png", "phase4_centrality.csv",
              "phase5_granger_pvalues.csv", "phase5_spillover_matrix.csv",
              "phase5_spillover_heatmap.png", "phase5_covid_event_study.csv",
              "phase5_covid_event_study.png"]:
        print(f"  {f}")
