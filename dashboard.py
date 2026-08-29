"""
Global Market Contagion Network - Interactive Dashboard v3 (Phase 6)
=====================================================================
Interactive Plotly network (drag / zoom / hover), timeline slider with crisis
quick-jumps and animated playback, pair-level correlation explorer with VIX
overlay, Diebold-Yilmaz spillover view, Granger causality table, dark/light
theme, and one-click data refresh from Yahoo Finance.

Usage:
    streamlit run dashboard.py
"""

import time

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Global Market Contagion Network",
                   initial_sidebar_state="expanded")

CRISIS_EVENTS = {
    "GFC crash": "2008-10-15",
    "China devaluation": "2015-08-24",
    "COVID panic peak": "2020-03-16",
    "2022 rate-hike bear": "2022-06-13",
}
CRISIS_WINDOWS = {
    "GFC (Sep08-Mar09)": ("2008-09-01", "2009-03-31"),
    "China deval. (Aug15-Feb16)": ("2015-08-10", "2016-02-11"),
    "COVID (Feb19-Mar23 '20)": ("2020-02-19", "2020-03-23"),
    "Rate-hike bear (Jan-Oct22)": ("2022-01-03", "2022-10-12"),
}
VRECT_FILL = ["rgba(239,83,80,0.12)", "rgba(171,71,188,0.14)",
              "rgba(255,167,38,0.16)", "rgba(141,110,99,0.12)"]
PALETTE = ["#7EB3E8", "#FFB55C", "#8CD18C", "#F08A8A", "#CBA6DD", "#7FD1C8"]
STEP_DAYS = {"1 month": 21, "1 quarter": 63, "6 months": 126, "1 year": 252}
REFRESH_TICKERS = {
    "S&P500 (US)": "^GSPC", "Nasdaq (US)": "^IXIC", "FTSE100 (UK)": "^FTSE",
    "DAX (Germany)": "^GDAXI", "CAC40 (France)": "^FCHI", "Nikkei225 (Japan)": "^N225",
    "HangSeng (HK)": "^HSI", "Shanghai (China)": "000001.SS", "Nifty50 (India)": "^NSEI",
    "Bovespa (Brazil)": "^BVSP", "TSX (Canada)": "^GSPTSE", "ASX200 (Australia)": "^AXJO",
}
THEMES = {
    "Dark": {"template": "plotly_dark", "paper": "#0e1117", "plot": "#0e1117",
             "font": "#fafafa", "node_line": "#fafafa", "edge": "#5b6673",
             "mean_line": "#9aa5b1"},
    "Light": {"template": "plotly_white", "paper": "#ffffff", "plot": "#ffffff",
              "font": "#222222", "node_line": "#1a1a1a", "edge": "#9aa5b1",
              "mean_line": "#777777"},
}

if "play" not in st.session_state:
    st.session_state.play = False


# ----------------------------------------------------------------------
# DATA & CACHED COMPUTATION
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    prices = pd.read_csv("raw_prices.csv", index_col=0, parse_dates=True).ffill()
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    return prices, returns


@st.cache_data(ttl=3600)
def load_vix():
    try:
        import yfinance as yf
        v = yf.download("^VIX", start="2005-01-01", progress=False)
        if v.empty:
            return None
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        s = v["Close"] if "Close" in v.columns else v.iloc[:, -1]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze("columns")
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index().rename("VIX")
    except Exception:
        return None


def refresh_latest_data(days: str = "45d"):
    try:
        import yfinance as yf
        old = pd.read_csv("raw_prices.csv", index_col=0, parse_dates=True)
        frames = []
        for name, ticker in REFRESH_TICKERS.items():
            df = yf.download(ticker, period=days, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                col = "Close" if "Close" in df.columns else df.columns[-1]
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze("columns")
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames.append(s.rename(name))
        merged = pd.concat([old] + frames)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv("raw_prices.csv")
        return True, f"Data updated through {merged.index.max():%d %b %Y}"
    except Exception as e:
        return False, str(e)


@st.cache_data
def rolling_avg_corr(_returns: pd.DataFrame, window: int) -> pd.Series:
    rc = _returns.rolling(window).corr()
    n = _returns.shape[1]
    triu = np.triu_indices(n, k=1)
    means, dates = [], []
    for date, block in rc.groupby(level=0):
        c = block.to_numpy()[triu]
        c = c[~np.isnan(c)]
        means.append(c.mean() if len(c) else np.nan)
        dates.append(date)
    return pd.Series(means, index=pd.DatetimeIndex(dates))


@st.cache_data
def build_state(_returns: pd.DataFrame, asof_str: str, window: int, threshold: float):
    asof_ts = pd.Timestamp(asof_str)
    wr = _returns.loc[:asof_ts].tail(window)
    corr = wr.corr()

    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            w = float(corr.loc[a, b])
            if abs(w) >= threshold:
                G.add_edge(a, b, weight=round(w, 3))

    communities = (nx.community.louvain_communities(G, seed=42, weight="weight")
                   if G.number_of_edges() else [])
    deg = nx.degree_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except Exception:
        eig = {n: np.nan for n in G.nodes()}
    return corr, G, communities, deg, eig


@st.cache_data
def pair_corr(_returns: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    return _returns[a].rolling(window).corr(_returns[b])


# ----------------------------------------------------------------------
# PLOTTING HELPERS
# ----------------------------------------------------------------------
def add_crisis_shading(fig: go.Figure, annotate: bool = True):
    for i, (label, (s, e)) in enumerate(CRISIS_WINDOWS.items()):
        fig.add_vrect(x0=s, x1=e, fillcolor=VRECT_FILL[i],
                      layer="below", line_width=0,
                      annotation_text=label if annotate else None,
                      annotation_position="top left",
                      annotation_font_size=9)


def download_chart(fig, filename, label="📥 Download chart"):
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    st.download_button(label, html, file_name=filename, mime="text/html",
                       key=f"dl_{filename}", width="stretch")


def add_vix_overlay(fig: go.Figure, vix: pd.Series, thm: dict):
    fig.add_trace(go.Scatter(
        x=vix.index, y=vix.values, name="VIX (right axis)",
        yaxis="y2", mode="lines",
        line=dict(color=thm["font"], width=1),
        opacity=0.35, hovertemplate="VIX: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    title=None, range=[0, float(np.nanmax(vix.values)) * 1.05]),
        yaxis=dict(domain=[0, 0.88]),
    )


def plot_network_fig(G, pos, communities, deg, eig, window, threshold, asof_ts, thm):
    fig = go.Figure()
    comm_map = {n: ci for ci, cm in enumerate(communities) for n in cm}

    for u, v, d in G.edges(data=True):
        fig.add_trace(go.Scatter(
            x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]],
            mode="lines",
            line=dict(width=max(float(d["weight"]), 0.05) * 6, color=thm["edge"]),
            opacity=0.45, hoverinfo="text", showlegend=False,
            hovertemplate=f"<b>{u}</b> <-> <b>{v}</b><br>r = {d['weight']:.3f}<extra></extra>",
        ))

    comm_names = {ci: f"Cluster {ci} ({len(cm)} mkts)"
                  for ci, cm in enumerate(communities)}

    seen_clusters = set()
    for n in G.nodes():
        ci = comm_map.get(n, 0)
        ev = eig.get(n, np.nan)
        fig.add_trace(go.Scatter(
            x=[pos[n][0]], y=[pos[n][1]], mode="markers+text",
            marker=dict(size=24 + deg[n] * 60, color=PALETTE[ci % len(PALETTE)],
                        line=dict(width=1.5, color=thm["node_line"])),
            text=n.split(" (")[0], textposition="top center",
            textfont=dict(size=11, color=thm["font"]),
            name=comm_names.get(ci, f"Cluster {ci}"), legendgroup=f"g{ci}",
            showlegend=ci not in seen_clusters,
            customdata=[deg[n], ev],
            hovertemplate=(f"<b>{n}</b><br>cluster {ci}<br>"
                           "degree: %{customdata[0]:.2f}<br>"
                           "eigenvector: %{customdata[1]:.3f}<extra></extra>"),
        ))
        seen_clusters.add(ci)

    fig.update_layout(
        title=dict(text=f"Trailing {window}d network as of "
                        f"{asof_ts:%d %b %Y} (edge if |r| >= {threshold:.2f})",
                   font_size=14),
        height=600, margin=dict(l=10, r=10, t=52, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor=thm["paper"], plot_bgcolor=thm["plot"],
        font=dict(color=thm["font"]),
        uirevision=f"{window}|{threshold}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_size=12),
    )
    return fig


def plot_time_series(series: pd.Series, thm: dict, overall_mean: float | None = None,
                     asof_line=None, title: str = "", height: int = 320,
                     vix: pd.Series | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             line=dict(color="#5B9BD5", width=1.6),
                             name=series.name or "value"))
    add_crisis_shading(fig, annotate=vix is None)
    if overall_mean is not None:
        fig.add_hline(y=overall_mean, line=dict(color=thm["mean_line"], dash="dot", width=1),
                      annotation_text="long-run mean", annotation_font_size=9,
                      annotation_font_color=thm["mean_line"],
                      annotation_position="bottom right")
    if asof_line is not None:
        fig.add_vline(x=str(asof_line.date()), line=dict(color=thm["font"], dash="dash", width=1.2),
                      annotation_text=asof_line.strftime("%b %Y"), annotation_font_size=9)
    if vix is not None:
        add_vix_overlay(fig, vix.dropna(), thm)
    fig.update_layout(template=thm["template"],
                      title=dict(text=title, font_size=13), height=height,
                      margin=dict(l=10, r=vix is not None and 55 or 10, t=40, b=10),
                      paper_bgcolor=thm["paper"], plot_bgcolor=thm["plot"],
                      font=dict(color=thm["font"]),
                      legend=dict(orientation="h", y=-0.28, x=0, font_size=10),
                      hovermode="x unified")
    if vix is None:
        fig.update_yaxes(range=[0, 1])
    return fig


# ----------------------------------------------------------------------
# PAGE
# ----------------------------------------------------------------------
prices, returns = load_data()

# --- health check: verify required files exist ---
_required = ["raw_prices.csv", "phase5_spillover_matrix.csv",
             "phase5_granger_pvalues.csv", "phase8_gnn_predictions.csv",
             "phase8_gnn_metrics.csv"]
_missing = [f for f in _required if not __import__("os").path.exists(f)]
if _missing:
    st.warning(f"Missing output files: {', '.join(_missing)}. "
               "Run the analysis scripts first. Dashboard may be incomplete.")

# --- data freshness ---
data_age_days = (pd.Timestamp.now() - returns.index[-1]).days
if data_age_days <= 1:
    freshness = f"✅ Data fresh — updated {data_age_days}d ago"
elif data_age_days <= 7:
    freshness = f"⚠️ Data {data_age_days} days old"
else:
    freshness = f"🔴 Data {data_age_days} days stale"

st.sidebar.caption(freshness)

with st.sidebar:
    st.header("Controls")
    theme_name = st.radio("Theme", list(THEMES.keys()), horizontal=True,
                          key="theme_choice")
    thm = THEMES[theme_name]

    window = st.select_slider("Rolling window (trading days)",
                              [30, 45, 60, 90, 120], value=90)
    threshold = st.slider("Edge threshold |r|", 0.20, 0.90, 0.40, 0.05)
    st.divider()

    min_date = returns.index[min(window, len(returns) - 1)].date()
    max_date = returns.index[-1].date()

    if "_pending_date" in st.session_state:
        pending = st.session_state.pop("_pending_date")
        st.session_state.asof_date = pending

    if "asof_date" not in st.query_params:
        val = st.session_state.get("asof_date")
        if val is None:
            st.session_state.asof_date = max_date
        else:
            st.session_state.asof_date = min(max(val, min_date), max_date)
    asof = st.slider("As-of date", min_value=min_date, max_value=max_date,
                     format="DD MMM YYYY", key="asof_date")

    asof_ts = pd.Timestamp(asof)
    st.divider()
    pc1, pc2 = st.columns(2)
    if pc1.button("Play", disabled=st.session_state.play, width="stretch"):
        st.session_state.play = True
    if pc2.button("Pause", disabled=not st.session_state.play, width="stretch"):
        st.session_state.play = False
    step_label = st.select_slider("Playback step", list(STEP_DAYS.keys()), value="1 quarter")

    if st.session_state.play:
        idx = returns.index.searchsorted(asof_ts)
        nxt = min(idx + STEP_DAYS[step_label], len(returns) - 1)
        if nxt <= idx:
            st.session_state.play = False
        else:
            st.session_state["_pending_date"] = returns.index[nxt].date()
            time.sleep(0.3)
            st.rerun()

    st.divider()

    if st.button("Refresh latest data from Yahoo Finance", width="stretch"):
        with st.spinner("Fetching last 45 days for all 12 indices..."):
            ok, msg = refresh_latest_data()
        if ok:
            st.cache_data.clear()
            st.toast(msg, icon="✅")
            st.rerun()
        else:
            st.error(f"Refresh failed: {msg}")

corr, G, communities, deg, eig = build_state(returns, asof.isoformat(), window, threshold)
avg_corr_series = rolling_avg_corr(returns, window)
overall_avg = float(avg_corr_series.mean())
current_avg = float(avg_corr_series.loc[:asof_ts].iloc[-1])
vix = load_vix()

active_regime = next((lbl for lbl, (s, e) in CRISIS_WINDOWS.items()
                      if pd.Timestamp(s) <= asof_ts <= pd.Timestamp(e)), None)

if active_regime:
    regime_label = active_regime.split(" (")[0]
    regime_color = "🔴"
else:
    if current_avg > 0.6:
        regime_label = "High Stress"
        regime_color = "🟠"
    elif current_avg > 0.45:
        regime_label = "Elevated"
        regime_color = "🟡"
    else:
        regime_label = "Calm"
        regime_color = "🟢"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg pairwise corr", f"{current_avg:.3f}", f"{current_avg - overall_avg:+.3f} vs mean")
c2.metric("Edges above threshold", f"{G.number_of_edges()} / {len(corr)*(len(corr)-1)//2}")
c3.metric("Network density", f"{nx.density(G):.2f}")
hub = max(deg, key=deg.get)
c4.metric("Most connected hub", hub.split(" (")[0], f"degree {deg[hub]:.2f}")
c5.metric("Market regime", f"{regime_color} {regime_label}")

# --- Systemic Risk Score (0-100) ---
corr_score = min(current_avg / 0.8, 1.0) * 100
vix_val = float(vix.loc[:asof_ts].iloc[-1]) if vix is not None and len(vix.loc[:asof_ts]) else 20.0
vix_score = min(max((vix_val - 10) / 30, 0), 1) * 100
density_score = nx.density(G) * 100 / 0.667
risk_score = int(0.4 * corr_score + 0.35 * vix_score + 0.25 * density_score)
risk_score = max(0, min(100, risk_score))

if risk_score >= 75:
    risk_color = "#d32f2f"
    risk_label = "HIGH RISK"
elif risk_score >= 50:
    risk_color = "#f57c00"
    risk_label = "ELEVATED"
elif risk_score >= 30:
    risk_color = "#fbc02d"
    risk_label = "MODERATE"
else:
    risk_color = "#388e3c"
    risk_label = "LOW RISK"

st.markdown(f"""
<div style="background:{risk_color}22;border:1px solid {risk_color};border-radius:8px;
padding:10px 16px;margin:6px 0;display:flex;align-items:center;gap:12px">
  <span style="font-size:28px;font-weight:700;color:{risk_color}">{risk_score}</span>
  <div>
    <span style="font-size:13px;font-weight:600;color:{risk_color}">Systemic Risk Index</span><br>
    <span style="font-size:11px;color:#888">corr {corr_score:.0f} · vix {vix_score:.0f} · density {density_score:.0f}</span>
  </div>
  <span style="margin-left:auto;font-size:14px;font-weight:700;color:{risk_color}">{risk_label}</span>
</div>
""", unsafe_allow_html=True)

# --- Correlation Spike Alerts ---
alert_pairs = []
triu_pairs = [(corr.columns[i], corr.columns[j])
              for i in range(len(corr.columns)) for j in range(i+1, len(corr.columns))]
if len(returns) > 60:
    cur_corr_vals = corr.values[np.triu_indices(len(corr), k=1)]
    prev_date = asof_ts - pd.Timedelta(days=35)
    prev_wr = returns.loc[:prev_date].tail(window)
    if len(prev_wr) >= window:
        prev_corr = prev_wr.corr()
        prev_vals = prev_corr.values[np.triu_indices(len(prev_corr), k=1)]
        deltas = np.abs(cur_corr_vals - prev_vals)
        top_idx = np.argsort(deltas)[::-1][:5]
        alert_pairs = [(triu_pairs[k][0], triu_pairs[k][1], cur_corr_vals[k], deltas[k])
                       for k in top_idx if deltas[k] > 0.10]

if alert_pairs:
    alert_html = '<div style="background:#ff174415;border:1px solid #ff174455;border-radius:8px;padding:8px 14px;margin:4px 0">'
    alert_html += '<span style="font-size:12px;font-weight:600;color:#ff5252">🔔 Correlation Spikes (last 30 days)</span><br>'
    for a, b, val, delta in alert_pairs:
        direction = "↑" if val > (prev_vals[triu_pairs.index((a,b))] if (a,b) in triu_pairs else val) else "→"
        alert_html += f'<span style="font-size:11px;color:#ccc">• {a.split(" (")[0]} ↔ {b.split(" (")[0]}: <b>{val:.2f}</b> ({delta:+.2f} {direction})</span><br>'
    alert_html += '</div>'
    st.markdown(alert_html, unsafe_allow_html=True)

btns = st.columns([0.55, 1, 1, 1, 1, 0.85])
for col, (label, d) in zip(btns[1:], CRISIS_EVENTS.items()):
    if col.button(label, width="stretch"):
        st.session_state["_pending_date"] = pd.Timestamp(d).date()
        st.rerun()

tab_net, tab_pair, tab_spill, tab_pred, tab_evol, tab_about = st.tabs(
    ["🕸️ Network", "📈 Pairs", "🌊 Spillover & Causality",
     "🤖 GNN Prediction", "⏱️ Evolution", "ℹ️ About"])

# ---------------------------------------------------------------- network tab
with tab_net:
    left, right = st.columns([1.15, 1])
    with left:
        if G.number_of_edges():
            pos = nx.spring_layout(G, seed=42, k=0.95, weight="weight")
            _net_fig = plot_network_fig(G, pos, communities, deg, eig,
                                        window, threshold, asof_ts, thm)
            st.plotly_chart(_net_fig, width="stretch")
            download_chart(_net_fig, "network_graph.html")
        else:
            st.warning("No edges at this threshold/date combination — "
                       "lower the edge threshold.")
    with right:
        hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                       zmin=-0.2, zmax=1, aspect="auto")
        hm.update_layout(template=thm["template"],
                         title=dict(text=f"Correlation matrix ({window}d ending "
                                         f"{asof_ts:%d %b %Y})", font_size=13),
                         height=520, margin=dict(l=10, r=10, t=44, b=10),
                         paper_bgcolor=thm["paper"], font=dict(color=thm["font"]))
        hm.update_coloraxes(colorbar=dict(thickness=12, len=0.7))
        st.plotly_chart(hm, width="stretch")

    if vix is not None:
        show_vix = st.checkbox("Overlay VIX fear gauge (right axis)", value=True)
    else:
        show_vix = False
        st.caption("VIX overlay unavailable (could not fetch ^VIX).")

    st.plotly_chart(plot_time_series(
        avg_corr_series.rename(f"{window}-day avg pairwise corr"), thm,
        overall_mean=overall_avg, asof_line=asof_ts,
        title="Average pairwise correlation through time (shaded = crisis windows)",
        height=360, vix=vix if show_vix else None), width="stretch")

    with st.expander("Centrality ranking & detected clusters"):
        cent_tbl = pd.DataFrame({
            "degree": pd.Series(deg),
            "eigenvector": pd.Series(eig),
            "cluster": [next((ci for ci, cm in enumerate(communities) if n in cm), -1)
                        for n in corr.columns],
        }).sort_values("degree", ascending=False).round(3)
        st.dataframe(cent_tbl, width="stretch")
        st.download_button("Download centrality CSV",
                           cent_tbl.to_csv().encode(), "centrality_snapshot.csv", "text/csv")
        for ci, cm in enumerate(communities):
            st.markdown(f"**Cluster {ci}:** {', '.join(sorted(cm))}")

    # --- Period comparison mode ---
    with st.expander("⚖️ Compare: current vs historical period"):
        cmp_label = st.selectbox(
            "Compare against", list(CRISIS_EVENTS.keys()) + ["Long-run average"],
            key="cmp_select")
        st.caption("Top KPI deltas between current view and selected benchmark")
        if cmp_label == "Long-run average":
            cmp_avg = float(avg_corr_series.mean())
            cmp_metric = {
                "Avg pairwise corr": (current_avg, cmp_avg),
                "Network density": (nx.density(G), 0.39),
            }
        else:
            cmp_date = pd.Timestamp(CRISIS_EVENTS[cmp_label])
            cmp_wr = returns.loc[:cmp_date].tail(window)
            if len(cmp_wr) >= window:
                cmp_corr = cmp_wr.corr()
                cmp_vals = cmp_corr.values[np.triu_indices(len(cmp_corr), k=1)]
                cmp_avg = float(cmp_vals.mean()) if len(cmp_vals) else np.nan
                cmp_G = nx.Graph()
                cmp_G.add_nodes_from(cmp_corr.columns)
                for i, a in enumerate(cmp_corr.columns):
                    for b in cmp_corr.columns[i + 1:]:
                        w = float(cmp_corr.loc[a, b])
                        if abs(w) >= threshold:
                            cmp_G.add_edge(a, b, weight=round(w, 3))
                cmp_metric = {
                    "Avg pairwise corr": (current_avg, cmp_avg),
                    "Network density": (nx.density(G), nx.density(cmp_G)),
                    "Edges above threshold": (G.number_of_edges(), cmp_G.number_of_edges()),
                }
            else:
                cmp_metric = {}
        if cmp_metric:
            cmp_rows = []
            for k, (cur, base) in cmp_metric.items():
                delta = (cur - base)
                cmp_rows.append({"Metric": k, "Current": f"{cur:.3f}" if isinstance(cur, float) else cur,
                                 "Benchmark": f"{base:.3f}" if isinstance(base, float) else base,
                                 "Delta": f"{delta:+.3f}"})
            st.dataframe({"Metric": [], "Current": [], "Benchmark": [], "Delta": []}) if not cmp_rows else None
            for m in cmp_rows:
                dtxt = m["Delta"]
                sign = "🔼" if dtxt.startswith("+") else "🔽"
                st.markdown(f"**{m['Metric']}**: {m['Current']} vs {m['Benchmark']} — {sign} {dtxt}")

# ------------------------------------------------------------------ pair tab
with tab_pair:
    pa1, pa2, pa3 = st.columns([1, 1, 1])
    mkts = list(returns.columns)
    a = pa1.selectbox("Market A", mkts, index=0)
    b = pa2.selectbox("Market B", mkts, index=min(8, len(mkts) - 1))
    pw = pa3.select_slider("Correlation window", [30, 60, 90, 120, 180, 250], value=90)

    if a != b:
        s = pair_corr(returns, a, b, pw).rename(f"{pw}d correlation")
        use_vix_pair = vix is not None and st.checkbox("Overlay VIX (right axis)",
                                                       value=False, key="pair_vix")
        st.plotly_chart(plot_time_series(s, thm, title=f"{a} vs {b}",
                                         height=400,
                                         vix=vix if use_vix_pair else None),
                        width="stretch")
        s_valid = s.dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest", f"{s_valid.iloc[-1]:.2f}")
        imax, imin = s_valid.idxmax(), s_valid.idxmin()
        m2.metric("Max", f"{s_valid.max():.2f}", imax.strftime("%b %Y"))
        m3.metric("Min", f"{s_valid.min():.2f}", imin.strftime("%b %Y"))
        pre = s_valid.loc[:"2020-02-18"].mean()
        cov = s_valid.loc["2020-02-19":"2020-04-30"].mean()
        m4.metric("COVID shift", f"{cov - pre:+.2f}", f"pre {pre:.2f} -> crash {cov:.2f}")
    else:
        st.info("Pick two different markets.")

# --------------------------------------------------------------- spill tab
with tab_spill:
    try:
        Smat = pd.read_csv("phase5_spillover_matrix.csv", index_col=0)
    except FileNotFoundError:
        Smat = None
        st.info("Run `python advanced_analysis.py` once to generate spillover outputs.")

    if Smat is not None:
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            sh = px.imshow(Smat, text_auto=".2f", color_continuous_scale="Viridis",
                           aspect="auto")
            sh.update_layout(template=thm["template"],
                             title=dict(text="Generalized FEVD spillover matrix "
                                             "(VAR, H=10)", font_size=13),
                             height=500, margin=dict(l=10, r=10, t=44, b=10),
                             paper_bgcolor=thm["paper"], font=dict(color=thm["font"]))
            st.plotly_chart(sh, width="stretch")
        with sc2:
            S = Smat.to_numpy()
            k = S.shape[0]
            names = [c.replace("TO ", "") for c in Smat.columns]
            to_o = S.sum(axis=0) - np.diag(S)
            fr_o = S.sum(axis=1) - np.diag(S)
            net_df = (pd.DataFrame({"net_spillover_%": (to_o - fr_o) * 100}, index=names)
                      .sort_values("net_spillover_%"))
            total = S[~np.eye(k, dtype=bool)].sum() / k * 100
            nb = px.bar(net_df.reset_index(), x="net_spillover_%", y="index",
                        orientation="h", color="net_spillover_%",
                        color_continuous_scale=["#4575b4", "#ffffbf", "#d73027"])
            nb.update_layout(template=thm["template"],
                             title=dict(text=f"Net shock exporter (+) vs receiver (-)"
                                           f" · total index {total:.1f}%", font_size=13),
                             height=500, margin=dict(l=10, r=10, t=44, b=10),
                             yaxis_title=None, coloraxis_showscale=False,
                             paper_bgcolor=thm["paper"], font=dict(color=thm["font"]))
            st.plotly_chart(nb, width="stretch")

        try:
            gmat = pd.read_csv("phase5_granger_pvalues.csv", index_col=0)
            links = (gmat.stack().rename("p").reset_index()
                     .rename(columns={"level_0": "cause", "level_1": "effect"})
                     .sort_values("p").head(15))
            st.markdown("**Top 15 Granger-causal links (cause -> effect)**")
            links["p"] = links["p"].map(lambda p: f"{p:.2e}")
            st.dataframe(links, hide_index=True, width="stretch")
            st.download_button("Download full Granger p-value matrix",
                               gmat.to_csv().encode(), "granger_pvalues.csv", "text/csv")
        except FileNotFoundError:
            pass

    # --- Rolling spillover (Phase 7) ---
    try:
        rso = pd.read_csv("phase7_rolling_spillover.csv", parse_dates=["date"])
        st.markdown("**Rolling Diebold-Yilmaz total spillover** — 2-yr window, one-month "
                    "steps. Shows how contagion transmission rose around each crisis.")
        rf = go.Figure()
        rf.add_trace(go.Scatter(x=rso["date"], y=rso["total_spillover"],
                                mode="lines", name="Total spillover",
                                line=dict(color="#5B9BD5", width=2)))
        for i, (lbl, (s, e)) in enumerate(CRISIS_WINDOWS.items()):
            rf.add_vrect(x0=s, x1=e, fillcolor=VRECT_FILL[i], layer="below",
                         line_width=0, annotation_text=lbl.split(" (")[0],
                         annotation_position="top left", annotation_font_size=9)
        rf.add_hline(y=rso["total_spillover"].mean(),
                     line=dict(color=thm["mean_line"], dash="dot", width=1),
                     annotation_text="mean", annotation_position="bottom right",
                     annotation_font_size=9)
        rf.update_layout(template=thm["template"],
                         title=dict(text="Rolling total spillover index through time",
                                    font_size=13),
                         height=340, margin=dict(l=10, r=10, t=40, b=10),
                         paper_bgcolor=thm["paper"], font=dict(color=thm["font"]),
                         hovermode="x unified")
        st.plotly_chart(rf, width="stretch")
        download_chart(rf, "rolling_spillover.html")
        last_rso = rso["total_spillover"].iloc[-1]
        static_ref = f"{total:.1f}%" if "total" in dir() else "n/a"
        st.caption(f"Latest reading: {last_rso:.1f}% "
                   f"({rso['date'].iloc[-1]:%b %Y}) vs full-sample static {static_ref}")
    except FileNotFoundError:
        st.info("Run `python rolling_spillover.py` to generate the rolling spillover series.")

    # --- Tail dependence (opt-in, generated by optional_analysis.py) ---
    try:
        tup = pd.read_csv("phaseB_upper_tail_dependence.csv", index_col=0)
        tevt = pd.read_csv("phaseB_evt_tail_index.csv")
        st.markdown("**Tail-dependence network** — upper-tail co-crash probability "
                    "(q=0.10). Nodes coloured by EVT tail index ξ (higher = fatter "
                    "right tail / more extreme-crash-prone).")
        td1, td2 = st.columns([1, 1])
        with td1:
            tv = np.triu(tup.values, k=1)
            mask = tv == 0
            th = px.imshow(np.ma.masked_where(tv == 0, tv),
                           text_auto=".2f", color_continuous_scale="YlOrRd",
                           zmin=0, zmax=0.8, aspect="auto",
                           labels=dict(color="Upper-tail dep."))
            th.update_layout(template=thm["template"],
                             title=dict(text="Upper-tail dependence (co-crash)",
                                        font_size=13),
                             height=480, margin=dict(l=10, r=10, t=44, b=10),
                             paper_bgcolor=thm["paper"],
                             font=dict(color=thm["font"]))
            th.update_yaxes(autorange="reversed")
            st.plotly_chart(th, width="stretch")
        with td2:
            evt_sorted = tevt.sort_values("tail_index_xi")
            names = [n.split(" (")[0] for n in evt_sorted["market"]]
            eb = px.bar(evt_sorted, x="tail_index_xi", y=names, orientation="h",
                        color="tail_index_xi", color_continuous_scale="Magma")
            eb.update_layout(template=thm["template"],
                             title=dict(text="EVT tail index ξ by market", font_size=13),
                             height=480, margin=dict(l=10, r=10, t=44, b=10),
                             xaxis_title="ξ", yaxis_title=None,
                             coloraxis_showscale=False,
                             paper_bgcolor=thm["paper"],
                             font=dict(color=thm["font"]))
            st.plotly_chart(eb, width="stretch")
        download_chart(th, "tail_dependence.html")
    except FileNotFoundError:
        st.caption("Run `python optional_analysis.py` to generate tail-dependence outputs.")

# --------------------------------------------------------------- pred tab
with tab_pred:
    try:
        met8 = pd.read_csv("phase8_gnn_metrics.csv")
        pred8 = pd.read_csv("phase8_gnn_predictions.csv", parse_dates=["date"])
    except FileNotFoundError:
        met8 = None
        st.info("Run `python gnn_predict.py` once to generate GNN outputs.")

    if met8 is not None:
        st.markdown("**Next-month correlation prediction** — edge regression, 66 "
                    "pairs/month · GATv2 graph network with persistence residual · "
                    "temporal split (COVID held out of training)")
        mc = st.columns(len(met8))
        for i, row in met8.reset_index(drop=True).iterrows():
            mc[i].metric(str(row["model"]), f"MAE {row['MAE']:.3f}",
                         f"R2 {row['R2']:+.2f}")

        te = pred8[pred8["date"] > "2021-06-30"]
        avg = te.groupby("date")[["actual", "gnn", "persistence"]].mean().reset_index()

        f1 = go.Figure()
        add_crisis_shading(f1, annotate=False)
        f1.add_trace(go.Scatter(x=avg["date"], y=avg["actual"], name="Actual",
                                line=dict(color=thm["font"], width=2.2)))
        f1.add_trace(go.Scatter(x=avg["date"], y=avg["gnn"], name="GNN predicted",
                                line=dict(color="#d62728", width=1.6)))
        f1.add_trace(go.Scatter(x=avg["date"], y=avg["persistence"], name="Persistence",
                                line=dict(color="#5B9BD5", width=1.2, dash="dot")))
        f1.update_layout(template=thm["template"],
                         title=dict(text="Test period (Jul 2021 - Dec 2024): "
                                         "avg pairwise correlation", font_size=13),
                         height=330, margin=dict(l=10, r=10, t=44, b=10),
                         legend=dict(orientation="h", y=-0.25, x=0, font_size=10),
                         paper_bgcolor=thm["paper"], font=dict(color=thm["font"]),
                         hovermode="x unified")
        st.plotly_chart(f1, width="stretch")
        download_chart(f1, "gnn_prediction_timeline.html")

        sc1, sc2 = st.columns([1, 1])
        with sc1:
            f2 = go.Figure()
            f2.add_trace(go.Histogram2dContour(
                x=te["actual"], y=te["gnn"], colorscale="Hot",
                contours=dict(showlabels=True), ncontours=15,
                hovertemplate="density<extra></extra>"))
            f2.add_trace(go.Scatter(
                x=te["actual"], y=te["gnn"], mode="markers",
                marker=dict(size=3, color="white", opacity=0.25),
                hovertemplate="actual %{x:.2f}<br>pred %{y:.2f}<extra></extra>",
                showlegend=False))
            f2.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                         line=dict(color=thm["mean_line"], width=1.5, dash="dash"))
            f2.update_layout(template=thm["template"],
                             title=dict(text="Edge-level: actual vs predicted",
                                        font_size=12),
                             height=380, margin=dict(l=10, r=10, t=40, b=10),
                             xaxis_title="actual", yaxis_title="GNN",
                             paper_bgcolor=thm["paper"], font=dict(color=thm["font"]))
            st.plotly_chart(f2, width="stretch")
        with sc2:
            pair_mae = (te.assign(err=(te["gnn"] - te["actual"]).abs())
                        .groupby("pair")["err"].mean().sort_values())
            st.markdown("**Best predicted pairs**")
            st.dataframe(pair_mae.head(5).round(3).rename("MAE"),
                         height=175, width="stretch")
            st.markdown("**Hardest pairs**")
            st.dataframe(pair_mae.tail(5).sort_values(ascending=False)
                         .round(3).rename("MAE"), height=175, width="stretch")

        st.download_button("Download GNN predictions CSV",
                           pred8.to_csv().encode(),
                           "phase8_gnn_predictions.csv", "text/csv")

# ------------------------------------------------------------ evolution tab
with tab_evol:
    st.markdown("**Correlation matrix through time** — drag the slider or press Play to "
                "watch how the 12-market correlation structure evolves (red = high corr).")
    ev_c1, ev_c2 = st.columns([3, 1])
    with ev_c1:
        ev_step = st.select_slider("Interval", ["3 months", "6 months", "1 year", "2 years"],
                                   value="1 year", key="ev_step")
        ev_months = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}[ev_step]
    with ev_c2:
        ev_play = st.button("▶ Play" if not st.session_state.get("ev_play", False) else "⏸ Pause",
                            key="ev_play_btn")

    ev_wins = []
    ev_dates = list(returns.index)
    ev_start_idx = window
    for i in range(ev_start_idx, len(ev_dates), ev_months):
        ev_wins.append(ev_dates[i])

    if ev_play and not st.session_state.get("ev_playing", False):
        st.session_state.ev_playing = True

    ev_idx = st.session_state.get("ev_idx", 0)
    if ev_play:
        ev_idx = (ev_idx + 1) % len(ev_wins)
        st.session_state.ev_idx = ev_idx
        time.sleep(0.5)
        st.rerun()

    chosen_ev = st.select_slider("Year", list(ev_wins)[::-1][::-1],
                                 value=ev_wins[min(ev_idx, len(ev_wins)-1)],
                                 key="ev_year_slider")
    chosen_ts = pd.Timestamp(chosen_ev)
    ev_wr = returns.loc[:chosen_ts].tail(window)
    ev_corr = ev_wr.corr()

    ev_hm = px.imshow(ev_corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                      zmin=-0.2, zmax=1, aspect="auto")
    ev_hm.update_layout(template=thm["template"],
                        title=dict(text=f"Correlation matrix as of {chosen_ts:%d %b %Y}",
                                   font_size=13),
                        height=560, margin=dict(l=10, r=10, t=44, b=10),
                        paper_bgcolor=thm["paper"], font=dict(color=thm["font"]))
    ev_hm.update_coloraxes(colorbar=dict(thickness=12, len=0.7))
    st.plotly_chart(ev_hm, width="stretch")
    download_chart(ev_hm, "evolution_matrix.html")

    ev_avg_series = rolling_avg_corr(returns, window)
    st.plotly_chart(plot_time_series(
        ev_avg_series.rename(f"{window}-day avg pairwise corr"), thm,
        overall_mean=overall_avg, asof_line=chosen_ts,
        title="Average pairwise correlation with selected date marked",
        height=300, vix=None), width="stretch")

# --------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        """
### Methodology
- **Data**: daily closes for 12 global indices (Yahoo Finance), forward-filled across holidays,
  converted to log returns. Use the sidebar **Refresh** button to pull the latest 45 days.
- **Network**: an edge joins two markets when their trailing-window Pearson correlation exceeds
  the threshold. Node size = degree centrality, colour = Louvain community.
- **Pair analysis**: rolling correlation between any two markets, optionally overlaid with the
  VIX fear gauge to see how correlation spikes coincide with volatility regimes.
- **Spillover**: Diebold-Yilmaz index from a VAR(AIC-lag) generalized FEVD, horizon H=10.
- **Causality**: Granger tests (lags 1-5) on stationary log returns; interpret as *predictive*
  precedence, not economic causation.
- **GNN prediction**: monthly graph snapshots (trailing 90d correlations) feed a 2-layer
  GATv2 network that predicts every pair's next-month correlation; trained on data through
  Jun-2019 with COVID kept in validation, evaluated on 2021-2024 against persistence and
  Ridge baselines.

### Reproduce
```
pip install yfinance pandas numpy matplotlib seaborn networkx statsmodels plotly streamlit torch torch_geometric
python global_market_network.py     # fetch data + baseline visuals
python advanced_analysis.py         # phases 3-5 outputs used by this dashboard
python gnn_predict.py               # phase 8 GNN prediction outputs
streamlit run dashboard.py          # this dashboard
```

### Caveats
Daily closes ignore intra-day time-zone leads; Pearson captures linear dependence only;
correlation spikes in crises do not imply contagion channels by themselves.
        """
    )

st.divider()
st.caption("Data: Yahoo Finance · Rolling Pearson correlation · Louvain communities · "
           "Diebold-Yilmaz spillover · Correlation does not imply causation.")
