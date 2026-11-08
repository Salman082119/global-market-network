"""
Multi-Asset Global Market Advisor - Dashboard (Phase A - Core)
===============================================================
Tabs:
  1. 📊 Buy/Sell Advisor - ranked table, Top Buy / Top Sell cards, signal breakdown
  2. 📈 Price & Indicators - interactive per-asset chart with EMA/RSI/MACD context
  3. 📜 Backtest & Honesty - strategy vs benchmark, disclaimers
  4. 🧭 Market Context - the earlier global-market correlation/risk dashboard

Data files:
    run `python multi_asset_advisor.py` first to produce
    advisor_signals.csv / advisor_backtest.csv / advisor_prices.csv

Usage: streamlit run advisor_dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Multi-Asset Global Advisor",
                   initial_sidebar_state="expanded")

LABEL_COLORS = {
    "Strong Buy": "#1b7a3d", "Buy": "#4caf50",
    "Hold": "#fbc02d", "Sell": "#ef6c00", "Strong Sell": "#b71c1c",
}
EMOJI = {"Strong Buy": "🟢🟢", "Buy": "🟢", "Hold": "🟡", "Sell": "🔴",
         "Strong Sell": "🔴🔴"}

# ---------------------------------------------------------------- load
@st.cache_data(ttl=600, show_spinner=False)
def read_outputs():
    """Read committed CSVs (cached 10 min)."""
    sig = pd.read_csv("advisor_signals.csv")
    bt = pd.read_csv("advisor_backtest.csv", parse_dates=["date"])
    px_df = pd.read_csv("advisor_prices.csv", index_col=0, parse_dates=True)
    return sig, bt, px_df


@st.cache_data(ttl=600, show_spinner=False)
def live_fetch():
    """Fetch fresh data + regenerate signals/backtest CSVs, then read them."""
    from multi_asset_advisor import fetch_and_score
    fetch_and_score(progress_fn=None)
    return read_outputs.__wrapped__()


def load_data():
    try:
        return read_outputs()
    except Exception:
        # CSV missing/stale -> fetch fresh, then rebuild
        return live_fetch()


def calc_indicators(close, name):
    c = close.dropna()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macdsig = macd.ewm(span=9, adjust=False).mean()
    return c, ema20, ema50, ema200, rsi, macd, macdsig, name


try:
    SIG, BT, PX = load_data()
    HAVE_DATA = True
except Exception as e:
    HAVE_DATA = False
    st.error("Outputs missing. Run `python multi_asset_advisor.py` first. "
             f"({e})")


def advisor_header():
    st.markdown(
        "## 🌍 Multi-Asset Global Market Advisor\n"
        "Diversified global scan — indices, stocks, precious metals, FX, commodities "
        "and crypto. Every asset scored 0–100 by 5 technical signals "
        "(trend · RSI · MACD · volatility · price position).")

# ---------------------------------------------------------------- tab 1
def tab_advisor():
    st.markdown("### 📊 Buy / Sell Advisor")
    cats = ["All"] + sorted(SIG["category"].unique().tolist())
    filt_cat = st.selectbox("Asset category", cats)
    data = SIG if filt_cat == "All" else SIG[SIG["category"] == filt_cat]

    score_range = st.slider("Score range", 0, 100, (0, 100), 5)
    data = data[(data["score"] >= score_range[0]) & (data["score"] <= score_range[1])]

    qual = st.radio("Only show strong signals", ["All", "Only Strong Buy/Buy",
                                                  "Only Strong Sell/Sell"],
                    horizontal=True)
    if qual == "Only Strong Buy/Buy":
        data = data[data["label"].isin(["Strong Buy", "Buy"])]
    elif qual == "Only Strong Sell/Sell":
        data = data[data["label"].isin(["Strong Sell", "Sell"])]

    buys = data[data["label"].isin(["Strong Buy", "Buy"])].head(5)
    sells = data[data["label"].isin(["Strong Sell", "Sell"])].head(5)
    b1, b2 = st.columns([1, 1])
    with b1:
        st.markdown("#### 🟢 Top Buys")
        if len(buys):
            for _, r in buys.iterrows():
                st.markdown(
                    f"**{r['asset']}** — {EMOJI[r['label']]} {r['label']} "
                    f"({r['score']:.0f}/100)")
                st.caption(f"{r['category']} · price {r['price']:.2f} · "
                           f"RSI {r['rsi_raw']}")
        else:
            st.caption("No buys in this filter.")
    with b2:
        st.markdown("#### 🔴 Top Sells")
        if len(sells):
            for _, r in sells.iterrows():
                st.markdown(
                    f"**{r['asset']}** — {EMOJI[r['label']]} {r['label']} "
                    f"({r['score']:.0f}/100)")
                st.caption(f"{r['category']} · price {r['price']:.2f} · "
                           f"RSI {r['rsi_raw']}")
        else:
            st.caption("No sells in this filter.")

    st.markdown("---")
    st.markdown("#### 🔥 Today's top movers (quick)")
    moves = {c: daily_move(c) for c in PX.columns}
    moves = {k: v for k, v in moves.items() if v is not None and np.isfinite(v)}
    mdf = pd.DataFrame({"asset": list(moves), "move": list(moves.values())})
    mdf = mdf.sort_values("move", ascending=False)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**Top gainers**")
        for _, r in mdf.head(3).iterrows():
            st.markdown(f"- {r['asset']}: **{r['move']*100:+.1f}%**")
    with mc2:
        st.markdown("**Top losers**")
        for _, r in mdf.tail(3).iloc[::-1].iterrows():
            st.markdown(f"- {r['asset']}: **{r['move']*100:+.1f}%**")
    with mc3:
        avg_move = mdf["move"].mean() * 100
        up = (mdf["move"] > 0).mean() * 100
        st.metric("Avg session move", f"{avg_move:+.2f}%")
        st.metric("Share of assets up", f"{up:.0f}%")
        st.metric("Assets tracked", f"{len(mdf)}")

    st.markdown("---")
    st.markdown("#### Full ranked signal table")
    view = data.copy()
    view["signal"] = view["label"].apply(lambda x: f"{EMOJI[x]} {x}")
    table = view[["asset", "category", "price", "score", "signal",
                  "rsi_raw", "vol_ratio"]].rename(columns={
        "asset": "Asset", "category": "Category", "price": "Price",
        "score": "Score", "signal": "Signal", "rsi_raw": "RSI",
        "vol_ratio": "Vol ratio"})
    st.dataframe(table, width='stretch', height=420)
    st.download_button("📥 Download signal table (CSV)", SIG.to_csv().encode(),
                       "advisor_signals.csv", "text/csv", key="dl_sig")


# ---------------------------------------------------------------- tab 2
def tab_chart():
    st.markdown("### 📈 Price & Technicals")
    asset = st.selectbox("Asset", SIG.sort_values("asset")["asset"].tolist(),
                         key="chart_asset")
    c = PX[asset].dropna()
    close, e20, e50, e200, rsi, macd, msig, _ = calc_indicators(c, asset)
    if len(close) > 200:
        e200 = e200
    else:
        e200 = e50

    n = st.slider("Years to show", 1, 10, 3, key="chart_years")
    idx = close.index[-n * 252:] if len(close) > n * 252 else close.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=close.loc[idx], name="Price",
                             line=dict(color="#2962ff", width=1.8)))
    fig.add_trace(go.Scatter(x=idx, y=e20.loc[idx], name="EMA20",
                             line=dict(width=1, color="#fbc02d")))
    fig.add_trace(go.Scatter(x=idx, y=e50.loc[idx], name="EMA50",
                             line=dict(width=1, color="#d81b60")))
    if len(close) > 200:
        fig.add_trace(go.Scatter(x=idx, y=e200.loc[idx], name="EMA200",
                                 line=dict(width=1, color="#7b1fa2")))
    fig.update_layout(template="plotly_dark", height=480,
                      title=dict(text=f"{asset} — price + moving averages",
                                 font_size=13),
                      margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig, width='stretch')
    with c2:
        row = SIG[SIG["asset"] == asset].iloc[0]
        st.metric("Composite score", f"{row['score']:.1f}/100", row["label"])
        st.metric("Signal breakdown",
                  f"T{row['trend']:.0f} R{row['rsi']:.0f} M{row['macd']:.0f} "
                  f"V{row['vol']:.0f} P{row['pos']:.0f}")
        st.markdown("*Signal legend: T=trend, R=RSI, M=MACD, V=volatility, "
                    "P=price position (each 0-100)*")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=idx, y=rsi.loc[idx], name="RSI",
                              line=dict(color="#26a69a", width=1.3)))
    fig2.add_hline(y=70, line=dict(color="red", dash="dot", width=1))
    fig2.add_hline(y=30, line=dict(color="green", dash="dot", width=1))
    fig2.add_trace(go.Scatter(x=idx, y=macd.loc[idx], name="MACD",
                              yaxis="y2", line=dict(color="#ffb74d", width=1.2)))
    fig2.add_trace(go.Scatter(x=idx, y=msig.loc[idx], name="MACD signal",
                              yaxis="y2", line=dict(color="#fff", width=1, dash="dot")))
    fig2.update_layout(template="plotly_dark", height=300,
                       title=dict(text="RSI (0-100) & MACD", font_size=12),
                       margin=dict(l=10, r=10, t=40, b=10),
                       yaxis=dict(range=[0, 100]), hovermode="x unified",
                       yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig2, width='stretch')


# ---------------------------------------------------------------- tab 3
def tab_backtest():
    st.markdown("### 📜 Backtest & Honesty")
    st.markdown(
        "This backtest re-ranks all assets by **12-month momentum each month** and "
        "holds the top 25%. It compares that strategy against an equal-weight "
        "buy-and-hold of everything. **No look-ahead** was used — each month's pick "
        "uses only data up to that month. Past performance ≠ future returns.")
    if len(BT) == 0:
        st.info("No backtest data (file may be short).")
        return
    b1, b2 = st.columns([2, 1])
    with b1:
        strat = (1 + BT["strategy"]).cumprod()
        bench = (1 + BT["benchmark"]).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=BT["date"], y=strat, name="Momentum strategy",
                                 line=dict(color="#26a69a", width=2)))
        fig.add_trace(go.Scatter(x=BT["date"], y=bench, name="Equal-weight benchmark",
                                 line=dict(color="#fbc02d", width=1.6, dash="dot")))
        fig.update_layout(template="plotly_dark", height=420,
                          title=dict(text="Growth of 1.00 (compounding)",
                                     font_size=13),
                          margin=dict(l=10, r=10, t=40, b=10),
                          hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
    with b2:
        st.metric("Strategy multiplier", f"{strat.iloc[-1]:.2f}x")
        st.metric("Benchmark multiplier", f"{bench.iloc[-1]:.2f}x")
        out = strat.iloc[-1] / bench.iloc[-1]
        st.metric("Strategy ÷ benchmark", f"{out:.2f}x")
        win = (BT["strategy"] > BT["benchmark"]).mean()
        st.metric("Months strategy beat benchmark",
                  f"{win*100:.0f}%")
        st.markdown(f"*Best month strategy: {BT['strategy'].max()*100:+.1f}% · "
                    f"worst: {BT['strategy'].min()*100:+.1f}%*")

    st.markdown("#### Drawdown (peak-to-trough) — strategy vs benchmark")
    strat_peak = strat.cummax()
    bench_peak = bench.cummax()
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=BT["date"], y=(strat / strat_peak - 1) * 100,
                                name="Strategy drawdown %", fill="tozeroy",
                                line=dict(color="#ef5350", width=1)))
    fig_dd.add_trace(go.Scatter(x=BT["date"], y=(bench / bench_peak - 1) * 100,
                                name="Benchmark drawdown %", fill="tozeroy",
                                line=dict(color="#fbc02d", width=1, dash="dot")))
    fig_dd.update_layout(template="plotly_dark", height=320,
                         title=dict(text="Running drawdown (%)", font_size=13),
                         yaxis=dict(title="Drawdown %"), hovermode="x unified",
                         margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_dd, width='stretch')

    st.warning(
        "⚠️ **Not financial advice.** Signals are purely technical and can be "
        "wrong. This is an educational tool. Do your own research (DYOR) and "
        "consider consulting a SEBI/registered financial adviser. Markets carry "
        "risk of loss.")


# ---------------------------------------------------------------- tab 4
def tab_context():
    st.markdown("### 🧭 Market Context (correlation & contagion)")
    st.caption("Higher-level view of how global markets move together — see the "
               "dedicated dashboard for full interactivity.")

    rets = PX.pct_change().dropna(how="all")
    ind_cols = [c for c in PX.columns if CAT_IND.get(c, "") == "Indices"]
    if not ind_cols:
        ind_cols = PX.columns[:14]
    present = [c for c in ind_cols if c in PX.columns]
    if len(present) < 2:
        st.info("Not enough index series in saved prices; showing assets only.")
        idx_corr = rets.corr()
    else:
        idx_corr = rets[present].corr()
    idx_corr = idx_corr.dropna(how="all").dropna(axis=1, how="all")
    hm = px.imshow(idx_corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                   zmin=-0.2, zmax=1, aspect="auto")
    hm.update_layout(template="plotly_dark", height=620,
                     title=dict(text="Correlation of major indices (log returns)",
                                font_size=13),
                     margin=dict(l=10, r=10, t=44, b=10))
    st.plotly_chart(hm, width='stretch')
    st.caption("High pairwise correlation during crises means diversification "
               "frictions — see the earlier global-market dashboard for rolling "
               "spillover and contagion analysis.")


CAT_IND = {}


def _build_cat():
    for _, r in SIG.iterrows():
        CAT_IND[r["asset"]] = r["category"]


# ---------------------------------------------------------------- tab 5 perf
def period_returns():
    today = PX.index[-1]
    for col in PX.columns:
        c = PX[col].dropna()
        if len(c) < 2:
            continue
        use = c
        l1m = use.iloc[-1] / use[use.index <= today - pd.DateOffset(months=1)].iloc[-1] - 1 \
            if len(use[use.index <= today - pd.DateOffset(months=1)]) else np.nan
        l3m = use.iloc[-1] / use[use.index <= today - pd.DateOffset(months=3)].iloc[-1] - 1 \
            if len(use[use.index <= today - pd.DateOffset(months=3)]) else np.nan
        l6m = use.iloc[-1] / use[use.index <= today - pd.DateOffset(months=6)].iloc[-1] - 1 \
            if len(use[use.index <= today - pd.DateOffset(months=6)]) else np.nan
        l1y = use.iloc[-1] / use[use.index <= today - pd.DateOffset(years=1)].iloc[-1] - 1 \
            if len(use[use.index <= today - pd.DateOffset(years=1)]) else np.nan
        ytd = use.iloc[-1] / use[use.index <= pd.Timestamp(f"{today.year}-01-01")].iloc[-1] - 1 \
            if len(use[use.index <= pd.Timestamp(f"{today.year}-01-01")]) else np.nan
        out = {"asset": col, "1M": l1m, "3M": l3m, "6M": l6m, "1Y": l1y, "YTD": ytd}
        for k in ["1M", "3M", "6M", "1Y", "YTD"]:
            if np.isfinite(out[k]):
                pass
            else:
                out[k] = np.nan
        yield out


def tab_performance():
    st.markdown("### 📈 Performance (period returns)")
    rows = list(period_returns())
    if not rows:
        st.info("Not enough price history.")
        return
    df = pd.DataFrame(rows).set_index("asset")

    # max drawdown per asset
    dd = {}
    for col in PX.columns:
        c = PX[col].dropna()
        if len(c) < 30:
            continue
        peak = c.cummax()
        draw = c / peak - 1
        dd[col] = float(draw.min())
    df["MaxDD"] = pd.Series(dd)

    sel = st.multiselect("Select assets", df.index.tolist(),
                         default=df.index.tolist()[:12])
    if sel:
        view = df.loc[sel].copy()
    else:
        view = df
    hm = px.imshow(view[["1M", "3M", "6M", "1Y", "MaxDD"]].T,
                   text_auto=".0%", color_continuous_scale="RdYlGn",
                   zmin=-0.3, zmax=0.3, aspect="auto")
    hm.update_layout(template="plotly_dark", height=420,
                     title=dict(text="Returns heatmap (%); MaxDD = worst peak-to-trough",
                                font_size=13),
                     margin=dict(l=10, r=10, t=44, b=10))
    st.plotly_chart(hm, width='stretch')

    st.markdown("#### Period returns table")
    show = view[["1M", "3M", "6M", "1Y", "MaxDD"]].round(3)
    st.dataframe(show, width='stretch')
    st.download_button("📥 Download returns CSV", show.to_csv().encode(),
                       "advisor_period_returns.csv", "text/csv", key="dl_perf")

    worst = df["MaxDD"].dropna().sort_values().head(8).index.tolist()
    st.markdown("#### ⚠️ Deepest drawdowns (crash-risk assets)")
    for w in worst:
        st.markdown(f"- **{w}** — {df.loc[w, 'MaxDD']*100:.0f}% peak-to-trough "
                    f"(historical worst) · current score {SIG[SIG['asset']==w]['score'].iloc[0]:.0f}")


# ---------------------------------------------------------------- tab 6 port
def tab_portfolio():
    st.markdown("### 💼 Portfolio Builder")
    budget = st.number_input("Budget (INR or your currency)", min_value=1000.0,
                             value=100000.0, step=10000.0, key="budget")
    risk = st.radio("Risk appetite", ["Conservative", "Balanced", "Aggressive"],
                    horizontal=True, index=1, key="risk")

    # weights by risk
    w_buy, w_hold, w_min_score = {"Conservative": (0.6, 0.4, 65),
                                  "Balanced": (0.75, 0.25, 60),
                                  "Aggressive": (0.9, 0.1, 55)}[risk]

    strong = SIG[(SIG["label"].isin(["Strong Buy", "Buy"])) &
                 (SIG["score"] >= w_min_score)].sort_values("score", ascending=False)
    if strong.empty:
        strong = SIG.sort_values("score", ascending=False).head(6)

    # diversification: avoid duplicate high-correlation pick (simple: cap per category)
    top = []
    cat_count = {}
    for _, r in strong.iterrows():
        cc = cat_count.get(r["category"], 0)
        if cc >= 3:
            continue
        top.append(r)
        cat_count[r["category"]] = cc + 1
        if len(top) == 8:
            break

    st.markdown(f"**Suggested allocation (₹{budget:,.0f}, {risk}):**")
    n = len(top)
    total_score = sum(r["score"] for r in top)
    rows = []
    for r in top:
        if risk == "Conservative":
            share = 1.0 / n
        else:
            share = r["score"] / total_score
        allocate = round(budget * share, 0)
        rows.append({"Asset": r["asset"], "Category": r["category"],
                     "Signal": r["label"], "Score": r["score"],
                     "Weight": f"{share*100:.1f}%",
                     "Suggested ₹": int(allocate)})
    port_df = pd.DataFrame(rows)
    st.markdown(f"*{risk} strategy — invest across {len(top)} assets, "
                f"{'higher weight to stronger scores' if risk != 'Conservative' else 'equal weight across all'}.*")
    st.dataframe(port_df, width='stretch')
    st.caption(f"Note: allocation is risk-score-weighted and category-diversified "
               f"(max 3 per category). Hold ~{w_hold*100:.0f}% cash buffer in built-up "
               f"risk. Not financial advice.")
    hb = st.slider("Cash buffer to keep (%)", 0, 50, 15, key="buffer")
    invest = budget * (1 - hb / 100)
    st.metric("Amount to deploy", f"{invest:,.0f}", f"keeping {budget-invest:,.0f} cash")


# ---------------------------------------------------------------- tab 7 strat
def tab_strategy():
    st.markdown("### 🎯 Strategy presets")
    preset = st.selectbox("Strategy lens", ["Momentum", "Balanced", "Swing trade",
                                             "Long-term value", "Defensive"],
                          key="preset")
    # Revise scores under each lens by re-weighting the 5 signal components
    temp = SIG.copy()
    w = {"Momentum": dict(trend=0.20, rsi=0.15, macd=0.25, vol=0.10, pos=0.30),
         "Balanced": dict(trend=0.22, rsi=0.22, macd=0.20, vol=0.18, pos=0.18),
         "Swing trade": dict(trend=0.15, rsi=0.30, macd=0.20, vol=0.15, pos=0.20),
         "Long-term value": dict(trend=0.35, rsi=0.10, macd=0.15, vol=0.15, pos=0.25),
         "Defensive": dict(trend=0.30, rsi=0.15, macd=0.15, vol=0.25, pos=0.15)}[preset]
    temp["score"] = (temp["trend"] * w["trend"] + temp["rsi"] * w["rsi"] +
                     temp["macd"] * w["macd"] + temp["vol"] * w["vol"] +
                     temp["pos"] * w["pos"])
    temp["label"] = np.where(temp["score"] >= 75, "Strong Buy",
                     np.where(temp["score"] >= 60, "Buy",
                     np.where(temp["score"] >= 40, "Hold",
                     np.where(temp["score"] >= 25, "Sell", "Strong Sell"))))
    temp = temp.sort_values("score", ascending=False)
    st.markdown(f"**Top picks under '{preset}' strategy:**")
    view = temp[["asset", "category", "score", "label"]].head(10).copy()
    view["signal"] = view["label"].apply(lambda x: f"{EMOJI[x]} {x}")
    st.dataframe(view[["asset", "category", "score", "signal"]], width='stretch')
    st.caption("Same assets, re-weighted signals for a different trading style.")


# ---------------------------------------------------------------- tab 8 priv
def tab_private():
    st.markdown("### 🧪 Compare Mode")
    a = st.selectbox("Asset A", SIG.sort_values("asset")["asset"].tolist(),
                     key="cmp_a", index=0)
    b = st.selectbox("Asset B", SIG.sort_values("asset")["asset"].tolist(),
                     key="cmp_b", index=1)
    if a == b:
        st.info("Pick two different assets.")
        return
    ca = PX[a].dropna()
    cb = PX[b].dropna()
    # normalize to 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ca.index, y=100 * ca / ca.iloc[0],
                             name=a, line=dict(width=1.6)))
    fig.add_trace(go.Scatter(x=cb.index, y=100 * cb / cb.iloc[0],
                             name=b, line=dict(width=1.6)))
    fig.update_layout(template="plotly_dark", height=400,
                      title=dict(text=f"{a} vs {b} (rebased to 100)", font_size=13),
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, width='stretch')
    ra = PX[a].pct_change().dropna()
    rb = PX[b].pct_change().dropna()
    joint = pd.concat([ra, rb], axis=1, join="inner").dropna()
    corr = joint.corr().iloc[0, 1] if len(joint) else np.nan
    sa = SIG[SIG["asset"] == a].iloc[0]
    sb = SIG[SIG["asset"] == b].iloc[0]
    st.metric("Correlation (returns)", f"{corr:.2f}")
    st.metric(f"{a} signal", sa["label"], f"{sa['score']:.0f}")
    st.metric(f"{b} signal", sb["label"], f"{sb['score']:.0f}")
    if corr > 0.8:
        st.warning("These two are highly correlated — buying both is like buying "
                   "the same bet twice. Diversification is limited.")

    st.markdown("#### 52-week context")
    c1, c2 = st.columns(2)
    for col, cdiv in [(a, c1), (b, c2)]:
        c = PX[col].dropna().tail(270)
        hi = c.max()
        price = c.iloc[-1]
        off_high = (price - hi) / hi * 100
        with cdiv:
            st.metric(f"{col} — off 52wk high", f"{off_high:.1f}%",
                      f"price {price:.2f}")
            st.caption("0% = at 52-week high (breakout zone); more negative = cheaper "
                       "vs its high.")


# ---------------------------------------------------------------- tab 9 move
def daily_move(col):
    c = PX[col].dropna()
    if len(c) < 2:
        return np.nan
    return float(c.iloc[-1] / c.iloc[-2] - 1)


def tab_move():
    st.markdown("### 🔥 Market Move (latest session)")
    moves = {c: daily_move(c) for c in PX.columns}
    moves = {k: v for k, v in moves.items() if v is not None and np.isfinite(v)}
    df = pd.DataFrame({"asset": list(moves), "move": list(moves.values())})
    df["cat"] = df["asset"].map(CAT_IND)
    df["move%"] = df["move"] * 100
    df = df.sort_values("move", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Top gainers")
        g = df.head(6)
        fig = go.Figure(go.Bar(x=g["move%"], y=g["asset"], orientation="h",
                               marker_color="#26a69a"))
        fig.update_layout(template="plotly_dark", height=280,
                          xaxis_title="Move %", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.markdown("#### 📉 Top losers")
        l = df.tail(6).iloc[::-1]
        fig = go.Figure(go.Bar(x=l["move%"], y=l["asset"], orientation="h",
                               marker_color="#ef5350"))
        fig.update_layout(template="plotly_dark", height=280,
                          xaxis_title="Move %", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width='stretch')
    st.markdown("#### Full session moves")
    v = df[["asset", "cat", "move%"]].rename(columns={"asset": "Asset",
                                                      "cat": "Category",
                                                      "move%": "Move %"})
    st.dataframe(v, width='stretch')
    st.caption("Latest available trading session (last close vs previous close).")


# ---------------------------------------------------------------- tab 10 news
POS_WORDS = ["up", "gain", "beat", "surge", "rally", "record", "growth", "outperform",
             "rise", "jump", "strong", "bullish", "profit", "upgrade", "positive"]
NEG_WORDS = ["down", "fall", "miss", "drop", "crash", "loss", "weak", "downgrade",
             "decline", "slump", "bearish", "sell", "worry", "fear", "inflation",
             "recession", "negative", "cut", "lawsuit", "investigation"]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_sentiment(limit=10):
    import yfinance as yf
    out = []
    tick_map = {r["asset"]: UNIVERSE_TICK.get(r["asset"]) for _, r in SIG.iterrows()}
    scores = SIG.sort_values("score", ascending=False)["asset"].tolist()[:limit]
    for asset in scores:
        tick = tick_map.get(asset)
        if not tick:
            continue
        try:
            t = yf.Ticker(tick)
            news = t.news[:8] if hasattr(t, "news") and t.news else []
            if not news:
                continue
            text = " ".join(str(n.get("title", "")) + " " +
                            " ".join(map(str, n.get("relatedTickers", [])))
                            for n in news if isinstance(n, dict))
            tl = text.lower()
            pos = sum(1 for w in POS_WORDS if w in tl)
            neg = sum(1 for w in NEG_WORDS if w in tl)
            senti = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
            out.append({"asset": asset, "pos": pos, "neg": neg,
                        "sentiment": round(senti, 2),
                        "lean": "Positive" if senti > 0.15 else
                                ("Negative" if senti < -0.15 else "Neutral"),
                        "headlines": news[0]["title"] if news and isinstance(news[0], dict) else ""})
        except Exception:
            continue
    return pd.DataFrame(out)


UNIVERSE_TICK = {}


def _build_ticks():
    # reuse the tickers from the advisor universe module (lazy import handled in tab)
    UNIVERSE_TICK.update({
        "🇺🇸 S&P500": "^GSPC", "🇺🇸 NASDAQ100": "^NDX", "🇺🇸 Dow": "^DJI",
        "🇮🇳 NIFTY50": "^NSEI", "🇮🇳 SENSEX": "^BSESN", "🇬🇧 FTSE100": "^FTSE",
        "🇩🇪 DAX": "^GDAXI", "🇫🇷 CAC40": "^FCHI", "🇯🇵 Nikkei225": "^N225",
        "🇭🇰 HangSeng": "^HSI", "🇨🇳 Shanghai": "000001.SS", "🇧🇷 Bovespa": "^BVSP",
        "🇨🇦 TSX": "^GSPTSE", "🇦🇺 ASX200": "^AXJO",
        "🍎 Apple": "AAPL", "🚀 Tesla": "TSLA", "🖥️ Microsoft": "MSFT",
        "🔍 Google": "GOOGL", "🛒 Amazon": "AMZN", "📱 Meta": "META",
        "🧠 NVIDIA": "NVDA", "🏦 JP Morgan": "JPM", "🥤 Coca-Cola": "KO",
        "⚡ Reliance": "RELIANCE.NS", "💻 TCS": "TCS.NS", "🏦 HDFC Bank": "HDFCBANK.NS",
        "🚗 Maruti": "MARUTI.NS", "☕ Asian Paints": "ASIANPAINT.NS",
        "🥇 Gold": "GC=F", "🥈 Silver": "SI=F", "🗻 Platinum": "PL=F",
        "💱 EUR/USD": "EURUSD=X", "💱 GBP/USD": "GBPUSD=X",
        "💱 USD/JPY": "JPY=X", "💱 USD/INR": "INR=X", "💱 DXY Dollar": "DX-Y.NYB",
        "🛢️ Crude WTI": "CL=F", "🛢️ Brent": "BZ=F", "🟠 Copper": "HG=F",
        "😨 VIX": "^VIX", "₿ Bitcoin": "BTC-USD", "Ξ Ethereum": "ETH-USD"})


def tab_news():
    st.markdown("### 📰 News & Sentiment")
    st.caption("Recent headlines mood for the top-scored assets (positive/negative "
               "keyword count). News is live from Yahoo Finance.")
    _build_ticks()
    n_cnt = st.slider("How many assets to scan", 5, 20, 10, key="news_n")
    with st.spinner("Fetching news..."):
        df = fetch_news_sentiment(int(n_cnt))
    if df.empty:
        st.info("No recent news available right now.")
        return
    df = df.sort_values("sentiment", ascending=False)
    colors = df["lean"].map({"Positive": "#26a69a", "Negative": "#ef5350",
                             "Neutral": "#fbc02d"})
    fig = go.Figure(go.Bar(x=df["sentiment"], y=df["asset"], orientation="h",
                           marker_color=colors, text=df["lean"]))
    fig.update_layout(template="plotly_dark", height=max(320, len(df) * 32),
                      xaxis_title="Sentiment score (-1 negative … +1 positive)",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')
    show = df[["asset", "pos", "neg", "sentiment", "lean", "headlines"]]
    st.dataframe(show, width='stretch')


# ---------------------------------------------------------------- tab 11 52wk
def tab_week52():
    st.markdown("### 🏆 52-Week Scan")
    st.caption("Which assets sit near their 52-week high (breakout potential 🔥) or "
               "deep near their 52-week low (dip watch 🎯).")
    rows = []
    for col in PX.columns:
        c = PX[col].dropna().tail(270)
        if len(c) < 60:
            continue
        hi = c.max()
        lo = c.min()
        price = c.iloc[-1]
        dist_hi = (price - hi) / hi
        dist_lo = (price - lo) / lo
        rng = hi - lo if hi != lo else 0
        pos52 = (price - lo) / rng if rng else 0.5
        rows.append({"asset": col, "cat": CAT_IND.get(col), "price": price,
                     "52w_high": hi, "52w_low": lo, "dist_from_high%": dist_hi * 100,
                     "dist_from_low%": dist_lo * 100, "pos52": pos52 * 100})
    df = pd.DataFrame(rows)
    near_high = df[df["pos52"] >= 92].sort_values("pos52", ascending=False)
    near_low = df[df["pos52"] <= 8].sort_values("pos52")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔥 Near 52-week high (breakout)")
        if len(near_high):
            st.dataframe(near_high[["asset", "cat", "price", "dist_from_high%"]]
                         .rename(columns={"asset": "Asset", "cat": "Category",
                                          "price": "Price",
                                          "dist_from_high%": "% off high"})
                         .round(2), width='stretch')
        else:
            st.caption("No asset within 8% of 52-week high.")
    with c2:
        st.markdown("#### 🎯 Near 52-week low (dip watch)")
        if len(near_low):
            st.dataframe(near_low[["asset", "cat", "price", "dist_from_low%"]]
                         .rename(columns={"asset": "Asset", "cat": "Category",
                                          "price": "Price",
                                          "dist_from_low%": "% above low"})
                         .round(2), width='stretch')
        else:
            st.caption("No asset within 8% of 52-week low.")
    st.markdown("#### Interactive 52-week position")
    sel = st.multiselect("Select assets", df["asset"].tolist(),
                         default=df["asset"].tolist()[:14], key="w52sel")
    if sel:
        fig = go.Figure()
        for a in sel:
            r = df[df["asset"] == a].iloc[0]
            fig.add_trace(go.Bar(x=[a], y=[r["pos52"]], name=a,
                                 marker_color=EMOJI_SCORE(r["pos52"])))
        fig.update_layout(template="plotly_dark", height=400,
                          yaxis=dict(range=[0, 100]),
                          title=dict(text="52-week price position (% of range)",
                                     font_size=13),
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')


def EMOJI_SCORE(v):
    return "#26a69a" if v >= 60 else "#fbc02d" if v >= 40 else "#ef5350"


# ---------------------------------------------------------------- tab 12 heatmap
def tab_heatmap():
    st.markdown("### 🌐 Global Heatmap")
    cats = ["All"] + sorted(SIG["category"].unique().tolist())
    fc = st.selectbox("Category", cats, key="hm_cat")
    data = SIG if fc == "All" else SIG[SIG["category"] == fc]
    fig = px.treemap(data, path=["category", "asset"], values="score",
                     color="label",
                     color_discrete_map=LABEL_COLORS,
                     title=None)
    fig.update_traces(textinfo="label+value", textfont_size=14)
    fig.update_layout(template="plotly_dark", height=560,
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')
    st.markdown("#### Sector / country allocation")
    data = data.copy()
    data["country"] = data["asset"].apply(country_of)
    alloc = data.groupby(["country", "label"]).size().reset_index(name="count")
    fig2 = px.bar(alloc, x="country", y="count", color="label",
                  color_discrete_map=LABEL_COLORS)
    fig2.update_layout(template="plotly_dark", height=360,
                       title=dict(text="Assets by region/sector", font_size=13),
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width='stretch')
    st.caption("Treemap: box size = signal score, colour = Strong Buy → Strong Sell.")


COUNTRY_MAP = None


def country_of(asset):
    global COUNTRY_MAP
    if COUNTRY_MAP is None:
        COUNTRY_MAP = {
            "🇺🇸 S&P500": "US", "🇺🇸 NASDAQ100": "US", "🇺🇸 Dow": "US",
            "🇮🇳 NIFTY50": "India", "🇮🇳 SENSEX": "India", "🇬🇧 FTSE100": "Europe",
            "🇩🇪 DAX": "Europe", "🇫🇷 CAC40": "Europe", "🇯🇵 Nikkei225": "Japan",
            "🇭🇰 HangSeng": "Asia", "🇨🇳 Shanghai": "China", "🇧🇷 Bovespa": "LatAm",
            "🇨🇦 TSX": "Canada", "🇦🇺 ASX200": "Australia",
            "🍎 Apple": "US", "🚀 Tesla": "US", "🖥️ Microsoft": "US",
            "🔍 Google": "US", "🛒 Amazon": "US", "📱 Meta": "US", "🧠 NVIDIA": "US",
            "🏦 JP Morgan": "US", "🥤 Coca-Cola": "US", "⚡ Reliance": "India",
            "💻 TCS": "India", "🏦 HDFC Bank": "India", "🚗 Maruti": "India",
            "☕ Asian Paints": "India",
            "🥇 Gold": "Metals", "🥈 Silver": "Metals", "🗻 Platinum": "Metals",
            "💱 EUR/USD": "FX", "💱 GBP/USD": "FX", "💱 USD/JPY": "FX",
            "💱 USD/INR": "FX", "💱 DXY Dollar": "FX",
            "🛢️ Crude WTI": "Commodity", "🛢️ Brent": "Commodity", "🟠 Copper": "Commodity",
            "😨 VIX": "Volatility", "₿ Bitcoin": "Crypto", "Ξ Ethereum": "Crypto"}
    return COUNTRY_MAP.get(asset, "Other")


# ---------------------------------------------------------------- tab 13 deep
def tab_deep():
    st.markdown("### 📊 Deep Analysis")
    st.markdown("#### Correlation matrix (all assets)")
    rets = PX.pct_change().dropna(how="all")
    corr = rets.corr()
    maxcols = 12
    sel = st.multiselect("Select assets for matrix", corr.columns.tolist(),
                         default=corr.columns.tolist()[:maxcols], key="corr_sel")
    if len(sel) >= 2:
        m = corr.loc[sel, sel].dropna(how="all").dropna(axis=1, how="all")
        hm = px.imshow(m, text_auto=".2f", color_continuous_scale="RdYlGn",
                       zmin=-0.2, zmax=1, aspect="auto")
        hm.update_layout(template="plotly_dark", height=600,
                         title=dict(text="Pairwise return correlation", font_size=13),
                         margin=dict(l=10, r=10, t=44, b=10))
        st.plotly_chart(hm, width='stretch')

    st.markdown("#### Market regime detector (200-day)")
    st.caption("Bull if price above its 200-day average & making progress; bear if "
               "below; otherwise sideways.")
    regs = []
    for col in PX.columns:
        c = PX[col].dropna().tail(400)
        if len(c) < 200:
            continue
        ma200 = c.rolling(200).mean()
        last = c.iloc[-1]
        ma = ma200.iloc[-1]
        pct_above = (c.iloc[-1] / c.iloc[-1 - 20] - 1) * 100 if len(c) > 240 else 0
        if np.isnan(ma):
            reg = "Sideways"
        elif last > ma and pct_above > 1:
            reg = "Bull"
        elif last < ma and pct_above < -1:
            reg = "Bear"
        else:
            reg = "Sideways"
        regs.append({"asset": col, "cat": CAT_IND.get(col), "regime": reg,
                     "20d%": round(pct_above, 2)})
    rdf = pd.DataFrame(regs)
    for r in ["Bull", "Sideways", "Bear"]:
        sub = rdf[rdf["regime"] == r]
        st.markdown(f"**{r}** — {len(sub)} assets")
    rb = px.histogram(rdf, x="regime", color="regime",
                      color_discrete_map={"Bull": "#26a69a", "Bear": "#ef5350",
                                          "Sideways": "#fbc02d"})
    rb.update_layout(template="plotly_dark", height=320,
                     title=dict(text="Regime distribution", font_size=13),
                     margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(rb, width='stretch')

    st.markdown("#### Sector rotation (3M momentum by category)")
    mom = {}
    lookback = 63
    for col in PX.columns:
        c = PX[col].dropna()
        if len(c) > lookback:
            mom[col] = c.iloc[-1] / c.iloc[-1 - lookback] - 1
    cat_mom = {}
    for col, m in mom.items():
        cat = CAT_IND.get(col, "Other")
        cat_mom.setdefault(cat, []).append(m)
    cm = pd.DataFrame({"category": [k for k in cat_mom],
                       "avg_3m_mom": [np.mean(v) * 100 for v in cat_mom.values()]})
    cm = cm.sort_values("avg_3m_mom", ascending=False)
    fig = go.Figure(go.Bar(x=cm["avg_3m_mom"], y=cm["category"], orientation="h",
                           marker_color=["#26a69a" if v > 0 else "#ef5350"
                                         for v in cm["avg_3m_mom"]]))
    fig.update_layout(template="plotly_dark", height=360,
                      xaxis_title="Avg 3-month momentum %",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------- PDF export
def export_pdf():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    buf = None
    try:
        import io
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            # table page
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            tbl = SIG[["asset", "category", "price", "score", "label"]].copy()
            tbl.columns = ["Asset", "Category", "Price", "Score", "Signal"]
            tab = ax.table(cellText=tbl.values, colLabels=tbl.columns,
                           loc="center", cellLoc="left")
            tab.auto_set_font_size(False)
            tab.set_fontsize(7)
            tab.scale(1, 1.4)
            ax.set_title("Global Multi-Asset Advisor Report", fontsize=14, pad=20)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            # score distribution page
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(SIG["score"], bins=20, color="#42a5f5")
            ax.set_title("Signal score distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            pdf.savefig(fig)
            plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        st.error(f"PDF export failed: {e}")
        return None


# ------------------------------------------------------------------ main
def main():
    global SIG, BT, PX, HAVE_DATA
    advisor_header()
    with st.sidebar:
        st.markdown("### 🌗 Theme")
        theme = st.radio("Theme", ["Dark", "Light"], horizontal=True,
                         index=0, key="theme")
        if theme == "Light":
            st.markdown("<style>section.main{filter:invert(1) hue-rotate(180deg);"
                        "}</style>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Data")
        last_date = PX.index.max().date() if HAVE_DATA and len(PX) else None
        st.caption(f"Prices through **{last_date}**")
        if st.button("🔄 Refresh live data", use_container_width=True):
            with st.spinner("Fetching fresh market data..."):
                read_outputs.clear()
                live_fetch.clear()
                try:
                    globals_updated = load_data()
                    SIG, BT, PX = globals_updated
                    HAVE_DATA = True
                    _build_cat()
                    st.success("Updated with live prices!")
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
        st.markdown("### 📄 Report")
        if st.button("📄 Download PDF report", use_container_width=True):
            pdf = export_pdf()
            if pdf is not None:
                st.download_button("Save report.pdf", pdf, "advisor_report.pdf",
                                   "application/pdf")
    if not HAVE_DATA:
        return
    _build_cat()
    _build_ticks()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(
        ["📊 Advisor", "📈 Technicals", "📜 Backtest", "🧭 Context",
         "📈 Performance", "💼 Portfolio", "🎯 Presets", "🏅 Compare",
         "🔥 Market Move", "📰 News", "🏆 52-Week", "🌐 Heatmap", "📊 Deep"])
    with tab1:
        tab_advisor()
    with tab2:
        tab_chart()
    with tab3:
        tab_backtest()
    with tab4:
        tab_context()
    with tab5:
        tab_performance()
    with tab6:
        tab_portfolio()
    with tab7:
        tab_strategy()
    with tab8:
        tab_private()
    with tab9:
        tab_move()
    with tab10:
        tab_news()
    with tab11:
        tab_week52()
    with tab12:
        tab_heatmap()
    with tab13:
        tab_deep()


if __name__ == "__main__":
    main()
