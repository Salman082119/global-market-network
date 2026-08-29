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
@st.cache_data(ttl=600)
def load_data():
    sig = pd.read_csv("advisor_signals.csv")
    bt = pd.read_csv("advisor_backtest.csv", parse_dates=["date"])
    px_df = pd.read_csv("advisor_prices.csv", index_col=0, parse_dates=True)
    return sig, bt, px_df


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


# ------------------------------------------------------------------ main
def main():
    advisor_header()
    if not HAVE_DATA:
        return
    _build_cat()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["📊 Buy/Sell Advisor", "📈 Price & Technicals",
         "📜 Backtest & Honesty", "🧭 Market Context",
         "📈 Performance", "💼 Portfolio", "🎯 Strategy Presets", "🏅 Compare"])
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


if __name__ == "__main__":
    main()
