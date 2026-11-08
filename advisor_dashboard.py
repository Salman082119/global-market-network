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
                   page_icon="🌍", initial_sidebar_state="expanded")

st.markdown(
    "<style>"
    "@media (max-width: 640px){"
    ".block-container{padding:1rem .6rem 2rem;}"
    "h1,h2,h3{line-height:1.2;}"
    "div[data-testid='stMetricValue']{font-size:1.2rem;}"
    "}"
    "</style>", unsafe_allow_html=True)

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


# ---------------------------------------------------------------- analytics
TRADING_DAYS = 252


def _returns(col):
    return PX[col].dropna().pct_change().dropna()


def _ann_return(r):
    if len(r) < 20:
        return np.nan
    growth = float((1 + r).prod())
    years = len(r) / TRADING_DAYS
    if growth <= 0 or years <= 0:
        return np.nan
    return growth ** (1 / years) - 1


def risk_metrics(col):
    """Annualised return/vol, Sharpe, Sortino and max drawdown for one asset."""
    r = _returns(col)
    if len(r) < 30:
        return None
    c = PX[col].dropna()
    ann_ret = _ann_return(r)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    downside = r[r < 0]
    dvol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 2 else np.nan
    sortino = ann_ret / dvol if dvol else np.nan
    mdd = float((c / c.cummax() - 1).min())
    return {"AnnRet": ann_ret, "Vol": ann_vol, "Sharpe": sharpe,
            "Sortino": sortino, "MaxDD": mdd}


def portfolio_risk(weights):
    """Annualised metrics for a weighted portfolio. weights: asset -> weight."""
    cols = [a for a in weights if a in PX.columns and weights[a] > 0]
    if not cols:
        return None
    rets = PX[cols].pct_change().dropna(how="all").fillna(0.0)
    w = np.array([weights[a] for a in cols], dtype=float)
    w = w / w.sum()
    pr = rets.mul(w, axis=1).sum(axis=1)
    ann_ret = _ann_return(pr)
    ann_vol = float(pr.std() * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    downside = pr[pr < 0]
    dvol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 2 else np.nan
    sortino = ann_ret / dvol if dvol else np.nan
    eq = (1 + pr).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return {"AnnRet": ann_ret, "Vol": ann_vol, "Sharpe": sharpe,
            "Sortino": sortino, "MaxDD": mdd}


def bollinger(close, n=20, k=2):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return ma, ma + k * sd, ma - k * sd


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv(ticker):
    """Cached OHLCV download used for ATR / ADX / volume overlays."""
    import yfinance as yf
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].sort_index()


def adx_atr(ohlcv, n=14):
    high, low, close = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    up, dn = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return plus_di, minus_di, adx, atr


def explain_score(row):
    """Rule-based natural-language explanation of a 0-100 signal score."""
    comps = [("Trend", row["trend"]), ("RSI / momentum", row["rsi"]),
             ("MACD", row["macd"]), ("Volatility", row["vol"]),
             ("52-week position", row["pos"])]
    notes = []
    t = row["trend"]
    if t >= 90:
        notes.append("**Trend is bullish** — price is above both EMA50 and EMA200.")
    elif t >= 70:
        notes.append("**Trend is constructive** — price is above EMA50.")
    elif t >= 35:
        notes.append("**Trend is mixed** — price is below EMA50 but above EMA200.")
    else:
        notes.append("**Trend is bearish** — price is below its key moving averages.")
    rv = row.get("rsi_raw", np.nan)
    if np.isfinite(rv):
        if rv < 30:
            notes.append(f"**RSI {rv:.0f}** — oversold, often a mean-reversion buy zone.")
        elif rv > 70:
            notes.append(f"**RSI {rv:.0f}** — overbought, momentum may be stretched.")
        else:
            notes.append(f"**RSI {rv:.0f}** — neutral momentum range.")
    if row["macd"] >= 65:
        notes.append("**MACD is positive** — short-term momentum leads the signal line.")
    else:
        notes.append("**MACD is weak/negative** — momentum has not confirmed the move.")
    vr = row.get("vol_ratio", np.nan)
    if np.isfinite(vr):
        if vr < 1.2:
            notes.append(f"**Volatility is contained** ({vr:.2f}× its own average).")
        elif vr < 1.6:
            notes.append(f"**Volatility is elevated** ({vr:.2f}× its own average).")
        else:
            notes.append(f"**Volatility is high** ({vr:.2f}× its own average) — wider risk.")
    p = row["pos"]
    zone = "near the high (breakout zone)" if p >= 70 else (
        "mid-range" if p >= 30 else "near the low (dip zone)")
    notes.append(f"**52-week range position {p:.0f}/100** — {zone}.")
    best = max(comps, key=lambda x: x[1])[0]
    worst = min(comps, key=lambda x: x[1])[0]
    head = (f"Composite score **{row['score']:.1f}/100 → {row['label']}**. "
            f"Strongest component: **{best}**; weakest: **{worst}**.")
    return head, notes


def walk_forward_table(bt, years=3):
    """Split the backtest into rolling windows and score each out-of-sample block."""
    if bt is None or len(bt) < 12:
        return pd.DataFrame()
    df = bt.copy()
    df["date"] = pd.to_datetime(df["date"])
    start, end = df["date"].min(), df["date"].max()
    rows = []
    s = start
    while s < end:
        e = min(s + pd.DateOffset(years=years), end + pd.Timedelta(days=1))
        sub = df[(df["date"] >= s) & (df["date"] < e)]
        if len(sub) >= 6:
            sr = float((1 + sub["strategy"]).prod() - 1)
            br = float((1 + sub["benchmark"]).prod() - 1)
            sd = sub["strategy"].std()
            shp = (sub["strategy"].mean() / sd * np.sqrt(12)) if sd else np.nan
            rows.append({"Window": f"{s.date()} → {(e - pd.Timedelta(days=1)).date()}",
                         "Strategy %": sr * 100, "Benchmark %": br * 100,
                         "Excess %": (sr - br) * 100, "Sharpe (strat)": shp,
                         "Months": len(sub)})
        s = e
    return pd.DataFrame(rows)


def weight_sensitivity(n=200, jitter=0.5):
    """Perturb the 5 component weights and measure top-10 pick stability."""
    base = {"trend": 0.22, "rsi": 0.22, "macd": 0.20, "vol": 0.18, "pos": 0.18}
    base_top = set(SIG.nlargest(10, "score")["asset"])
    rng = np.random.default_rng(42)
    overlaps = []
    for _ in range(n):
        w = {k: max(0.01, base[k] * (1 + rng.uniform(-jitter, jitter))) for k in base}
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        s = (SIG["trend"] * w["trend"] + SIG["rsi"] * w["rsi"] +
             SIG["macd"] * w["macd"] + SIG["vol"] * w["vol"] +
             SIG["pos"] * w["pos"])
        top = set(SIG.assign(_s=s).nlargest(10, "_s")["asset"])
        overlaps.append(len(base_top & top) / 10)
    return np.array(overlaps)


def record_snapshot():
    st.session_state["sig_snapshot"] = dict(zip(SIG["asset"], SIG["label"]))
    st.session_state["snapshot_time"] = pd.Timestamp.now()


def signal_changes():
    snap = st.session_state.get("sig_snapshot")
    if not snap:
        return pd.DataFrame()
    now = dict(zip(SIG["asset"], SIG["label"]))
    rows = [{"asset": a, "old": snap[a], "now": now[a]}
            for a in now if a in snap and snap[a] != now[a]]
    return pd.DataFrame(rows)


def quick_update():
    """Lightweight live update: pull the last month for every ticker, merge,
    re-score and rewrite the CSVs (much faster than a full re-download)."""
    import yfinance as yf
    from multi_asset_advisor import UNIVERSE, CATEGORY, score_asset, backtest
    raw = yf.download(list(UNIVERSE.values()), period="1mo", interval="1d",
                      progress=False, auto_adjust=False, group_by="ticker",
                      threads=True)
    latest = {}
    for name, tick in UNIVERSE.items():
        try:
            sub = raw[tick] if isinstance(raw.columns, pd.MultiIndex) else raw
            col = ("Adj Close" if "Adj Close" in sub.columns and
                   sub["Adj Close"].notna().any() else "Close")
            s = sub[col].dropna()
            if len(s):
                latest[name] = s
        except Exception:
            continue
    if not latest:
        raise RuntimeError("no live prices downloaded")
    upd = pd.DataFrame(latest)
    upd.index = pd.to_datetime(upd.index).tz_localize(None)
    combined = pd.concat([PX, upd]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")].ffill()
    rows = []
    for name in combined.columns:
        b = score_asset(combined[name])
        rows.append({"asset": name, "category": CATEGORY.get(name, "Other"), **b})
    sig = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    bt = backtest(combined)
    combined.to_csv("advisor_prices.csv")
    sig.to_csv("advisor_signals.csv", index=False)
    bt.to_csv("advisor_backtest.csv", index=False)
    return sig, bt, combined


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
    st.markdown("#### 👀 Watchlist & signal-change alerts")
    all_assets = SIG.sort_values("asset")["asset"].tolist()
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = SIG.head(5)["asset"].tolist()
    watch = st.multiselect("Watchlist assets", all_assets, key="watchlist")
    if watch:
        wdf = SIG[SIG["asset"].isin(watch)][
            ["asset", "category", "price", "score", "label"]].copy()
        wdf["signal"] = wdf["label"].apply(lambda x: f"{EMOJI[x]} {x}")
        st.dataframe(wdf[["asset", "category", "price", "score", "signal"]]
                     .rename(columns={"asset": "Asset", "category": "Category",
                                      "price": "Price", "score": "Score",
                                      "signal": "Signal"}),
                     width='stretch', hide_index=True)
    ac1, ac2 = st.columns([1, 2])
    with ac1:
        if st.button("📌 Snapshot signals", key="snap_btn",
                     help="Save current labels so later changes are flagged."):
            record_snapshot()
            st.success("Snapshot saved.")
        if st.button("🧹 Clear snapshot", key="snap_clear"):
            st.session_state.pop("sig_snapshot", None)
        if st.session_state.get("snapshot_time") is not None:
            st.caption(f"Snapshot: {st.session_state['snapshot_time']:%d %b %Y %H:%M}")
    with ac2:
        ch = signal_changes()
        if not ch.empty:
            for _, r in ch.iterrows():
                st.warning(f"**{r['asset']}** — signal changed "
                           f"{r['old']} → {r['now']}")
        elif st.session_state.get("sig_snapshot"):
            st.success("No signal changes since the last snapshot.")

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
    oc1, oc2, oc3 = st.columns(3)
    show_bb = oc1.checkbox("Bollinger Bands (20, 2σ)", value=True, key="show_bb")
    show_atr = oc2.checkbox("ATR & ADX", value=True, key="show_atr")
    show_vol = oc3.checkbox("Volume", value=True, key="show_vol")

    fig = go.Figure()
    if show_bb:
        bb_ma, bb_up, bb_lo = bollinger(close)
        fig.add_trace(go.Scatter(x=idx, y=bb_up.loc[idx], name="BB upper",
                                 line=dict(width=1, color="#90caf9", dash="dot")))
        fig.add_trace(go.Scatter(x=idx, y=bb_lo.loc[idx], name="BB lower",
                                 line=dict(width=1, color="#90caf9", dash="dot"),
                                 fill="tonexty", fillcolor="rgba(144,202,249,0.08)"))
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

    tick = UNIVERSE_TICK.get(asset)
    if (show_atr or show_vol) and tick:
        try:
            ohlcv = fetch_ohlcv(tick)
        except Exception:
            ohlcv = pd.DataFrame()
        if ohlcv.empty:
            st.caption("OHLCV data unavailable right now — ATR/ADX/volume skipped.")
        else:
            sub = ohlcv.loc[ohlcv.index >= idx[0]]
            if show_atr and {"High", "Low", "Close"}.issubset(ohlcv.columns):
                pdi, mdi, adxv, atrv = adx_atr(ohlcv)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=sub.index, y=adxv.loc[sub.index],
                                          name="ADX",
                                          line=dict(color="#ffb74d", width=1.4)))
                fig3.add_trace(go.Scatter(x=sub.index, y=pdi.loc[sub.index],
                                          name="+DI",
                                          line=dict(color="#26a69a", width=1, dash="dot")))
                fig3.add_trace(go.Scatter(x=sub.index, y=mdi.loc[sub.index],
                                          name="−DI",
                                          line=dict(color="#ef5350", width=1, dash="dot")))
                fig3.add_hline(y=25, line=dict(color="gray", dash="dot", width=1))
                fig3.update_layout(
                    template="plotly_dark", height=260,
                    title=dict(text=f"ADX / directional indicators · ATR(14) = "
                                    f"{atrv.iloc[-1]:.2f}", font_size=12),
                    margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
                st.plotly_chart(fig3, width='stretch')
            if show_vol and "Volume" in ohlcv.columns and ohlcv["Volume"].notna().any():
                v = ohlcv["Volume"].loc[sub.index]
                vc = np.where(v.pct_change().fillna(0) >= 0, "#26a69a", "#ef5350")
                fig4 = go.Figure(go.Bar(x=sub.index, y=v, marker_color=vc))
                fig4.update_layout(template="plotly_dark", height=220,
                                   title=dict(text="Volume", font_size=12),
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig4, width='stretch')
    elif show_atr or show_vol:
        st.caption("No ticker mapping for this asset — ATR/ADX/volume skipped.")

    with st.expander("🤖 Why this score? (rule-based explanation)"):
        head, notes = explain_score(row)
        st.markdown(head)
        for nt in notes:
            st.markdown(f"- {nt}")
        st.caption("Generated from the five score components — deterministic and "
                   "rule-based, not an AI/LLM model.")


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

    st.markdown("#### Walk-forward windows")
    st.caption("Splits history into sequential out-of-sample blocks and checks "
               "whether the edge is consistent or concentrated in one regime.")
    wf_years = st.slider("Window length (years)", 2, 6, 3, key="wf_years")
    wf = walk_forward_table(BT, wf_years)
    if wf.empty:
        st.info("Not enough backtest history for this window length.")
    else:
        st.dataframe(wf.round(2), width='stretch', hide_index=True)
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(x=wf["Window"], y=wf["Strategy %"],
                                name="Strategy", marker_color="#26a69a"))
        fig_wf.add_trace(go.Bar(x=wf["Window"], y=wf["Benchmark %"],
                                name="Benchmark", marker_color="#fbc02d"))
        fig_wf.update_layout(template="plotly_dark", height=320, barmode="group",
                             title=dict(text="Return by walk-forward window (%)",
                                        font_size=13),
                             yaxis=dict(title="Return %"),
                             margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_wf, width='stretch')
        beat = (wf["Excess %"] > 0).mean() * 100
        st.caption(f"Strategy beat the benchmark in **{beat:.0f}%** of windows "
                   f"({int((wf['Excess %'] > 0).sum())}/{len(wf)}).")

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

    st.markdown("#### 📐 Risk-adjusted metrics (annualised)")
    risk_rows = []
    for col in df.index:
        m = risk_metrics(col)
        if m:
            risk_rows.append({"asset": col, **m})
    if risk_rows:
        rdf = pd.DataFrame(risk_rows).set_index("asset")
        show_cols = [a for a in view.index if a in rdf.index]
        rv = rdf.loc[show_cols].copy()
        for c in ["AnnRet", "Vol", "MaxDD"]:
            rv[c] = rv[c] * 100
        rv = rv.rename(columns={"AnnRet": "Ann. return %", "Vol": "Vol %",
                                "Sharpe": "Sharpe", "Sortino": "Sortino",
                                "MaxDD": "Max DD %"})
        st.dataframe(rv.round(2), width='stretch')
        st.caption("Sharpe/Sortino use daily returns annualised by √252 with a 0% "
                   "risk-free rate. Max DD is measured over full available history.")
        top_sharpe = rdf["Sharpe"].dropna().sort_values(ascending=False).head(5)
        st.markdown("**Best Sharpe ratios:** " +
                    ", ".join(f"{a} ({v:.2f})" for a, v in top_sharpe.items()))
        st.download_button("📥 Download risk metrics CSV", rv.to_csv().encode(),
                           "advisor_risk_metrics.csv", "text/csv", key="dl_risk")


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
    weights = {}
    for r in top:
        if risk == "Conservative":
            share = 1.0 / n
        else:
            share = r["score"] / total_score
        allocate = round(budget * share, 0)
        weights[r["asset"]] = share
        rows.append({"Asset": r["asset"], "Category": r["category"],
                     "Signal": r["label"], "Score": r["score"],
                     "Weight": f"{share*100:.1f}%",
                     "Suggested ₹": int(allocate)})
    port_df = pd.DataFrame(rows)
    st.markdown(f"*{risk} strategy — invest across {len(top)} assets, "
                f"{'higher weight to stronger scores' if risk != 'Conservative' else 'equal weight across all'}.*")
    st.dataframe(port_df, width='stretch')
    pm = portfolio_risk(weights)
    if pm:
        st.markdown("#### 📐 Portfolio risk snapshot")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Ann. return", f"{pm['AnnRet']*100:.1f}%")
        p2.metric("Ann. volatility", f"{pm['Vol']*100:.1f}%")
        p3.metric("Sharpe", f"{pm['Sharpe']:.2f}")
        p4.metric("Max drawdown", f"{pm['MaxDD']*100:.1f}%")
        st.caption("Computed on daily returns of the suggested weights "
                   "(cash buffer excluded). Historical, not a forecast.")
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

    st.markdown("#### 🎛️ Signal weight sensitivity")
    st.caption("Randomly perturbs the 5 component weights 200 times and measures "
               "how often the baseline top-10 picks survive. High overlap = robust.")
    with st.spinner("Running weight perturbations..."):
        overlaps = weight_sensitivity()
    avg_overlap = float(overlaps.mean()) * 100
    st.metric("Top-10 overlap with baseline", f"{avg_overlap:.0f}%")
    verdict = ("Very robust" if avg_overlap >= 85 else
               "Reasonably robust" if avg_overlap >= 70 else
               "Weight-sensitive" if avg_overlap >= 50 else "Fragile")
    st.markdown(f"**Robustness verdict: {verdict}**")
    hist = go.Figure(go.Histogram(x=overlaps * 100, nbinsx=20,
                                  marker_color="#42a5f5"))
    hist.update_layout(template="plotly_dark", height=280,
                       title=dict(text="Distribution of top-10 overlap (%)",
                                  font_size=13),
                       xaxis_title="Overlap %", yaxis_title="Count",
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist, width='stretch')
    st.caption("A 50% jitter is applied to each weight before renormalising.")


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
        if st.button("⚡ Quick price update", use_container_width=True,
                     help="Faster: pulls the last month for all tickers and re-scores."):
            with st.spinner("Pulling latest quotes..."):
                try:
                    SIG, BT, PX = quick_update()
                    HAVE_DATA = True
                    _build_cat()
                    read_outputs.clear()
                    st.success("Quick update done.")
                except Exception as e:
                    st.error(f"Quick update failed: {e}")
        st.markdown("### 🔄 Auto-refresh")
        auto = st.toggle("Enable auto-refresh", value=False, key="auto_refresh")
        st.slider("Every (seconds)", 30, 300, 60, 30, key="auto_interval",
                  disabled=not auto)
        if auto:
            st.caption("Pulls latest quotes and reruns on each cycle.")
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

    if st.session_state.get("auto_refresh"):
        import time
        time.sleep(int(st.session_state.get("auto_interval", 60)))
        try:
            SIG, BT, PX = quick_update()
            HAVE_DATA = True
            _build_cat()
            read_outputs.clear()
        except Exception:
            pass
        st.rerun()


if __name__ == "__main__":
    main()
