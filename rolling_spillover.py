"""
Rolling Diebold-Yilmaz Spillover Index (Phase 7)
=================================================
Recomputes the total spillover index on a rolling window (VAR with AIC lag,
generalized FEVD, H=10) to trace how contagion transmission evolved through
time (GFC, COVID, rate-hike cycle...).

Outputs:
    phase7_rolling_spillover.csv  (date, total_spillover)
    phase7_rolling_spillover.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

ROWS = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
ROWS = ROWS.ffill().dropna()
ROLL = 500          # trailing trading days per window (~2 years)
STEP = 21           # one month advance
H = 10              # FEVD horizon
MAXLAG = 8

WINDOWS = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "China deval.": ("2015-08-10", "2016-02-11"),
    "COVID": ("2020-02-19", "2020-03-23"),
    "Rate-hike bear": ("2022-01-03", "2022-10-12"),
}
WINDOW_COLORS = {"GFC": "#ef5350", "China deval.": "#ab47bc",
                 "COVID": "#ffa726", "Rate-hike bear": "#8d6e63"}


def spillover_index(df: pd.DataFrame, h=H, maxlag=MAXLAG):
    try:
        model = VAR(df)
        sel = model.select_order(maxlags=maxlag)
        n = int(sel.aic if np.isscalar(sel.aic) else np.min(sel.aic))
        n = max(1, min(n, MAXLAG))
        res = model.fit(n)
        fevd = res.fevd(h)
        d = np.asarray(fevd.decomp)        # (k, steps, k): d[to, step, from]
        S = d[:, -1, :].T                  # (k, k) S[from, to] at last horizon
        col_sum = S.sum(axis=0)
        col_sum[col_sum == 0] = 1e-12
        S = S / col_sum
        k = S.shape[0]
        total = S[~np.eye(k, dtype=bool)].sum() / k * 100
        return float(total)
    except Exception:
        return np.nan


def main():
    idx = ROWS.index
    dates, totals = [], []
    for i in range(ROLL, len(idx), STEP):
        win = ROWS.iloc[i - ROLL:i]
        if len(win) < ROLL:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = spillover_index(win)
        if np.isfinite(t):
            dates.append(idx[i])
            totals.append(t)

    out = pd.DataFrame({"date": dates, "total_spillover": totals})
    out.to_csv("phase7_rolling_spillover.csv", index=False)
    print(f"Rolling spillover: {len(out)} windows "
          f"({out['date'].iloc[0]:%Y-%m-%d} -> {out['date'].iloc[-1]:%Y-%m-%d})")

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=110)
    ax.plot(out["date"], out["total_spillover"], color="#1f77b4", lw=1.6, label="Total spillover index (%)")
    for lbl, (s, e) in WINDOWS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.2,
                   color=WINDOW_COLORS[lbl], label=lbl)
    ax.axhline(out["total_spillover"].mean(), ls="--", lw=0.8, color="#555", label="Mean")
    ax.set_title("Rolling Diebold-Yilmaz Total Spillover Index")
    ax.set_ylabel("Spillover (%)")
    ax.legend(fontsize=8, ncol=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("phase7_rolling_spillover.png", dpi=110)
    plt.close(fig)
    print("Saved phase7_rolling_spillover.csv + .png")


if __name__ == "__main__":
    main()
