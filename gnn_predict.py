"""
Global Market Network - Phase 8: GNN Contagion Prediction
==========================================================
Graph Neural Network that predicts next-month correlations for all 66 index
pairs from the previous month's correlation network.

Task      : edge regression (66 pairs per monthly graph snapshot)
Model     : 2x GATv2Conv (edge-weighted) + MLP pair decoder
Baselines : persistence (next = current) and closed-form Ridge regression
Split     : temporal - train <= Jun-2019, val Jul-2019..Jun-2021 (COVID),
            test >= Jul-2021

Outputs:
    phase8_gnn_predictions.csv, phase8_gnn_metrics.csv,
    phase8_pred_vs_actual.png

Usage:
    python gnn_predict.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

SEED = 42
HIDDEN = 32
DROPOUT = 0.2
LR = 3e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 400
PATIENCE = 40
BATCH = 16
LOOKBACK = 90
FWD = 21
TRAIN_END = pd.Timestamp("2019-06-30")
VAL_END = pd.Timestamp("2021-06-30")

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------------
def build_snapshots(returns: pd.DataFrame):
    idx = returns.index
    month_last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    names = list(returns.columns)
    k = len(names)
    triu_r, triu_c = np.triu_indices(k, k=1)
    pairs = [(names[i], names[j]) for i, j in zip(triu_r, triu_c)]

    ei = torch.tensor(np.stack([np.concatenate([triu_r, triu_c]),
                                np.concatenate([triu_c, triu_r])]), dtype=torch.long)
    snaps = []
    for me in month_last:
        p = idx.get_loc(me)
        if p < LOOKBACK - 1 or p + FWD >= len(idx):
            continue
        corr = returns.iloc[p - LOOKBACK + 1:p + 1].corr().to_numpy()
        if np.isnan(corr).any():
            continue
        y_mat = returns.iloc[p + 1:p + FWD + 1].corr().to_numpy()
        if np.isnan(y_mat).any() or len(returns.iloc[p + 1:p + FWD + 1]) < 15:
            continue
        r30 = returns.iloc[p - 29:p + 1]
        feats = np.stack([r30.mean().to_numpy(),
                          r30.std().to_numpy(),
                          (corr.sum(1) - 1) / (k - 1),
                          (np.abs(corr).sum(1) - 1) / (k - 1)], axis=1)
        snaps.append({
            "date": me,
            "x": feats.astype(np.float32),
            "ew": corr.astype(np.float32),
            "y": y_mat[triu_r, triu_c].astype(np.float32),
            "cur": corr[triu_r, triu_c].astype(np.float32),
            "avg_corr": float(np.nanmean(corr[triu_r, triu_c])),
        })
    return snaps, pairs, ei


def make_data(snap, ei, feat_mu, feat_sd):
    x = (snap["x"] - feat_mu) / feat_sd
    ei_np = ei.numpy()
    ew = snap["ew"][ei_np[0], ei_np[1]].astype(np.float32)
    return Data(x=torch.tensor(x), edge_index=ei,
                edge_attr=torch.tensor(ew).reshape(-1, 1),
                y=torch.tensor(snap["y"]), cur=torch.tensor(snap["cur"]))


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
class PairGNN(nn.Module):
    def __init__(self, n_feat=4, hid=HIDDEN, drop=DROPOUT):
        super().__init__()
        self.c1 = GATv2Conv(n_feat, hid, edge_dim=1, dropout=drop)
        self.c2 = GATv2Conv(hid, hid, edge_dim=1, dropout=drop)
        self.dec = nn.Sequential(nn.Linear(hid * 3, 64), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, data, triu_r, triu_c):
        z = F.elu(self.c1(data.x, data.edge_index, data.edge_attr))
        z = F.elu(self.c2(z, data.edge_index, data.edge_attr))
        zi, zj = z[triu_r], z[triu_c]
        delta = self.dec(torch.cat([zi, zj, zi * zj], dim=-1)).squeeze(-1)
        return torch.clamp(data.cur + 0.5 * torch.tanh(delta), -0.99, 0.99)


def ridge_fit_predict(x_tr, y_tr, x_te, lam=1.0):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
    X = (x_tr - mu) / sd
    A = np.hstack([X, np.ones((len(X), 1))])
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y_tr)
    Xte = (x_te - mu) / sd
    return np.hstack([Xte, np.ones((len(Xte), 1))]) @ W


def metrics(y, yhat):
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    return mae, rmse, r2


# ----------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------
def main():
    returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
    snaps, pairs, ei = build_snapshots(returns)
    dates = [s["date"] for s in snaps]
    print(f"Snapshots: {len(snaps)} ({dates[0]:%b %Y} -> {dates[-1]:%b %Y}), "
          f"{len(pairs)} pairs each")

    tr_idx = [i for i, d in enumerate(dates) if d <= TRAIN_END]
    va_idx = [i for i, d in enumerate(dates) if TRAIN_END < d <= VAL_END]
    te_idx = [i for i, d in enumerate(dates) if d > VAL_END]
    print(f"Split -> train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")

    all_feats = np.concatenate([snaps[i]["x"] for i in tr_idx])
    feat_mu, feat_sd = all_feats.mean(0), all_feats.std(0) + 1e-9

    data = [make_data(s, ei, feat_mu, feat_sd) for s in snaps]
    k = data[0].x.shape[0]
    triu_r, triu_c = np.triu_indices(k, k=1)
    tr_r, tr_c = torch.tensor(triu_r), torch.tensor(triu_c)

    y_tr_all = np.concatenate([snaps[i]["y"] for i in tr_idx])
    y_va_all = np.concatenate([snaps[i]["y"] for i in va_idx])

    model = PairGNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, bad = np.inf, None, 0
    order = list(tr_idx)
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(order)
        model.train()
        for b in range(0, len(order), BATCH):
            batch = [data[i] for i in order[b:b + BATCH]]
            opt.zero_grad()
            loss = sum(F.mse_loss(model(d, tr_r, tr_c), d.y) for d in batch)
            (loss / len(batch)).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = np.concatenate([model(data[i], tr_r, tr_c).numpy()
                                      for i in va_idx])
        val_mae = float(np.mean(np.abs(va_pred - y_va_all)))
        if val_mae < best_val - 1e-5:
            best_val, bad = val_mae, 0
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | val MAE {val_mae:.4f} (best {best_val:.4f})")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    rows = []
    with torch.no_grad():
        for split, ids in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
            for i in ids:
                gnn = model(data[i], tr_r, tr_c).numpy()
                rows.append(pd.DataFrame({
                    "date": snaps[i]["date"],
                    "pair": [f"{a} <-> {b}" for a, b in pairs],
                    "actual": snaps[i]["y"], "gnn": gnn,
                    "persistence": snaps[i]["cur"],
                }))
    pred_df = pd.concat(rows, ignore_index=True)
    pred_df.to_csv("phase8_gnn_predictions.csv", index=False)

    te = pred_df[pred_df["date"] > VAL_END]
    ridge_feats = {i: np.column_stack([snaps[i]["cur"], np.full(66, snaps[i]["avg_corr"])])
                   for i in range(len(snaps))}
    X_tr = np.concatenate([ridge_feats[i] for i in tr_idx])
    y_tr_pairs = np.concatenate([snaps[i]["y"] for i in tr_idx])
    ridge_te_pred = []
    for i in te_idx:
        ridge_te_pred.append(ridge_fit_predict(X_tr, y_tr_pairs, ridge_feats[i]))
    ridge_te = np.concatenate(ridge_te_pred)

    y_te = te["actual"].to_numpy()
    results = []
    for name, yhat in [("GNN (GATv2)", te["gnn"].to_numpy()),
                       ("Persistence", te["persistence"].to_numpy()),
                       ("Ridge", ridge_te)]:
        mae, rmse, r2 = metrics(y_te, yhat)
        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"  {name:12s} | MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:+.3f}")

    met = pd.DataFrame(results).round(4)
    met.to_csv("phase8_gnn_metrics.csv", index=False)

    avg = (pred_df.groupby(["date"]).agg(actual=("actual", "mean"),
                                         gnn=("gnn", "mean"),
                                         pers=("persistence", "mean"))
           .loc[lambda d: d.index > VAL_END])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(avg.index, avg["actual"], color="black", lw=1.6, label="Actual")
    axes[0].plot(avg.index, avg["gnn"], color="#d62728", lw=1.4, label="GNN predicted")
    axes[0].plot(avg.index, avg["pers"], color="#1f77b4", lw=1.2, ls="--",
                 label="Persistence")
    axes[0].set_title("Next-month avg pairwise correlation - test period", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].scatter(te["actual"], te["gnn"], s=6, alpha=0.35,
                    color="#d62728", edgecolors="none")
    lim = [-1, 1]
    axes[1].plot(lim, lim, color="black", lw=1)
    axes[1].set_xlabel("Actual next-month corr")
    axes[1].set_ylabel("GNN predicted")
    axes[1].set_title(f"Edge-level predictions (test) - "
                      f"MAE {results[0]['MAE']:.3f}", fontsize=12)
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    plt.tight_layout()
    plt.savefig("phase8_pred_vs_actual.png", dpi=150)
    print("Saved phase8_pred_vs_actual.png")
    print("\nMetrics saved to phase8_gnn_metrics.csv | predictions to "
          "phase8_gnn_predictions.csv")


if __name__ == "__main__":
    main()
