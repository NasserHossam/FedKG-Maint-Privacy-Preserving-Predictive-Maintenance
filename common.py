"""Shared utilities for FedKG-Maint (used by all later steps)."""
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RUL_CAP = 125.0
CLIENTS = ["FD001", "FD002", "FD003"]
HELD_OUT = "FD004"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_factory(fd, proc_dir="processed"):
    d = np.load(os.path.join(proc_dir, f"{fd}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_loader(X, y, batch_size=256, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32) / RUL_CAP)  # targets scaled to [0, 1]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features=17, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------- metrics ----------------
def nasa_score(pred, true):
    d = pred - true                      # d < 0: early prediction, d > 0: late (penalized more)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


@torch.no_grad()
def predict(model, X, batch_size=1024):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) * RUL_CAP


def evaluate(model, X, y):
    p = predict(model, X)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return {"rmse": rmse, "score": nasa_score(p, y)}


# ---------------- centralized-style training (also used by local-only) ----------------
def train_model(Xtr, ytr, Xva, yva, epochs=15, lr=1e-3, batch_size=256, verbose=True):
    """Standard supervised training; keeps the best weights by validation RMSE."""
    model = LSTMRegressor(n_features=Xtr.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = make_loader(Xtr, ytr, batch_size, shuffle=True)

    best_rmse, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = evaluate(model, Xva, yva)
        if val["rmse"] < best_rmse:
            best_rmse, best_state = val["rmse"], copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {ep:2d}/{epochs} | val RMSE {val['rmse']:.2f}")
    model.load_state_dict(best_state)
    return model
