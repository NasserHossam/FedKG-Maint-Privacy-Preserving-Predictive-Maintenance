"""
Step 6: Azure Predictive Maintenance -- federated failure prediction (next-24h failure).

Factories = machine models. Clients: model1, model2, model3.  Held-out factory: model4.
Per client, time split: train < Sep 2015 | val Sep-Oct | test Nov-Dec.
Held-out factory is evaluated on ALL its data (it is never used for training/selection).
Normalization statistics are computed locally in each factory.

Setups: local-only (each client) | centralized (pooled) | FedAvg | FedProx
Metrics: PR-AUC and F1 (threshold chosen on validation data only).

Usage:
    pip install scikit-learn pandas
    python step6_azure.py --data_dir .                 # 1 seed
    python step6_azure.py --data_dir . --seeds 3
"""
import argparse
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from azure_prep import load_or_build

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_VAL = np.datetime64("2015-09-01T00:00:00").astype("datetime64[s]").astype("int64")
T_TEST = np.datetime64("2015-11-01T00:00:00").astype("datetime64[s]").astype("int64")
NEG_KEEP = 0.10   # keep 10% of negative TRAIN rows (all positives kept); eval sets stay untouched


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


# ---------------- metrics ----------------
@torch.no_grad()
def probs(model, X, bs=8192):
    model.eval()
    out = [torch.sigmoid(model(torch.tensor(X[i:i + bs], device=DEVICE))).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[np.argmax(f1)]) if len(th) else 0.5


def f1_at(y, p, thr):
    pred = p >= thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float(((~pred) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def ap(y, p):
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


# ---------------- data ----------------
def prepare(d, clients, holdout, rng):
    X, y, hour, model = d["X"], d["y"], d["hour"], d["model"]
    data = {}
    for name in clients:
        idx = np.where(model == name)[0]
        tr = idx[hour[idx] < T_VAL]
        mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6          # LOCAL statistics
        Xc = ((X[idx] - mean) / std).astype(np.float32)
        h, yc = hour[idx], y[idx]
        trm, vam, tem = h < T_VAL, (h >= T_VAL) & (h < T_TEST), h >= T_TEST
        # subsample negatives of the training period only
        keep = np.where(trm & ((yc == 1) | (rng.random(len(yc)) < NEG_KEEP)))[0]
        data[name] = dict(Xtr=Xc[keep], ytr=yc[keep].astype(np.float32),
                          Xva=Xc[vam], yva=yc[vam], Xte=Xc[tem], yte=yc[tem])
    idx = np.where(model == holdout)[0]
    mean, std = X[idx].mean(0), X[idx].std(0) + 1e-6             # label-free local statistics
    data[holdout] = dict(X=((X[idx] - mean) / std).astype(np.float32), y=y[idx])
    return data


# ---------------- training ----------------
def fit(model, X, y, epochs, lr, bs, mu=0.0, ref=None):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            if mu > 0:
                loss = loss + (mu / 2) * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), ref))
            loss.backward(); opt.step()
    return model


def train_standard(Xtr, ytr, Xva, yva, epochs, lr, bs):
    model = MLP(Xtr.shape[1]).to(DEVICE)
    best, state = -1, None
    for _ in range(epochs):
        fit(model, Xtr, ytr, 1, lr, bs)
        s = np.nan_to_num(ap(yva, probs(model, Xva)))
        if s > best:
            best, state = s, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model


def train_federated(data, clients, strategy, rounds, local_epochs, lr, bs, mu):
    n_in = data[clients[0]]["Xtr"].shape[1]
    g = MLP(n_in).to(DEVICE)
    sizes = [len(data[c]["Xtr"]) for c in clients]
    Xva = np.concatenate([data[c]["Xva"] for c in clients])
    yva = np.concatenate([data[c]["yva"] for c in clients])
    best, state = -1, None
    n_params = sum(p.numel() for p in g.parameters())
    for r in range(rounds):
        ref = [p.detach().clone() for p in g.parameters()]
        states = []
        for c in clients:
            m = copy.deepcopy(g)
            fit(m, data[c]["Xtr"], data[c]["ytr"], local_epochs, lr, bs,
                mu if strategy == "fedprox" else 0.0, ref)
            states.append({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
        tot = float(sum(sizes))
        g.load_state_dict({k: sum(s[k] * (n / tot) for s, n in zip(states, sizes)) for k in states[0]})
        s = np.nan_to_num(ap(yva, probs(g, Xva)))
        if s > best:
            best, state = s, copy.deepcopy(g.state_dict())
    g.load_state_dict(state)
    return g, 2 * len(clients) * n_params * 4 * rounds / 1e6


def evaluate_global(model, data, clients, holdout):
    """Global model: threshold from pooled client validation; own = mean over client test sets."""
    Xva = np.concatenate([data[c]["Xva"] for c in clients])
    yva = np.concatenate([data[c]["yva"] for c in clients])
    thr = best_threshold(yva, probs(model, Xva))
    own_ap, own_f1 = [], []
    for c in clients:
        p = probs(model, data[c]["Xte"])
        own_ap.append(ap(data[c]["yte"], p)); own_f1.append(f1_at(data[c]["yte"], p, thr))
    ph = probs(model, data[holdout]["X"])
    return dict(own_prauc=np.nanmean(own_ap), own_f1=np.mean(own_f1),
                heldout_prauc=ap(data[holdout]["y"], ph), heldout_f1=f1_at(data[holdout]["y"], ph, thr))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data_dir", default=".")
    ap_.add_argument("--holdout", default="model4")
    ap_.add_argument("--epochs", type=int, default=15)
    ap_.add_argument("--rounds", type=int, default=25)
    ap_.add_argument("--local_epochs", type=int, default=1)
    ap_.add_argument("--mu", type=float, default=0.01)
    ap_.add_argument("--lr", type=float, default=1e-3)
    ap_.add_argument("--bs", type=int, default=512)
    ap_.add_argument("--seeds", type=int, default=1)
    ap_.add_argument("--drop", nargs="*", default=[], help="drop feature groups by name prefix, e.g. days_since error age")
    ap_.add_argument("--out", default="results/step6_azure.csv")
    a = ap_.parse_args()

    print(f"Device: {DEVICE}")
    d = load_or_build(a.data_dir)
    if a.drop:
        names = [str(n) for n in d["feature_names"]]
        keep = [i for i, n in enumerate(names) if not any(n.startswith(p) for p in a.drop)]
        print(f"Dropping feature groups {a.drop}: {len(names)} -> {len(keep)} features")
        d = dict(d); d["X"] = d["X"][:, keep]; d["feature_names"] = np.array(names)[keep]
    models = sorted(set(d["model"]))
    clients = [m for m in models if m != a.holdout]
    print(f"Clients: {clients} | held-out factory: {a.holdout}")
    print(f"Positive rate overall: {100 * d['y'].mean():.2f}%")

    rows = []
    for seed in range(a.seeds):
        set_seed(seed)
        data = prepare(d, clients, a.holdout, np.random.default_rng(seed))

        # ---- local-only ----
        for c in clients:
            set_seed(seed)
            m = train_standard(data[c]["Xtr"], data[c]["ytr"], data[c]["Xva"], data[c]["yva"],
                               a.epochs, a.lr, a.bs)
            thr = best_threshold(data[c]["yva"], probs(m, data[c]["Xva"]))
            po, ph = probs(m, data[c]["Xte"]), probs(m, data[a.holdout]["X"])
            rows.append(dict(setup=f"local-only ({c})", seed=seed,
                             own_prauc=ap(data[c]["yte"], po), own_f1=f1_at(data[c]["yte"], po, thr),
                             heldout_prauc=ap(data[a.holdout]["y"], ph),
                             heldout_f1=f1_at(data[a.holdout]["y"], ph, thr), comm_mb=0.0))
            print(f"[seed {seed}] local-only {c}: held-out PR-AUC {rows[-1]['heldout_prauc']:.3f}")

        # ---- centralized ----
        set_seed(seed)
        Xtr = np.concatenate([data[c]["Xtr"] for c in clients]); ytr = np.concatenate([data[c]["ytr"] for c in clients])
        Xva = np.concatenate([data[c]["Xva"] for c in clients]); yva = np.concatenate([data[c]["yva"] for c in clients])
        m = train_standard(Xtr, ytr, Xva, yva, a.epochs, a.lr, a.bs)
        rows.append(dict(setup="centralized (pooled)", seed=seed, comm_mb=0.0,
                         **evaluate_global(m, data, clients, a.holdout)))
        print(f"[seed {seed}] centralized: held-out PR-AUC {rows[-1]['heldout_prauc']:.3f}")

        # ---- federated ----
        for strat in ["fedavg", "fedprox"]:
            set_seed(seed)
            m, comm = train_federated(data, clients, strat, a.rounds, a.local_epochs, a.lr, a.bs, a.mu)
            rows.append(dict(setup=strat if strat == "fedavg" else f"fedprox (mu={a.mu})", seed=seed,
                             comm_mb=comm, **evaluate_global(m, data, clients, a.holdout)))
            print(f"[seed {seed}] {strat}: held-out PR-AUC {rows[-1]['heldout_prauc']:.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    summ = df.groupby("setup")[["own_prauc", "own_f1", "heldout_prauc", "heldout_f1", "comm_mb"]].mean().round(3)
    print("\n================ AZURE: SUMMARY (mean over seeds) ================")
    print(f"own_* = clients' test period (Nov-Dec) | heldout_* = unseen factory {a.holdout}\n")
    print(summ.to_string())
    print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
