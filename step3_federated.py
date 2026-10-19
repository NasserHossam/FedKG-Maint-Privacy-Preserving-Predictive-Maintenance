"""
Step 3: Federated learning for predictive maintenance.

Clients : FD001, FD002, FD003 (raw data never leaves a client; only weights are exchanged)
Held-out: FD004 (never used for training OR model selection)

Strategies
  fedavg  : clients train locally, server averages weights (weighted by #samples)
  fedprox : same, but local loss adds (mu/2)*||w - w_global||^2 to limit client drift

The best global model is chosen by validation RMSE on the CLIENTS' validation sets only.

Usage:
    python step3_federated.py                         # 1 seed, 20 rounds
    python step3_federated.py --seeds 3 --rounds 25 --mu 0.01
"""
import argparse
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import (CLIENTS, HELD_OUT, DEVICE, LSTMRegressor, set_seed,
                    load_factory, make_loader, evaluate)


def local_update(global_model, X, y, local_epochs, lr, batch_size, mu):
    """Train a copy of the global model on one client's data; return its new weights."""
    model = copy.deepcopy(global_model).to(DEVICE)
    global_params = [p.detach().clone() for p in global_model.to(DEVICE).parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = make_loader(X, y, batch_size, shuffle=True)

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if mu > 0:  # FedProx proximal term
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def aggregate(states, sizes):
    """Weighted average of client weights (FedAvg aggregation)."""
    total = float(sum(sizes))
    avg = {}
    for k in states[0]:
        avg[k] = sum(s[k] * (n / total) for s, n in zip(states, sizes))
    return avg


def run_federated(data, strategy, rounds, local_epochs, lr, batch_size, mu, seed, lr_decay=1.0):
    set_seed(seed)
    g_model = LSTMRegressor(n_features=data[CLIENTS[0]]["X_train"].shape[-1]).to(DEVICE)
    n_params = sum(p.numel() for p in g_model.parameters())
    sizes = [len(data[c]["X_train"]) for c in CLIENTS]

    # pooled VALIDATION set of the clients (used only for model selection)
    Xva = np.concatenate([data[c]["X_val"] for c in CLIENTS])
    yva = np.concatenate([data[c]["y_val"] for c in CLIENTS])

    best_val, best_state, best_round = float("inf"), None, 0
    history = []
    prox_mu = mu if strategy == "fedprox" else 0.0

    for r in range(1, rounds + 1):
        states = [local_update(g_model, data[c]["X_train"], data[c]["y_train"],
                               local_epochs, lr * (lr_decay ** (r - 1)), batch_size, prox_mu)
                  for c in CLIENTS]
        g_model.load_state_dict(aggregate(states, sizes))

        val = evaluate(g_model, Xva, yva)
        held = evaluate(g_model, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])  # logged only
        history.append(dict(strategy=strategy, seed=seed, round=r,
                            val_rmse=val["rmse"], heldout_rmse=held["rmse"]))
        if val["rmse"] < best_val:
            best_val, best_round = val["rmse"], r
            best_state = copy.deepcopy(g_model.state_dict())
        print(f"  [{strategy} s{seed}] round {r:2d}/{rounds} | val RMSE {val['rmse']:.2f} "
              f"| (FD004 RMSE {held['rmse']:.2f})")

    g_model.load_state_dict(best_state)
    per = [evaluate(g_model, data[c]["X_test"], data[c]["y_test"]) for c in CLIENTS]
    held = evaluate(g_model, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])

    # communication: each round every client downloads + uploads the full model (float32)
    comm_mb_per_round = 2 * len(CLIENTS) * n_params * 4 / 1e6
    result = dict(setup=f"{strategy}" + (f" (mu={mu})" if strategy == "fedprox" else ""),
                  seed=seed, best_round=best_round,
                  own_rmse=float(np.mean([p["rmse"] for p in per])),
                  own_score=float(np.mean([p["score"] for p in per])),
                  heldout_rmse=held["rmse"], heldout_score=held["score"],
                  comm_mb_total=comm_mb_per_round * rounds)
    return result, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="processed")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--lr_decay", type=float, default=0.95, help="lr multiplier per round")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="results/step3_federated.csv")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    data = {fd: load_factory(fd, args.proc_dir) for fd in CLIENTS + [HELD_OUT]}
    results, histories = [], []

    for seed in range(args.seeds):
        for strategy in ["fedavg", "fedprox"]:
            print(f"\n=== {strategy.upper()} | seed {seed} ===")
            res, hist = run_federated(data, strategy, args.rounds, args.local_epochs,
                                      args.lr, args.batch_size, args.mu, seed, args.lr_decay)
            results.append(res)
            histories.extend(hist)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    pd.DataFrame(histories).to_csv(args.out.replace(".csv", "_history.csv"), index=False)

    summary = df.groupby("setup")[["own_rmse", "own_score", "heldout_rmse",
                                   "heldout_score", "comm_mb_total"]].mean().round(2)
    print("\n================ SUMMARY (mean over seeds) ================")
    print("own_*     = average over the 3 training clients' test sets")
    print(f"heldout_* = unseen factory {HELD_OUT}\n")
    print(summary.to_string())
    print(f"\nSaved: {args.out} (+ _history.csv with per-round curves)")


if __name__ == "__main__":
    main()
