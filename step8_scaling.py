"""
Step 8: Effect of the NUMBER of factories and of client HETEROGENEITY (C-MAPSS).

FedAvg is trained on every subset of the training clients {FD001, FD002, FD003}
(sizes k = 1, 2, 3) and always tested on the unseen factory FD004.

  - number of factories : compare k = 1, 2, 3
  - heterogeneity       : compare subsets of the same size but different composition,
                          e.g. {FD001, FD003} (both single operating condition: similar)
                          vs {FD001, FD002} (single vs six conditions: heterogeneous).
                          FD004 has six operating conditions and two fault modes, so
                          what matters is which of those a subset COVERS.
  - optional --central  : also trains a pooled (centralized) model on each subset, so
                          the FedAvg-minus-centralized gap shows the cost of federating
                          heterogeneous clients.

Uses the same FedAvg code as step3_federated.py. Fixed config for all subsets
(default 20 rounds x 1 local epoch) so every subset, including k=3, is comparable.

Usage:
    python step8_scaling.py                    # 1 seed
    python step8_scaling.py --central          # + centralized reference per subset
    python step8_scaling.py --rounds 15        # faster
"""
import argparse
import copy
import itertools
import os
import numpy as np
import pandas as pd
from common import (CLIENTS, HELD_OUT, DEVICE, LSTMRegressor, set_seed,
                    load_factory, evaluate, train_model)
from step3_federated import local_update, aggregate

# From the C-MAPSS readme
MULTI_CONDITION = {"FD001": False, "FD002": True, "FD003": False}   # FD004 has six conditions
FAN_FAULT = {"FD001": False, "FD002": False, "FD003": True}         # FD004 has HPC + Fan faults


def describe(subset):
    return dict(multi_condition_client="yes" if any(MULTI_CONDITION[c] for c in subset) else "no",
                fan_fault_covered="yes" if any(FAN_FAULT[c] for c in subset) else "no")


def fedavg_subset(data, clients, rounds, local_epochs, lr, lr_decay, batch_size, seed):
    set_seed(seed)
    g = LSTMRegressor(n_features=data[clients[0]]["X_train"].shape[-1]).to(DEVICE)
    n_params = sum(p.numel() for p in g.parameters())
    sizes = [len(data[c]["X_train"]) for c in clients]
    Xva = np.concatenate([data[c]["X_val"] for c in clients])
    yva = np.concatenate([data[c]["y_val"] for c in clients])

    best_val, best_state = float("inf"), None
    for r in range(1, rounds + 1):
        states = [local_update(g, data[c]["X_train"], data[c]["y_train"], local_epochs,
                               lr * (lr_decay ** (r - 1)), batch_size, 0.0) for c in clients]
        g.load_state_dict(aggregate(states, sizes))
        val = evaluate(g, Xva, yva)                       # selection on the subset's own validation data
        if val["rmse"] < best_val:
            best_val, best_state = val["rmse"], copy.deepcopy(g.state_dict())
    g.load_state_dict(best_state)

    own = float(np.mean([evaluate(g, data[c]["X_test"], data[c]["y_test"])["rmse"] for c in clients]))
    held = evaluate(g, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])
    comm = 2 * len(clients) * n_params * 4 * rounds / 1e6
    return own, held["rmse"], held["score"], comm


def central_subset(data, clients, epochs, seed):
    set_seed(seed)
    Xtr = np.concatenate([data[c]["X_train"] for c in clients]); ytr = np.concatenate([data[c]["y_train"] for c in clients])
    Xva = np.concatenate([data[c]["X_val"] for c in clients]); yva = np.concatenate([data[c]["y_val"] for c in clients])
    m = train_model(Xtr, ytr, Xva, yva, epochs, verbose=False)
    return evaluate(m, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])["rmse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="processed")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_decay", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--central", action="store_true", help="also train a pooled model per subset")
    ap.add_argument("--central_epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="results/step8_scaling.csv")
    a = ap.parse_args()

    print(f"Device: {DEVICE}")
    data = {fd: load_factory(fd, a.proc_dir) for fd in CLIENTS + [HELD_OUT]}
    subsets = [list(s) for k in (1, 2, 3) for s in itertools.combinations(CLIENTS, k)]

    rows = []
    for seed in range(a.seeds):
        for sub in subsets:
            name = "+".join(sub)
            print(f"\n[seed {seed}] FedAvg on {name} (k={len(sub)})")
            own, held, score, comm = fedavg_subset(data, sub, a.rounds, a.local_epochs, a.lr,
                                                   a.lr_decay, a.batch_size, seed)
            row = dict(subset=name, k=len(sub), seed=seed, **describe(sub), own_rmse=own,
                       heldout_rmse=held, heldout_score=score, comm_mb=comm)
            if a.central:
                row["central_heldout_rmse"] = central_subset(data, sub, a.central_epochs, seed)
                row["fed_minus_central"] = held - row["central_heldout_rmse"]
            rows.append(row)
            print(f"  FD004 RMSE {held:.2f}" + (f" | centralized {row['central_heldout_rmse']:.2f}" if a.central else ""))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    cols = ["own_rmse", "heldout_rmse", "heldout_score", "comm_mb"] + (["central_heldout_rmse", "fed_minus_central"] if a.central else [])
    by_subset = df.groupby(["k", "subset", "multi_condition_client", "fan_fault_covered"])[cols].mean().round(2)
    print("\n================ BY SUBSET (mean over seeds; test on unseen FD004) ================")
    print(by_subset.to_string())
    by_k = df.groupby("k")["heldout_rmse"].agg(["mean", "min", "max"]).round(2)
    print("\n================ BY NUMBER OF FACTORIES (FD004 RMSE across subsets) ================")
    print(by_k.to_string())
    print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
