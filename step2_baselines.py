"""
Step 2: reference setups for FedKG-Maint.

  (a) Local-only : each factory (FD001/2/3) trains an LSTM on its OWN data only.
  (b) Centralized: one LSTM trained on the POOLED data of FD001+FD002+FD003
                   (upper-bound reference; breaks privacy).

Every model is evaluated on:
  - its own factory test set(s)          -> in-distribution performance
  - the held-out factory FD004 test set  -> cross-factory generalization

Usage:
    python step2_baselines.py                      # quick: 1 seed, 15 epochs
    python step2_baselines.py --seeds 3 --epochs 20
"""
import argparse
import os
import numpy as np
import pandas as pd
from common import (CLIENTS, HELD_OUT, DEVICE, set_seed, load_factory,
                    train_model, evaluate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="processed")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="results/step2_baselines.csv")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    data = {fd: load_factory(fd, args.proc_dir) for fd in CLIENTS + [HELD_OUT]}
    rows = []

    for seed in range(args.seeds):
        # ---------- (a) local-only ----------
        for fd in CLIENTS:
            print(f"\n[seed {seed}] LOCAL-ONLY  {fd}")
            set_seed(seed)
            d = data[fd]
            model = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], args.epochs)
            own = evaluate(model, d["X_test"], d["y_test"])
            held = evaluate(model, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])
            print(f"  own test RMSE {own['rmse']:.2f} | held-out {HELD_OUT} RMSE {held['rmse']:.2f}")
            rows.append(dict(setup=f"local-only ({fd})", seed=seed,
                             own_rmse=own["rmse"], own_score=own["score"],
                             heldout_rmse=held["rmse"], heldout_score=held["score"]))

        # ---------- (b) centralized (pooled clients) ----------
        print(f"\n[seed {seed}] CENTRALIZED (pooled {', '.join(CLIENTS)})")
        set_seed(seed)
        Xtr = np.concatenate([data[f]["X_train"] for f in CLIENTS])
        ytr = np.concatenate([data[f]["y_train"] for f in CLIENTS])
        Xva = np.concatenate([data[f]["X_val"] for f in CLIENTS])
        yva = np.concatenate([data[f]["y_val"] for f in CLIENTS])
        model = train_model(Xtr, ytr, Xva, yva, args.epochs)
        per = [evaluate(model, data[f]["X_test"], data[f]["y_test"]) for f in CLIENTS]
        held = evaluate(model, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])
        own_rmse = float(np.mean([p["rmse"] for p in per]))
        own_score = float(np.mean([p["score"] for p in per]))
        print(f"  avg own-test RMSE {own_rmse:.2f} | held-out {HELD_OUT} RMSE {held['rmse']:.2f}")
        rows.append(dict(setup="centralized (pooled)", seed=seed,
                         own_rmse=own_rmse, own_score=own_score,
                         heldout_rmse=held["rmse"], heldout_score=held["score"]))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = df.groupby("setup")[["own_rmse", "own_score", "heldout_rmse", "heldout_score"]].mean().round(2)
    print("\n================ SUMMARY (mean over seeds) ================")
    print(f"own_*     = test set of the factory(ies) it trained on")
    print(f"heldout_* = test set of unseen factory {HELD_OUT}\n")
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
