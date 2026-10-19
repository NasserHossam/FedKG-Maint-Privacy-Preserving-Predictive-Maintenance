"""
Step 4: Privacy preservation via client-level differential privacy (DP-FedAvg).

Each round, every client's update (delta = local_weights - global_weights) is
  1. clipped to L2 norm <= C        (bounds one factory's influence)
  2. averaged across clients
  3. perturbed with Gaussian noise  N(0, (sigma * C / n_clients)^2) per weight
so the server (and anyone seeing the global model) learns little about any single factory.

sigma = 0 is the "no noise" control (still clipped, uniform averaging) so the
comparison isolates the effect of the noise.

Privacy accounting: Gaussian mechanism, full participation, Renyi-DP composition over
all rounds, converted to (epsilon, delta)-DP (add/remove-one-client neighbouring relation).

Usage:
    python step4_privacy.py                               # sigmas 0 0.05 0.5 5
    python step4_privacy.py --sigmas 0 0.01 0.05 --rounds 30
    python step4_privacy.py --clip 2.0                    # if too many updates are clipped
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import copy
from common import (CLIENTS, HELD_OUT, DEVICE, LSTMRegressor, set_seed,
                    load_factory, evaluate)
from step3_federated import local_update


def epsilon_gaussian(sigma, rounds, delta=1e-5):
    """(eps, delta)-DP for `rounds` compositions of the Gaussian mechanism (RDP accountant)."""
    if sigma <= 0:
        return float("inf")
    alphas = np.concatenate([np.linspace(1.01, 10, 500), np.linspace(10, 5000, 2000)])
    eps = rounds * alphas / (2 * sigma ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


def dp_aggregate(global_state, client_states, clip, sigma, gen):
    """Clip each client delta, average, add Gaussian noise. Returns (new_state, mean_norm, clip_frac)."""
    n = len(client_states)
    keys = list(global_state.keys())
    deltas, norms = [], []
    for s in client_states:
        d = {k: (s[k] - global_state[k]).float() for k in keys}
        norm = torch.sqrt(sum((v ** 2).sum() for v in d.values())).item()
        scale = min(1.0, clip / (norm + 1e-12))
        deltas.append({k: v * scale for k, v in d.items()})
        norms.append(norm)

    new_state = {}
    for k in keys:
        avg = sum(d[k] for d in deltas) / n
        if sigma > 0:
            noise = torch.randn(avg.shape, generator=gen) * (sigma * clip / n)
            avg = avg + noise
        new_state[k] = global_state[k] + avg
    clip_frac = float(np.mean([nm > clip for nm in norms]))
    return new_state, float(np.mean(norms)), clip_frac


def run_dp(data, sigma, clip, rounds, local_epochs, lr, lr_decay, batch_size, seed):
    set_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    g_model = LSTMRegressor(n_features=data[CLIENTS[0]]["X_train"].shape[-1]).to(DEVICE)
    Xva = np.concatenate([data[c]["X_val"] for c in CLIENTS])
    yva = np.concatenate([data[c]["y_val"] for c in CLIENTS])

    best_val, best_state, best_round = float("inf"), None, 0
    for r in range(1, rounds + 1):
        global_state = {k: v.detach().cpu().clone() for k, v in g_model.state_dict().items()}
        states = [local_update(g_model, data[c]["X_train"], data[c]["y_train"], local_epochs,
                               lr * (lr_decay ** (r - 1)), batch_size, 0.0) for c in CLIENTS]
        new_state, mean_norm, clip_frac = dp_aggregate(global_state, states, clip, sigma, gen)
        g_model.load_state_dict(new_state)

        val = evaluate(g_model, Xva, yva)
        if val["rmse"] < best_val:
            best_val, best_round = val["rmse"], r
            best_state = copy.deepcopy(g_model.state_dict())
        print(f"  [sigma={sigma} s{seed}] round {r:2d}/{rounds} | val RMSE {val['rmse']:.2f} "
              f"| mean update norm {mean_norm:.2f} | clipped {clip_frac:.0%}")

    g_model.load_state_dict(best_state)
    per = [evaluate(g_model, data[c]["X_test"], data[c]["y_test"]) for c in CLIENTS]
    held = evaluate(g_model, data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"])
    return dict(sigma=sigma, clip=clip, seed=seed, best_round=best_round,
                epsilon=epsilon_gaussian(sigma, rounds),
                own_rmse=float(np.mean([p["rmse"] for p in per])),
                heldout_rmse=held["rmse"], heldout_score=held["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="processed")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.5, 5.0])
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_decay", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="results/step4_privacy.csv")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    data = {fd: load_factory(fd, args.proc_dir) for fd in CLIENTS + [HELD_OUT]}
    rows = []
    for seed in range(args.seeds):
        for sigma in args.sigmas:
            print(f"\n=== DP-FedAvg | sigma={sigma} | clip={args.clip} | seed {seed} ===")
            rows.append(run_dp(data, sigma, args.clip, args.rounds, args.local_epochs,
                               args.lr, args.lr_decay, args.batch_size, seed))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby("sigma")[["epsilon", "own_rmse", "heldout_rmse", "heldout_score"]].mean().round(2)
    print("\n================ PRIVACY vs PERFORMANCE (mean over seeds) ================")
    print(f"epsilon: (eps, delta=1e-5)-DP over {args.rounds} rounds; inf = no noise\n")
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
