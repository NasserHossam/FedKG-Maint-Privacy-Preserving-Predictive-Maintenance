"""
Step 1: C-MAPSS data preparation for FedKG-Maint.

Each sub-dataset (FD001..FD004) is treated as one "factory".
- FD001, FD002, FD003 -> federated training clients
- FD004               -> held-out factory (never used in training)

Normalization statistics are computed LOCALLY inside each factory,
from that factory's own training engines only.

Usage:
    python step1_cmapss_prep.py --data_dir data/CMAPSS --out_dir processed
Output (one file per factory): processed/FD00X.npz with
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, mean, std
"""
import argparse
import os
import numpy as np
import pandas as pd

# ---------------- configuration ----------------
FACTORIES = ["FD001", "FD002", "FD003", "FD004"]
HELD_OUT = "FD004"
WINDOW = 30          # time-window length (cycles)
RUL_CAP = 125        # piece-wise linear RUL cap (standard in the literature)
VAL_FRAC = 0.2       # fraction of training ENGINES used for validation
SEED = 42

COLS = (["engine", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)])

# Sensors that are constant / uninformative in C-MAPSS (standard drop list).
# All factories must share the SAME feature schema so one federated model fits all.
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
FEATURES = [c for c in COLS[2:] if c not in DROP_SENSORS]   # 3 op settings + 14 sensors = 17


# ---------------- loading ----------------
def load_factory(data_dir, fd):
    """Return train df, test df, true RUL vector for the test engines."""
    def read(name):
        path = os.path.join(data_dir, name)
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df.iloc[:, :26]                      # drop any trailing empty columns
        df.columns = COLS
        return df

    train = read(f"train_{fd}.txt")
    test = read(f"test_{fd}.txt")
    rul = np.loadtxt(os.path.join(data_dir, f"RUL_{fd}.txt")).astype(float).ravel()
    return train, test, rul


# ---------------- targets ----------------
def add_train_rul(train):
    max_cycle = train.groupby("engine")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)
    return train


# ---------------- local normalization ----------------
def fit_local_scaler(train_df):
    """Mean/std computed ONLY from this factory's own training rows."""
    mean = train_df[FEATURES].mean()
    std = train_df[FEATURES].std().replace(0, 1.0)   # guard against zero variance
    return mean, std


def apply_scaler(df, mean, std):
    df = df.copy()
    df[FEATURES] = (df[FEATURES] - mean) / std
    return df


# ---------------- windowing ----------------
def make_train_windows(df):
    """Sliding windows (stride 1); label = RUL at the last cycle of the window."""
    X, y = [], []
    for _, g in df.groupby("engine"):
        feats = g[FEATURES].values.astype(np.float32)
        ruls = g["RUL"].values.astype(np.float32)
        if len(g) < WINDOW:                         # pad short engines by repeating first row
            pad = np.repeat(feats[:1], WINDOW - len(g), axis=0)
            feats = np.vstack([pad, feats])
            ruls = np.concatenate([np.repeat(ruls[:1], WINDOW - len(g)), ruls])
        for end in range(WINDOW, len(feats) + 1):
            X.append(feats[end - WINDOW:end])
            y.append(ruls[end - 1])
    return np.stack(X), np.array(y, dtype=np.float32)


def make_test_windows(df, true_rul):
    """One window per test engine (its last WINDOW cycles); label = official RUL (capped)."""
    X = []
    for _, g in df.groupby("engine"):
        feats = g[FEATURES].values.astype(np.float32)
        if len(feats) < WINDOW:
            pad = np.repeat(feats[:1], WINDOW - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-WINDOW:])
    y = np.minimum(true_rul, RUL_CAP).astype(np.float32)
    return np.stack(X), y


# ---------------- per-factory pipeline ----------------
def process_factory(data_dir, fd, rng):
    train, test, rul = load_factory(data_dir, fd)
    train = add_train_rul(train)

    # split TRAIN ENGINES into train / val (no window leakage across engines)
    engines = train["engine"].unique()
    rng.shuffle(engines)
    n_val = max(1, int(len(engines) * VAL_FRAC))
    val_ids = set(engines[:n_val])
    tr_df = train[~train["engine"].isin(val_ids)]
    va_df = train[train["engine"].isin(val_ids)]

    # local scaler fitted on this factory's training engines only
    mean, std = fit_local_scaler(tr_df)
    tr_df, va_df, te_df = (apply_scaler(d, mean, std) for d in (tr_df, va_df, test))

    X_tr, y_tr = make_train_windows(tr_df)
    X_va, y_va = make_train_windows(va_df)
    X_te, y_te = make_test_windows(te_df, rul)

    return dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
                X_test=X_te, y_test=y_te,
                feature_names=np.array(FEATURES),
                mean=mean.values.astype(np.float32),
                std=std.values.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/CMAPSS")
    ap.add_argument("--out_dir", default="processed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"{len(FEATURES)} features: {FEATURES}\n")
    for fd in FACTORIES:
        out = process_factory(args.data_dir, fd, rng)
        np.savez_compressed(os.path.join(args.out_dir, f"{fd}.npz"), **out)
        role = "HELD-OUT" if fd == HELD_OUT else "client"
        print(f"{fd} ({role:8s}) | train {out['X_train'].shape} | val {out['X_val'].shape} "
              f"| test {out['X_test'].shape} | RUL train mean {out['y_train'].mean():.1f}")
    print(f"\nSaved to {args.out_dir}/. Training clients: "
          f"{[f for f in FACTORIES if f != HELD_OUT]}, held-out: {HELD_OUT}")


if __name__ == "__main__":
    main()
