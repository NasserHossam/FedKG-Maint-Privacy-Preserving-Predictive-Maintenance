"""
Azure Predictive Maintenance: feature engineering + labels (no torch needed).

One row per machine-hour. Target: does ANY component fail within the next 24 hours?
Features (30): 4 raw sensors, 3h/24h rolling mean+std of each sensor (16),
               24h rolling error counts error1..5 (5), days since last maintenance of
               comp1..4 (4), machine age (1).

Factories = machine models (model1..model4), as suggested in the project description.
"""
import os
import numpy as np
import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
MAX_DAYS = 365.0


def build_features(data_dir):
    p = lambda f: os.path.join(data_dir, f)
    tel = pd.read_csv(p("PdM_telemetry.csv"), parse_dates=["datetime"])
    err = pd.read_csv(p("PdM_errors.csv"), parse_dates=["datetime"])
    maint = pd.read_csv(p("PdM_maint.csv"), parse_dates=["datetime"])
    fail = pd.read_csv(p("PdM_failures.csv"), parse_dates=["datetime"])
    mach = pd.read_csv(p("PdM_machines.csv"))

    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    feats = {s: tel[s].astype(np.float32) for s in SENSORS}

    # --- rolling sensor statistics (per machine) ---
    g = tel.groupby("machineID")[SENSORS]
    for w in (3, 24):
        mean = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
        std = g.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).sort_index().fillna(0)
        for s in SENSORS:
            feats[f"{s}_mean{w}h"] = mean[s].astype(np.float32)
            feats[f"{s}_std{w}h"] = std[s].astype(np.float32)

    # --- error counts, rolling 24h ---
    err["datetime"] = err["datetime"].dt.floor("h")
    cnt = err.groupby(["machineID", "datetime", "errorID"]).size().unstack("errorID", fill_value=0)
    for e in [f"error{i}" for i in range(1, 6)]:
        if e not in cnt.columns:
            cnt[e] = 0
    cnt = cnt[[f"error{i}" for i in range(1, 6)]].reset_index()
    m = tel[["machineID", "datetime"]].merge(cnt, on=["machineID", "datetime"], how="left").fillna(0)
    m = m.set_index(tel.index)
    roll = m.groupby(tel["machineID"])[[f"error{i}" for i in range(1, 6)]] \
            .rolling(24, min_periods=1).sum().reset_index(level=0, drop=True).sort_index()
    for e in roll.columns:
        feats[f"{e}_24h"] = roll[e].astype(np.float32)

    # --- days since last maintenance of each component ---
    base = tel[["machineID", "datetime"]].copy()
    base["_row"] = np.arange(len(base))
    base = base.sort_values("datetime")
    for comp in [f"comp{i}" for i in range(1, 5)]:
        mm = maint.loc[maint["comp"] == comp, ["machineID", "datetime"]].copy()
        mm["last_maint"] = mm["datetime"]
        mm = mm.sort_values("datetime")
        merged = pd.merge_asof(base, mm, on="datetime", by="machineID", direction="backward",
                                 allow_exact_matches=False)   # strictly BEFORE t (no leakage)
        days = (merged["datetime"] - merged["last_maint"]).dt.total_seconds() / 86400.0
        days = days.fillna(MAX_DAYS).clip(0, MAX_DAYS).astype(np.float32).values
        out = np.empty(len(tel), dtype=np.float32)
        out[merged["_row"].values] = days
        feats[f"days_since_{comp}"] = pd.Series(out, index=tel.index)

    # --- machine age ---
    age = tel["machineID"].map(mach.set_index("machineID")["age"]).astype(np.float32)
    feats["age"] = age

    names = list(feats.keys())
    X = np.stack([feats[n].values for n in names], axis=1).astype(np.float32)

    # --- label: failure within the next 24 hours (strictly after the current hour) ---
    y = np.zeros(len(tel), dtype=np.int8)
    tdt = tel["datetime"].values.astype("datetime64[s]")
    for mid, idx in tel.groupby("machineID").indices.items():
        F = np.sort(fail.loc[fail["machineID"] == mid, "datetime"].values.astype("datetime64[s]"))
        if len(F) == 0:
            continue
        t = tdt[idx]
        pos = np.searchsorted(F, t, side="right")   # first failure strictly AFTER t
        valid = pos < len(F)
        delta_h = np.full(len(t), np.inf)
        delta_h[valid] = (F[pos[valid]] - t[valid]) / np.timedelta64(1, "h")
        y[idx] = (delta_h <= 24).astype(np.int8)    # failure within (t, t+24h]

    model = tel["machineID"].map(mach.set_index("machineID")["model"]).astype(str).values
    return dict(X=X, y=y, machine=tel["machineID"].values, hour=tdt.astype("int64"),
                model=model, feature_names=np.array(names))


def load_or_build(data_dir, cache="processed/azure_features.npz"):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print("Building Azure features (one-off, ~1 minute)...")
    d = build_features(data_dir)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **d)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    args = ap.parse_args()
    d = load_or_build(args.data_dir)
    print("rows:", d["X"].shape, "| features:", len(d["feature_names"]))
    print("positive rate: %.2f%%" % (100 * d["y"].mean()))
    for mdl in sorted(set(d["model"])):
        msk = d["model"] == mdl
        print(f"  {mdl}: {len(set(d['machine'][msk]))} machines, rows {msk.sum()}, "
              f"positives {int(d['y'][msk].sum())} ({100 * d['y'][msk].mean():.2f}%)")
