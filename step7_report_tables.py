"""
Step 7: collect every results file into report-ready tables, significance tests and figures.

Reads from results/ (missing files are skipped):
  step2_baselines.csv, step3_federated.csv, step3_federated_history.csv,
  step4_privacy.csv, step4_privacy_clip3.csv, step5_explanations.csv,
  step6_azure.csv, noextra.csv (sensor-only Azure), no_days.csv, no_error.csv

Writes: results/report_tables.md  (+ PNG figures if matplotlib is installed)

Usage:
    pip install scipy matplotlib
    python step7_report_tables.py
"""
import os
import numpy as np
import pandas as pd

R = "results"


def path(name):
    return os.path.join(R, name)


def load(name):
    return pd.read_csv(path(name)) if os.path.exists(path(name)) else None


def fmt(x, nd=2):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{nd}f}"


def md_table(df, nd=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


def mean_std(df, group, cols, nd=2):
    """mean (± std when several seeds) per group."""
    out = []
    for g, sub in df.groupby(group, sort=False):
        row = {group: g}
        for c in cols:
            m, s = sub[c].mean(), sub[c].std()
            row[c] = f"{m:,.{nd}f} ± {s:,.{nd}f}" if len(sub) > 1 and not np.isnan(s) else fmt(m, nd)
        row["runs"] = str(len(sub))
        out.append(row)
    return pd.DataFrame(out)


def significance(df, metric, ref="fedavg"):
    """Paired t-test over seeds of every setup vs the reference setup."""
    try:
        from scipy import stats
    except ImportError:
        return "_(install scipy for significance tests)_"
    piv = df.pivot_table(index="seed", columns="setup", values=metric)
    if ref not in piv.columns or len(piv) < 3:
        return f"_(need >= 3 seeds; found {len(piv)} -- not enough for a significance test)_"
    rows = []
    for c in piv.columns:
        if c == ref:
            continue
        d = piv[ref] - piv[c]
        t, p = stats.ttest_rel(piv[ref], piv[c])
        rows.append({"comparison": f"{ref} vs {c}", "mean diff": d.mean(),
                     "p-value": p, "significant (p<0.05)": "yes" if p < 0.05 else "no"})
    return md_table(pd.DataFrame(rows), 4) + \
        f"\n\n_Paired t-test over {len(piv)} seeds; with so few seeds power is low, interpret cautiously._"


def section(title, body):
    return f"\n## {title}\n\n{body}\n"


def main():
    md = ["# FedKG-Maint: results summary\n"]

    # ---------------- C-MAPSS main ----------------
    s2, s3 = load("step2_baselines.csv"), load("step3_federated.csv")
    parts = []
    if s2 is not None:
        s2 = s2.assign(comm_mb=0.0); parts.append(s2)
    if s3 is not None:
        parts.append(s3.rename(columns={"comm_mb_total": "comm_mb"}))
    if parts:
        cm = pd.concat(parts, ignore_index=True)
        t = mean_std(cm, "setup", ["own_rmse", "heldout_rmse", "heldout_score", "comm_mb"])
        md.append(section("C-MAPSS: reference setups vs federated (held-out factory FD004)",
                          md_table(t) + "\n\nown_rmse = test sets of the training factories; "
                          "heldout_* = unseen factory FD004; score = NASA scoring function (lower is better)."))
        md.append(section("C-MAPSS: significance (FedAvg vs others, held-out RMSE)",
                          significance(cm, "heldout_rmse")))

    # ---------------- privacy ----------------
    for fname, label in [("step4_privacy.csv", "clip = 1.0"), ("step4_privacy_clip3.csv", "clip = 3.0")]:
        p = load(fname)
        if p is not None:
            p = p.assign(noise=p["sigma"] * p["clip"])
            t = p.groupby("sigma", as_index=False)[["noise", "epsilon", "own_rmse", "heldout_rmse"]].mean()
            t["epsilon"] = t["epsilon"].map(lambda e: "inf (no noise)" if np.isinf(e) else f"{e:,.1f}")
            md.append(section(f"Privacy vs performance (DP-FedAvg, {label}, delta=1e-5)",
                              md_table(t) + "\n\nnoise = sigma x clip is the effective noise scale; "
                              "with only 3 clients, meaningful epsilon (<10) requires noise that destroys accuracy."))

    # ---------------- Azure ----------------
    az = {"All features": load("step6_azure.csv"), "Sensors + age only": load("noextra.csv"),
          "No days_since": load("no_days.csv"), "No error counts": load("no_error.csv")}
    az = {k: v for k, v in az.items() if v is not None}
    for k, v in az.items():
        t = mean_std(v, "setup", ["own_prauc", "own_f1", "heldout_prauc", "heldout_f1", "comm_mb"], 3)
        md.append(section(f"Azure failure prediction -- {k} (held-out factory: model4)", md_table(t, 3)))
        sig = significance(v, "heldout_prauc")
        md.append(section(f"Azure significance -- {k} (held-out PR-AUC, FedAvg vs others)", sig))
    if len(az) > 1:
        comb = []
        for k, v in az.items():
            g = v.groupby("setup")["heldout_prauc"].mean().rename(k)
            comb.append(g)
        md.append(section("Azure feature ablation: held-out PR-AUC",
                          md_table(pd.concat(comb, axis=1).reset_index(), 3) +
                          "\n\nLarge drops show how much the maintenance/error features drive performance "
                          "(a property of the simulated dataset -- a limitation)."))

    # ---------------- KG grounding ----------------
    ex = load("step5_explanations.csv")
    if ex is not None:
        g = ex.groupby("mode").agg(explanations=("case", "count"), mentions=("n_mentions", "mean"),
                                   unsupported=("n_unsupported", "mean"),
                                   pct_with_hallucination=("n_unsupported", lambda s: 100 * (s > 0).mean()))
        g["grounded_%"] = 100 * (1 - ex.groupby("mode")["n_unsupported"].sum()
                                 / ex.groupby("mode")["n_mentions"].sum().clip(lower=1))
        md.append(section("Explanation grounding (held-out FD004 engines)", md_table(g.reset_index())))

    text = "\n".join(md)
    os.makedirs(R, exist_ok=True)
    with open(path("report_tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved: {path('report_tables.md')}")

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -> skipping figures; pip install matplotlib)")
        return

    h = load("step3_federated_history.csv")
    if h is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for strat, sub in h.groupby("strategy"):
            m = sub.groupby("round")[["val_rmse", "heldout_rmse"]].mean()
            ax[0].plot(m.index, m["val_rmse"], label=strat)
            ax[1].plot(m.index, m["heldout_rmse"], label=strat)
        ax[0].set_title("Validation RMSE (clients)"); ax[1].set_title("Held-out FD004 RMSE")
        for a in ax:
            a.set_xlabel("communication round"); a.set_ylabel("RMSE"); a.legend()
        fig.tight_layout(); fig.savefig(path("fig_fl_convergence.png"), dpi=150); plt.close(fig)

    p = load("step4_privacy_clip3.csv")
    if p is not None:
        m = p.groupby("sigma")["heldout_rmse"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(range(len(m)), m.values, marker="o")
        ax.set_xticks(range(len(m))); ax.set_xticklabels([str(s) for s in m.index])
        ax.set_yscale("log"); ax.set_xlabel("noise multiplier sigma"); ax.set_ylabel("held-out RMSE (log)")
        ax.set_title("Privacy vs accuracy (DP-FedAvg)")
        fig.tight_layout(); fig.savefig(path("fig_privacy_tradeoff.png"), dpi=150); plt.close(fig)

    if az:
        keys = list(az.keys())
        setups = sorted(set().union(*[set(v["setup"]) for v in az.values()]))
        fig, ax = plt.subplots(figsize=(9, 3.8))
        w = 0.8 / len(keys)
        for i, k in enumerate(keys):
            g = az[k].groupby("setup")["heldout_prauc"].mean()
            ax.bar(np.arange(len(setups)) + i * w, [g.get(s, np.nan) for s in setups], w, label=k)
        ax.set_xticks(np.arange(len(setups)) + w * (len(keys) - 1) / 2)
        ax.set_xticklabels(setups, rotation=30, ha="right")
        ax.set_ylabel("held-out PR-AUC"); ax.set_title("Azure: federated vs local, with/without extra features")
        ax.legend(); fig.tight_layout(); fig.savefig(path("fig_azure_ablation.png"), dpi=150); plt.close(fig)
    print("Figures saved to results/ (fig_*.png)")


if __name__ == "__main__":
    main()
