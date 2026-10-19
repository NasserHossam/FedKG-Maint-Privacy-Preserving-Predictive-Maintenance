"""
Step 5: Maintenance knowledge graph + grounded explanations (on the held-out factory FD004).

Pipeline per test engine:
  1. Model predicts RUL.
  2. Occlusion: zero out each sensor over the window; the sensors that change the prediction
     most are the model-faithful evidence.
  3. Explanation WITHOUT KG  : prediction + sensor codes only.
     Explanation WITH KG     : + retrieved KG facts (component, wear trend, failure mode, action).
  4. Checks: unsupported mentions / hallucinations, and a deletion-test of faithfulness.

By default explanations are template-based (free, deterministic). To also test a real LLM:
    set ANTHROPIC_API_KEY (or OPENAI_API_KEY), then
    python step5_explain.py --llm anthropic --model claude-sonnet-5-5
    python step5_explain.py --llm openai    --model gpt-4o-mini

Usage:
    pip install networkx
    python step5_explain.py                 # 30 engines, template explainers
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from common import (CLIENTS, HELD_OUT, DEVICE, LSTMRegressor, set_seed,
                    load_factory, predict, train_model)
import kg_explain as K

SENSOR_IDX = None  # filled in main: indices of sensor features (not op settings)


def get_model(data, epochs, path):
    if os.path.exists(path):
        m = LSTMRegressor(n_features=data[CLIENTS[0]]["X_train"].shape[-1]).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded model from {path}")
        return m
    print(f"No saved model: training a pooled LSTM for {epochs} epochs (explanations are model-agnostic)")
    Xtr = np.concatenate([data[c]["X_train"] for c in CLIENTS])
    ytr = np.concatenate([data[c]["y_train"] for c in CLIENTS])
    Xva = np.concatenate([data[c]["X_val"] for c in CLIENTS])
    yva = np.concatenate([data[c]["y_val"] for c in CLIENTS])
    m = train_model(Xtr, ytr, Xva, yva, epochs, verbose=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(m.state_dict(), path)
    return m


def occlusion_evidence(model, x, names, top_k=3):
    """x: one window (T, F). Returns (pred, evidence list, per-feature |delta| dict)."""
    base = predict(model, x[None])[0]
    batch = np.repeat(x[None], len(SENSOR_IDX), axis=0)
    for row, f in enumerate(SENSOR_IDX):
        batch[row, :, f] = 0.0                       # 0 = factory average (z-scored)
    deltas = np.abs(predict(model, batch) - base)
    order = np.argsort(-deltas)[:top_k]
    evidence = []
    for o in order:
        f = SENSOR_IDX[o]
        series = x[:, f]
        evidence.append(dict(sensor=names[f], importance=float(deltas[o]),
                             z=float(series[-5:].mean()),
                             slope=float(series[-5:].mean() - series[:5].mean())))
    return float(base), evidence, deltas


def deletion_test(model, x, evidence, names, rng, n_random=20):
    """Prediction change when the cited sensors are removed vs the same number of random sensors."""
    base = predict(model, x[None])[0]
    cited = [names.index(e["sensor"]) for e in evidence]
    xc = x.copy(); xc[:, cited] = 0.0
    d_cited = abs(predict(model, xc[None])[0] - base)
    d_rand = []
    for _ in range(n_random):
        idx = rng.choice(SENSOR_IDX, size=len(cited), replace=False)
        xr = x.copy(); xr[:, idx] = 0.0
        d_rand.append(abs(predict(model, xr[None])[0] - base))
    return float(d_cited), float(np.mean(d_rand))


def main():
    global SENSOR_IDX
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="processed")
    ap.add_argument("--n", type=int, default=30, help="number of held-out engines to explain")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--model_path", default="models/explain_model.pt")
    ap.add_argument("--llm", choices=["none", "anthropic", "openai"], default="none")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--kg_dir", default="kg")
    args = ap.parse_args()

    set_seed(0)
    rng = np.random.default_rng(0)
    data = {fd: load_factory(fd, args.proc_dir) for fd in CLIENTS + [HELD_OUT]}
    names = [str(n) for n in data[HELD_OUT]["feature_names"]]
    SENSOR_IDX = [i for i, n in enumerate(names) if n.startswith("s")]
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.kg_dir, exist_ok=True)

    # ---- each factory builds its OWN local KG from its OWN data (shared schema) ----
    kgs = {}
    print("\nLocal knowledge graphs:")
    for fd, d in data.items():
        kgs[fd] = K.build_local_kg(fd, d)
        with open(os.path.join(args.kg_dir, f"{fd}.json"), "w") as f:
            json.dump(K.kg_to_dict(kgs[fd]), f, indent=1)
        print(f"  {fd}: {kgs[fd].number_of_nodes()} nodes, {kgs[fd].number_of_edges()} edges")
    G = kgs[HELD_OUT]

    # ---- model + held-out cases ----
    model = get_model(data, args.epochs, args.model_path)
    X, y_true = data[HELD_OUT]["X_test"], data[HELD_OUT]["y_test"]
    n = min(args.n, len(X))
    modes = ["template_noKG", "template_KG"] + (["llm_noKG", "llm_KG"] if args.llm != "none" else [])

    rows, faith = [], []
    for i in range(n):
        pred, ev, _ = occlusion_evidence(model, X[i], names)
        facts, fm = K.retrieve_facts(G, HELD_OUT, ev)
        texts = {"template_noKG": K.explain_no_kg(pred, ev),
                 "template_KG": K.explain_kg(G, HELD_OUT, pred, ev)}
        if args.llm != "none":
            texts["llm_noKG"] = K.call_llm(args.llm, args.model, K.prompt_no_kg(HELD_OUT, pred, ev))
            texts["llm_KG"] = K.call_llm(args.llm, args.model, K.prompt_kg(HELD_OUT, pred, ev, facts))
        for mode in modes:
            chk = K.check_explanation(texts[mode], pred, ev, fm)
            rows.append(dict(case=i, mode=mode, true_rul=float(y_true[i]), pred_rul=pred,
                             failure_mode=fm, evidence=",".join(e["sensor"] for e in ev),
                             n_mentions=chk["n_mentions"], n_unsupported=chk["n_unsupported"],
                             unsupported="; ".join(chk["unsupported"]), text=texts[mode]))
        faith.append(deletion_test(model, X[i], ev, names, rng))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "step5_explanations.csv"), index=False)

    # ---- summary ----
    print(f"\n================ GROUNDING CHECK on {n} held-out {HELD_OUT} engines ================")
    summ = df.groupby("mode").agg(mentions=("n_mentions", "mean"),
                                  unsupported=("n_unsupported", "mean"),
                                  pct_with_hallucination=("n_unsupported", lambda s: 100 * (s > 0).mean()))
    summ["grounded_%"] = (100 * (1 - df.groupby("mode")["n_unsupported"].sum()
                                 / df.groupby("mode")["n_mentions"].sum().clip(lower=1)))
    print(summ.loc[modes].round(2).to_string())

    fc, fr = np.mean([f[0] for f in faith]), np.mean([f[1] for f in faith])
    print(f"\nFaithfulness (deletion test): removing the CITED sensors changes the RUL prediction by "
          f"{fc:.2f} cycles on average vs {fr:.2f} for random sensors ({fc / max(fr, 1e-9):.1f}x).")
    print("Note: with templates the 'with KG' text is built only from retrieved facts, so ~100% grounding "
          "is expected; the informative comparison is a real LLM (--llm) with vs without KG.")

    print("\n--- Example (case 0) ---")
    for mode in modes:
        print(f"[{mode}] {df[(df.case == 0) & (df['mode'] == mode)].iloc[0]['text']}\n")
    print(f"Saved: {args.out_dir}/step5_explanations.csv and {args.kg_dir}/FD00X.json")


if __name__ == "__main__":
    main()
