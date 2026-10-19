"""
Knowledge graph + grounded explanation utilities for FedKG-Maint (no torch needed).

Shared schema (ontology) agreed by all factories:
    Factory --EXPERIENCES--> FailureMode --AFFECTS--> Component --MONITORED_BY--> Sensor
    FailureMode --RECOMMENDS--> Action
    Sensor --WEAR_TREND_AT{direction, strength}--> Factory     (learned from the factory's OWN data)

Each factory builds its own local graph from its own processed training data.
Sensor/component meanings come from the public C-MAPSS documentation (Saxena et al., 2008).
The failure-mode -> component links and the maintenance ACTIONS are illustrative domain
knowledge written for this project (optional enrichment) -- review them before reporting.
"""
import re
import numpy as np
import networkx as nx

# code: (alias, description, component)
SENSORS = {
    "s2":  ("T24",     "LPC outlet temperature",         "Low-Pressure Compressor"),
    "s3":  ("T30",     "HPC outlet temperature",         "High-Pressure Compressor"),
    "s4":  ("T50",     "LPT outlet temperature",         "Low-Pressure Turbine"),
    "s7":  ("P30",     "HPC outlet pressure",            "High-Pressure Compressor"),
    "s8":  ("Nf",      "physical fan speed",             "Fan"),
    "s9":  ("Nc",      "physical core speed",            "Engine core"),
    "s11": ("Ps30",    "HPC outlet static pressure",     "High-Pressure Compressor"),
    "s12": ("phi",     "fuel-flow ratio at the HPC outlet", "High-Pressure Compressor"),
    "s13": ("NRf",     "corrected fan speed",            "Fan"),
    "s14": ("NRc",     "corrected core speed",           "Engine core"),
    "s15": ("BPR",     "bypass ratio",                   "Fan"),
    "s17": ("htBleed", "bleed enthalpy",                 "Bleed system"),
    "s20": ("W31",     "HPT coolant bleed",              "High-Pressure Turbine"),
    "s21": ("W32",     "LPT coolant bleed",              "Low-Pressure Turbine"),
}

FAILURE_MODES = {
    "HPC degradation": ["High-Pressure Compressor", "Engine core"],
    "Fan degradation": ["Fan"],
}
ACTIONS = {
    "HPC degradation": ["Schedule a borescope inspection of the high-pressure compressor",
                        "Plan HPC blade cleaning or replacement at the next shop visit"],
    "Fan degradation": ["Inspect the fan blades for erosion or damage",
                        "Check fan balance and bypass-duct condition"],
}
# From the C-MAPSS readme
FACTORY_FAILURES = {"FD001": ["HPC degradation"], "FD002": ["HPC degradation"],
                    "FD003": ["HPC degradation", "Fan degradation"],
                    "FD004": ["HPC degradation", "Fan degradation"]}
FACTORY_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

# component name -> regex that detects a mention in free text
COMPONENT_PATTERNS = {
    "High-Pressure Compressor": r"high[- ]pressure compressor|\bHPC\b",
    "Low-Pressure Compressor": r"low[- ]pressure compressor|\bLPC\b",
    "High-Pressure Turbine": r"high[- ]pressure turbine|\bHPT\b",
    "Low-Pressure Turbine": r"low[- ]pressure turbine|\bLPT\b",
    "Fan": r"\bfan\b",
    "Engine core": r"engine core",
    "Bleed system": r"bleed system",
}
FM_PATTERNS = {
    "HPC degradation": r"HPC degradation|high[- ]pressure compressor degradation",
    "Fan degradation": r"fan degradation",
}


# ------------------------------------------------------------------ KG construction
def build_local_kg(fd, d):
    """Build one factory's local KG from its own processed training data `d` (loaded npz dict)."""
    G = nx.MultiDiGraph()
    G.add_node(fd, type="Factory", conditions=FACTORY_CONDITIONS[fd])

    for fm in FACTORY_FAILURES[fd]:
        G.add_node(fm, type="FailureMode")
        G.add_edge(fd, fm, rel="EXPERIENCES")
        for comp in FAILURE_MODES[fm]:
            G.add_node(comp, type="Component")
            G.add_edge(fm, comp, rel="AFFECTS")
        for a in ACTIONS[fm]:
            G.add_node(a, type="Action")
            G.add_edge(fm, a, rel="RECOMMENDS")

    names = [str(n) for n in d["feature_names"]]
    last = d["X_train"][:, -1, :]            # most recent cycle of every training window
    y = d["y_train"]
    for s, (alias, desc, comp) in SENSORS.items():
        r = np.nan_to_num(np.corrcoef(last[:, names.index(s)], y)[0, 1])
        G.add_node(s, type="Sensor", alias=alias, desc=desc)
        G.add_node(comp, type="Component")
        G.add_edge(comp, s, rel="MONITORED_BY")
        # corr(sensor, RUL) > 0 -> sensor is high when healthy -> it DECREASES as wear grows
        G.add_edge(s, fd, rel="WEAR_TREND_AT",
                   direction="decreases" if r > 0 else "increases", strength=round(float(abs(r)), 2))
    return G


def kg_to_dict(G):
    return {"nodes": [{"id": n, **a} for n, a in G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **a} for u, v, a in G.edges(data=True)]}


# ------------------------------------------------------------------ retrieval
def sensor_modes(G, sensor):
    """Failure modes recorded at this factory that affect the component this sensor monitors."""
    comp = SENSORS[sensor][2]
    return [n for n, a in G.nodes(data=True)
            if a.get("type") == "FailureMode" and G.has_edge(n, comp)]


def wear_trend(G, fd, sensor):
    for _, v, a in G.out_edges(sensor, data=True):
        if v == fd and a.get("rel") == "WEAR_TREND_AT":
            return a["direction"], a["strength"]
    return None, 0.0


def pick_failure_mode(G, evidence):
    """Failure mode best supported by the (model-importance weighted) cited sensors."""
    score = {}
    for e in evidence:
        for fm in sensor_modes(G, e["sensor"]):
            score[fm] = score.get(fm, 0.0) + e["importance"]
    return max(score, key=score.get) if score else None


def risk_level(rul):
    return "critical" if rul < 30 else "warning" if rul < 60 else "normal"


def trend_word(slope):
    return "rising" if slope > 0.05 else "falling" if slope < -0.05 else "stable"


def retrieve_facts(G, fd, evidence):
    """Return the KG facts (as text lines) relevant to the cited sensors."""
    facts = []
    for e in evidence:
        s = e["sensor"]
        alias, desc, comp = SENSORS[s]
        direction, strength = wear_trend(G, fd, s)
        facts.append(f"Sensor {s} ({alias}, {desc}) monitors the {comp}.")
        facts.append(f"At {fd}, {s} tends to {direction.rstrip('s')} as wear progresses (|r|={strength}).")
    fm = pick_failure_mode(G, evidence)
    if fm:
        comps = ", ".join(FAILURE_MODES[fm])
        facts.append(f"{fd} has recorded failure mode '{fm}', which affects: {comps}.")
        for a in ACTIONS[fm]:
            facts.append(f"Recommended action for {fm}: {a}.")
    else:
        facts.append(f"No failure mode recorded at {fd} is linked to these sensors.")
    return facts, fm


# ------------------------------------------------------------------ explanations
def explain_no_kg(rul, evidence):
    codes = ", ".join(e["sensor"] for e in evidence)
    return (f"Predicted remaining useful life: about {rul:.0f} cycles ({risk_level(rul)}). "
            f"The prediction is most influenced by sensors {codes}. "
            f"Recommendation: schedule a maintenance inspection.")


def explain_kg(G, fd, rul, evidence):
    level = risk_level(rul)
    parts = [f"Predicted remaining useful life: about {rul:.0f} cycles ({level})."]
    ev_txt = []
    for e in evidence:
        s = e["sensor"]
        alias, desc, comp = SENSORS[s]
        direction, _ = wear_trend(G, fd, s)
        tw = trend_word(e["slope"])
        ok = (tw == "rising" and direction == "increases") or (tw == "falling" and direction == "decreases")
        cons = "consistent" if ok else "not consistent"
        ev_txt.append(f"{s} ({alias}, {desc}) is {e['z']:+.1f} standard deviations from the {fd} "
                      f"average and {tw}, {cons} with the wear trend seen at {fd}")
    parts.append("Key evidence: " + "; ".join(ev_txt) + ".")

    _, fm = retrieve_facts(G, fd, evidence)
    if fm:
        cited_comps = sorted({SENSORS[e["sensor"]][2] for e in evidence} & set(FAILURE_MODES[fm]))
        parts.append(f"Most likely failure mode: {fm}, affecting the {', '.join(cited_comps)}.")
        if level == "normal":
            parts.append(f"Action: continue routine monitoring and watch for {fm}.")
        elif level == "warning":
            parts.append(f"Action: {ACTIONS[fm][0]}.")
        else:
            parts.append(f"Action: {ACTIONS[fm][0]}; {ACTIONS[fm][1][0].lower() + ACTIONS[fm][1][1:]}.")
    else:
        parts.append(f"None of the failure modes recorded at {fd} is linked to these sensors; "
                     f"treat this as an unexplained anomaly. Action: schedule a diagnostic inspection.")
    return " ".join(parts)


# ------------------------------------------------------------------ LLM prompts
def prompt_no_kg(fd, rul, evidence):
    lines = "\n".join(f"- {e['sensor']}: {e['z']:+.1f} std devs from the factory average, "
                      f"{trend_word(e['slope'])}" for e in evidence)
    return (f"A predictive-maintenance model for turbofan engines at factory {fd} predicts a remaining "
            f"useful life of {rul:.0f} cycles ({risk_level(rul)}). The most influential sensors are:\n{lines}\n\n"
            "Write a short explanation and maintenance recommendation for a maintenance engineer "
            "(max 80 words).")


def prompt_kg(fd, rul, evidence, facts):
    lines = "\n".join(f"- {e['sensor']}: {e['z']:+.1f} std devs from the factory average, "
                      f"{trend_word(e['slope'])}" for e in evidence)
    f_txt = "\n".join(f"- {f}" for f in facts)
    return (f"A predictive-maintenance model for turbofan engines at factory {fd} predicts a remaining "
            f"useful life of {rul:.0f} cycles ({risk_level(rul)}). The most influential sensors are:\n{lines}\n\n"
            f"Knowledge-graph facts:\n{f_txt}\n\n"
            "Write a short explanation and maintenance recommendation for a maintenance engineer "
            "(max 80 words). Use ONLY the sensors, components, failure modes, numbers and actions that "
            "appear above. Do not mention anything that is not in the facts.")


def call_llm(provider, model, prompt):
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}])
        return r.content[0].text.strip()
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=300)
        return r.choices[0].message.content.strip()
    raise ValueError(f"unknown provider {provider}")


# ------------------------------------------------------------------ hallucination / grounding check
def find_mentions(text):
    """Entities mentioned in free text: sensors, components, failure modes."""
    found = set()
    for code, (alias, _, _) in SENSORS.items():
        if re.search(rf"\b{code}\b", text, re.I) or re.search(rf"\b{re.escape(alias)}\b", text):
            found.add(("sensor", code))
    for comp, pat in COMPONENT_PATTERNS.items():
        if re.search(pat, text, re.I):
            found.add(("component", comp))
    for fm, pat in FM_PATTERNS.items():
        if re.search(pat, text, re.I):
            found.add(("failure_mode", fm))
    return found


def check_explanation(text, pred_rul, evidence, fm):
    """Count mentions that are NOT supported by the case's evidence (cited sensors, their
    components, the selected failure mode) and numeric 'N cycles' claims that differ from the
    model's prediction."""
    ok_sensors = {e["sensor"] for e in evidence}
    ok_comps = {SENSORS[s][2] for s in ok_sensors}
    ok_fms = {fm} if fm else set()
    if fm:
        ok_comps |= set(FAILURE_MODES[fm])

    mentions = find_mentions(text)
    unsupported = []
    for kind, name in sorted(mentions):
        ok = (kind == "sensor" and name in ok_sensors) or \
             (kind == "component" and name in ok_comps) or \
             (kind == "failure_mode" and name in ok_fms)
        if not ok:
            unsupported.append(f"{kind}:{name}")
    n_mentions = len(mentions)

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:more\s+)?cycles?", text, re.I):
        n_mentions += 1
        if abs(float(m.group(1)) - pred_rul) > 1.5:
            unsupported.append(f"number:{m.group(1)} cycles")
    return {"n_mentions": n_mentions, "n_unsupported": len(unsupported), "unsupported": unsupported}
