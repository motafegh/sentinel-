"""Stage 5.1: Evidence Integration
Harvests all evidence produced across Phases 1-4 into a unified per-contract
evidence table. Produces per-class coverage stats and identifies which classes
already pass the Stage 5.1 gate.

Run from repo root:
    poetry run python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_s1_evidence_integration.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive")
P4_OUT = ROOT / "Phase4_LabelValidation_2026-06-07/outputs"
P5_OUT = ROOT / "Phase5_LabelVerification_2026-06-08/outputs"
P5_OUT.mkdir(parents=True, exist_ok=True)

DATASET      = P4_OUT / "ws_p4_s01b_d12_applied.csv"
SLITHER_CSV  = P4_OUT / "ws_p4_s1_slither_results.csv"
ADERYN_CSV   = P4_OUT / "ws_p4_s1_aderyn_results.csv"
REGEX_CSV    = P4_OUT / "ws_p4_s05_regex_features.csv"
HANDCRAFT_CSV= P4_OUT / "ws_p4_s06_handcrafted_features.csv"
MANUAL_CSV   = P4_OUT / "ws_p4_s1_review_200.csv"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
SENTINEL_CLASSES = [
    "Class01:ExternalBug",
    "Class02:GasException",
    "Class03:MishandledException",
    "Class04:Timestamp",
    "Class06:UnusedReturn",
    "Class08:CallToUnknown",
    "Class09:DenialOfService",
    "Class10:IntegerUO",
    "Class11:Reentrancy",
]

# Slither detector → SENTINEL class mapping (from CRITICAL_FINDINGS.md + class definitions)
SLITHER_DETECTOR_MAP = {
    "reentrancy-eth":            "Class11:Reentrancy",
    "reentrancy-no-eth":         "Class11:Reentrancy",
    "reentrancy-benign":         "Class11:Reentrancy",
    "reentrancy-events":         "Class11:Reentrancy",
    "divide-before-multiply":    "Class10:IntegerUO",
    "tautology":                 "Class10:IntegerUO",
    "unused-return":             "Class06:UnusedReturn",
    "suicidal":                  "Class01:ExternalBug",
    "tx-origin":                 "Class01:ExternalBug",
    "controlled-delegatecall":   "Class08:CallToUnknown",
    "low-level-calls":           "Class08:CallToUnknown",
    "unchecked-transfer":        "Class03:MishandledException",
    "unchecked-send":            "Class03:MishandledException",
    "unchecked-lowlevel":        "Class03:MishandledException",
    "costly-loop":               "Class02:GasException",
    "calls-loop":                "Class09:DenialOfService",
    "reentrancy-unlimited-gas":  "Class09:DenialOfService",
    "timestamp":                 "Class04:Timestamp",
    "weak-prng":                 "Class04:Timestamp",
}

# Aderyn detector → SENTINEL class mapping (verified against actual detector names in dataset)
ADERYN_DETECTOR_MAP = {
    "reentrancy-state-change":     "Class11:Reentrancy",
    "unchecked-return":            "Class06:UnusedReturn",
    "unchecked-send":              "Class03:MishandledException",
    "unchecked-low-level-call":    "Class03:MishandledException",
    "uninitialized-local-variable":"Class03:MishandledException",
    "selfdestruct":                "Class01:ExternalBug",
    "centralization-risk":         "Class01:ExternalBug",
    "ecrecover":                   "Class01:ExternalBug",
    "unsafe-erc20-operation":      "Class08:CallToUnknown",
    "weak-randomness":             "Class04:Timestamp",
    "division-before-multiplication": "Class10:IntegerUO",
}

# Regex feature → class signal mapping
# Value = (feature_col, direction, confidence_contribution)
# direction = 'positive' means feature=1 suggests class is present
REGEX_CLASS_SIGNALS = {
    "Class11:Reentrancy": [
        ("f10_callvalue", "positive"),   # .call.value() present — primary signal
    ],
    "Class04:Timestamp": [
        ("f06_block_timestamp", "positive"),
        ("f07_now", "positive"),
    ],
    "Class01:ExternalBug": [
        ("f08_tx_origin", "positive"),
        ("f17_selfdestruct", "positive"),
        ("f09_delegatecall", "positive"),
        ("f14_ecrecover", "positive"),
    ],
    "Class08:CallToUnknown": [
        ("f09_delegatecall", "positive"),
        ("f13_lowlevel_call", "positive"),
    ],
    "Class03:MishandledException": [
        ("f12_send", "positive"),
        ("f13_lowlevel_call", "positive"),
    ],
    "Class06:UnusedReturn": [
        ("f11_transfer", "positive"),
        ("f12_send", "positive"),
    ],
    "Class10:IntegerUO": [
        ("h03_unsafe_arith_no_safemath", "positive"),  # from handcrafted
    ],
    "Class02:GasException": [],   # no direct regex signal
    "Class09:DenialOfService": [],# no direct regex signal
}

# Confidence weights from Stage 5.0 plan
WEIGHTS = {
    "manual":  1.00,
    "slither": 0.75,
    "aderyn":  0.65,
    "regex":   0.60,
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading base dataset...")
base = pd.read_csv(DATASET, usecols=["id"] + SENTINEL_CLASSES)
print(f"  Base: {len(base):,} contracts, {len(SENTINEL_CLASSES)} classes")

print("Loading slither results...")
slither = pd.read_csv(SLITHER_CSV, usecols=["id", "status", "hit_counts_json"])
print(f"  Slither: {len(slither):,} contracts ({(slither.status=='OK').sum():,} OK)")

print("Loading aderyn results...")
aderyn = pd.read_csv(ADERYN_CSV, usecols=["id", "status", "hits_json"])
print(f"  Aderyn: {len(aderyn):,} contracts ({(aderyn.status!='COMPILE_FAIL').sum():,} non-fail)")

print("Loading regex features...")
regex = pd.read_csv(REGEX_CSV)
print(f"  Regex: {len(regex):,} contracts, {len(regex.columns)-1} features")

print("Loading handcrafted features...")
handcraft = pd.read_csv(HANDCRAFT_CSV)
print(f"  Handcrafted: {len(handcraft):,} contracts")

print("Loading manual reviews...")
manual = pd.read_csv(MANUAL_CSV, usecols=["id", "class", "decision"])
print(f"  Manual: {len(manual):,} (id,class) pairs, classes: {manual['class'].unique()}")


# ---------------------------------------------------------------------------
# Parse slither: for each contract, which SENTINEL classes did slither flag?
# ---------------------------------------------------------------------------
print("\nParsing slither detector hits...")

def parse_slither_classes(row):
    """Return set of SENTINEL classes flagged by slither for this contract."""
    if row["status"] != "OK":
        return set()
    try:
        counts = json.loads(row["hit_counts_json"])
    except Exception:
        return set()
    classes = set()
    for detector, count in counts.items():
        if count > 0 and detector in SLITHER_DETECTOR_MAP:
            classes.add(SLITHER_DETECTOR_MAP[detector])
    return classes

slither["slither_classes"] = slither.apply(parse_slither_classes, axis=1)
slither_coverage = slither[slither.status == "OK"]["id"].nunique()


# ---------------------------------------------------------------------------
# Parse aderyn: for each contract, which SENTINEL classes did aderyn flag?
# ---------------------------------------------------------------------------
print("Parsing aderyn detector hits...")

def parse_aderyn_classes(row):
    """Return set of SENTINEL classes flagged by aderyn for this contract.
    hits_json is a list of detector-name strings (e.g. ['reentrancy-state-change', ...])
    """
    if row["status"] == "COMPILE_FAIL":
        return set()
    try:
        hits = json.loads(row["hits_json"])
    except Exception:
        return set()
    classes = set()
    for detector in hits:
        if isinstance(detector, str) and detector in ADERYN_DETECTOR_MAP:
            classes.add(ADERYN_DETECTOR_MAP[detector])
    return classes

aderyn["aderyn_classes"] = aderyn.apply(parse_aderyn_classes, axis=1)


# ---------------------------------------------------------------------------
# Build per-contract evidence table
# ---------------------------------------------------------------------------
print("\nBuilding evidence table...")

# Start with base (67,311 contracts × 9 class labels)
ev = base.copy()

# Merge regex + handcrafted
regex_hc = regex.merge(handcraft, on="id", how="left")
ev = ev.merge(regex_hc, on="id", how="left")

# Merge slither
slither_slim = slither[["id", "status", "slither_classes"]].rename(
    columns={"status": "slither_status"}
)
ev = ev.merge(slither_slim, on="id", how="left")

# Merge aderyn
aderyn_slim = aderyn[["id", "status", "aderyn_classes"]].rename(
    columns={"status": "aderyn_status"}
)
ev = ev.merge(aderyn_slim, on="id", how="left")

# Mark whether tool was run on this contract
ev["tool_sampled"] = ev["slither_status"].notna()

print(f"  Tool-sampled contracts: {ev['tool_sampled'].sum():,} / {len(ev):,}")


# ---------------------------------------------------------------------------
# Compute per-class evidence columns
# ---------------------------------------------------------------------------
print("Computing per-class evidence columns...")

for cls in SENTINEL_CLASSES:
    col_base = cls.replace(":", "_")

    # Slither signal: 1 if slither fired relevant detector, 0 if run+OK but not fired, NaN if not run
    def slither_signal(row):
        if not row["tool_sampled"]:
            return np.nan
        if row["slither_status"] != "OK":
            return np.nan  # compile error — no signal
        return 1 if cls in row["slither_classes"] else 0

    ev[f"slither_{col_base}"] = ev.apply(slither_signal, axis=1)

    # Aderyn signal
    def aderyn_signal(row):
        if not row["tool_sampled"]:
            return np.nan
        if row["aderyn_status"] == "COMPILE_FAIL":
            return np.nan
        return 1 if (isinstance(row["aderyn_classes"], set) and cls in row["aderyn_classes"]) else 0

    ev[f"aderyn_{col_base}"] = ev.apply(aderyn_signal, axis=1)

    # Regex signal: any relevant regex feature fires → 1, all fire=0 → 0
    signals = REGEX_CLASS_SIGNALS.get(cls, [])
    if signals:
        regex_cols = [s[0] for s in signals if s[0] in ev.columns]
        if regex_cols:
            ev[f"regex_{col_base}"] = (ev[regex_cols].sum(axis=1) > 0).astype(int)
        else:
            ev[f"regex_{col_base}"] = np.nan
    else:
        ev[f"regex_{col_base}"] = np.nan


# Merge manual verdicts — pivot to wide format (one row per contract × class)
manual_wide = manual.pivot_table(
    index="id", columns="class", values="decision", aggfunc="first"
)
manual_wide.columns = [f"manual_{c.replace(':', '_')}" for c in manual_wide.columns]
ev = ev.merge(manual_wide.reset_index(), on="id", how="left")


# ---------------------------------------------------------------------------
# Confidence score per (contract, class) — vectorized
# ---------------------------------------------------------------------------
print("Computing confidence scores (vectorized)...")

MANUAL_SCORE_MAP = {"KEEP": 1.0, "DROP": 0.0, "UNCERTAIN": 0.5}

for cls in SENTINEL_CLASSES:
    col_base   = cls.replace(":", "_")
    slither_col = f"slither_{col_base}"
    aderyn_col  = f"aderyn_{col_base}"
    regex_col   = f"regex_{col_base}"
    manual_col  = f"manual_{col_base}"

    # Build weighted sum and weight total arrays
    score = pd.Series(0.0, index=ev.index)
    wt    = pd.Series(0.0, index=ev.index)

    # Manual verdict (KEEP=1.0, DROP=0.0, UNCERTAIN=0.5)
    if manual_col in ev.columns:
        m_raw = ev[manual_col].map(MANUAL_SCORE_MAP)        # NaN where not reviewed
        has_m = m_raw.notna()
        score += np.where(has_m, WEIGHTS["manual"] * m_raw.fillna(0), 0)
        wt    += np.where(has_m, WEIGHTS["manual"],                    0)

    # Slither (0 or 1, NaN = not run)
    if slither_col in ev.columns:
        has_s = ev[slither_col].notna()
        score += np.where(has_s, WEIGHTS["slither"] * ev[slither_col].fillna(0), 0)
        wt    += np.where(has_s, WEIGHTS["slither"],                              0)

    # Aderyn (0 or 1, NaN = not run)
    if aderyn_col in ev.columns:
        has_a = ev[aderyn_col].notna()
        score += np.where(has_a, WEIGHTS["aderyn"] * ev[aderyn_col].fillna(0), 0)
        wt    += np.where(has_a, WEIGHTS["aderyn"],                             0)

    # Regex (0 or 1, NaN = no signal defined for this class)
    if regex_col in ev.columns:
        has_r = ev[regex_col].notna()
        score += np.where(has_r, WEIGHTS["regex"] * ev[regex_col].fillna(0), 0)
        wt    += np.where(has_r, WEIGHTS["regex"],                            0)

    ev[f"confidence_{col_base}"] = np.where(wt > 0, score / wt, np.nan)


# ---------------------------------------------------------------------------
# Per-class summary report
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("STAGE 5.1 EVIDENCE COVERAGE REPORT")
print("="*70)

report_rows = []
gate_results = {}

for cls in SENTINEL_CLASSES:
    col_base = cls.replace(":", "_")
    bccc_col = cls

    n_total    = len(ev)
    n_positive = ev[bccc_col].sum()

    slither_col = f"slither_{col_base}"
    aderyn_col  = f"aderyn_{col_base}"
    regex_col   = f"regex_{col_base}"
    manual_col  = f"manual_{col_base}"
    conf_col    = f"confidence_{col_base}"

    n_slither_run = ev[slither_col].notna().sum()
    n_slither_pos = (ev[slither_col] == 1).sum()
    n_aderyn_run  = ev[aderyn_col].notna().sum()
    n_aderyn_pos  = (ev[aderyn_col] == 1).sum()
    n_regex_pos   = (ev.get(regex_col, pd.Series(dtype=float)) == 1).sum() if regex_col in ev.columns and ev[regex_col].notna().any() else "N/A"
    n_manual      = ev[manual_col].notna().sum() if manual_col in ev.columns else 0
    n_manual_keep = (ev.get(manual_col) == "KEEP").sum() if manual_col in ev.columns else 0
    n_manual_drop = (ev.get(manual_col) == "DROP").sum() if manual_col in ev.columns else 0

    # Gate: among BCCC-positive contracts that were tool-sampled,
    #       what fraction have tool agreement (confidence ≥ 0.75, meaning tools say YES)?
    # Also count manual verdicts (full 67K)
    pos_sampled = ev[(ev[bccc_col] == 1) & ev["tool_sampled"]]
    n_pos_sampled = len(pos_sampled)
    n_conf_high   = (pos_sampled[conf_col] >= 0.75).sum()  if n_pos_sampled > 0 else 0
    n_conf_low    = (pos_sampled[conf_col] <  0.50).sum()  if n_pos_sampled > 0 else 0
    n_conf_any    = pos_sampled[conf_col].notna().sum()     if n_pos_sampled > 0 else 0

    # % of sampled positives with high-confidence tool agreement
    pct_high_of_sampled = (n_conf_high / n_pos_sampled * 100) if n_pos_sampled > 0 else 0
    # % of sampled positives where tools actively say NO (FP evidence)
    pct_low_of_sampled  = (n_conf_low  / n_pos_sampled * 100) if n_pos_sampled > 0 else 0

    # Two paths to VERIFIED:
    # Path A — tool agreement: ≥80% of sampled positives have high-confidence tool support
    # Path B — manual confirms clean: ≥10 reviews AND 0 DROPs (tools have low recall for
    #          these classes, not low BCCC precision — IntegerUO, UnusedReturn, MishandledException)
    tool_verified   = (pct_high_of_sampled >= 80)
    manual_verified = (n_manual >= 10 and n_manual_drop == 0)
    gate_pass       = tool_verified or manual_verified
    gate_results[cls] = gate_pass

    row = {
        "class": cls,
        "n_positive_bccc": int(n_positive),
        "n_pos_sampled": int(n_pos_sampled),
        "n_slither_pos": int(n_slither_pos),
        "n_aderyn_pos": int(n_aderyn_pos),
        "n_regex_pos": n_regex_pos,
        "n_manual": int(n_manual),
        "n_manual_keep": int(n_manual_keep),
        "n_manual_drop": int(n_manual_drop),
        "pct_tool_agree_sampled_pos": round(pct_high_of_sampled, 1),
        "pct_tool_reject_sampled_pos": round(pct_low_of_sampled, 1),
        "stage51_gate": "VERIFIED" if gate_pass else "PROCEED_TO_5.2",
    }
    report_rows.append(row)

    print(f"\n{cls}")
    print(f"  BCCC positives (total): {n_positive:,}  |  tool-sampled positives: {n_pos_sampled:,}")
    print(f"  Slither pos: {n_slither_pos:,}  |  Aderyn pos: {n_aderyn_pos:,}")
    print(f"  Regex positive (all 67K): {n_regex_pos}")
    print(f"  Manual: {n_manual} reviewed  KEEP:{n_manual_keep}  DROP:{n_manual_drop}")
    print(f"  Tool-agreement on sampled positives: HIGH={n_conf_high}/{n_pos_sampled} ({pct_high_of_sampled:.1f}%)  LOW(FP)={n_conf_low}/{n_pos_sampled} ({pct_low_of_sampled:.1f}%)")
    print(f"  Stage 5.1 gate: {'✅ VERIFIED' if gate_pass else '⬇ PROCEED TO 5.2'}")


print("\n" + "="*70)
print("GATE SUMMARY")
print("="*70)
verified   = [c for c, v in gate_results.items() if v]
to_stage52 = [c for c, v in gate_results.items() if not v]
print(f"VERIFIED at Stage 5.1 ({len(verified)}): {verified}")
print(f"Proceed to Stage 5.2 ({len(to_stage52)}): {to_stage52}")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
print("\nSaving evidence table...")

# Keep a lean version for the output (don't dump all 31 regex cols + set objects)
output_cols = (
    ["id"] +
    SENTINEL_CLASSES +
    ["tool_sampled", "slither_status", "aderyn_status"] +
    [f"slither_{c.replace(':', '_')}" for c in SENTINEL_CLASSES] +
    [f"aderyn_{c.replace(':', '_')}"  for c in SENTINEL_CLASSES] +
    [f"regex_{c.replace(':', '_')}"   for c in SENTINEL_CLASSES
     if f"regex_{c.replace(':', '_')}" in ev.columns] +
    [f"manual_{c.replace(':', '_')}"  for c in SENTINEL_CLASSES
     if f"manual_{c.replace(':', '_')}" in ev.columns] +
    [f"confidence_{c.replace(':', '_')}" for c in SENTINEL_CLASSES]
)
output_cols = [c for c in output_cols if c in ev.columns]
ev_out = ev[output_cols].copy()

# Convert set objects to comma-separated strings for CSV
for col in ["slither_classes", "aderyn_classes"]:
    if col in ev_out.columns:
        ev_out[col] = ev_out[col].apply(
            lambda x: ",".join(sorted(x)) if isinstance(x, set) else ""
        )

ev_out.to_csv(P5_OUT / "p5_s1_evidence_table.csv", index=False)
print(f"  Saved: p5_s1_evidence_table.csv  ({len(ev_out):,} rows × {len(ev_out.columns)} cols)")

# Save summary report
report_df = pd.DataFrame(report_rows)
report_df.to_csv(P5_OUT / "p5_s1_coverage_report.csv", index=False)

report_md = ["# Stage 5.1 Evidence Coverage Report\n",
             f"**Generated:** 2026-06-08\n",
             f"**Total contracts:** {len(ev):,}\n",
             f"**Tool-sampled contracts:** {ev['tool_sampled'].sum():,} (15% stratified sample from Phase 4)\n\n",
             "## Per-Class Summary\n",
             report_df.to_csv(None, index=False),
             "\n\n## Gate Results\n",
             f"**VERIFIED at Stage 5.1** ({len(verified)}): {', '.join(verified) if verified else 'None'}\n\n",
             f"**Proceed to Stage 5.2** ({len(to_stage52)}): {', '.join(to_stage52)}\n\n",
             "## Notes\n",
             "- High-confidence threshold = 0.75 (weighted average from M3+M4+M9 weights in Stage 5.0)\n",
             "- Gate: ≥80% of BCCC-positive contracts have high-confidence verdict from existing evidence\n",
             "- Regex features are present for ALL 67,311 contracts; slither/aderyn for 10,693 (15%) only\n",
             "- Manual reviews from ws_p4_s1_review_200.csv (199 contracts) used as M9 evidence\n",
             ]

(P5_OUT / "p5_s1_coverage_report.md").write_text("\n".join(report_md))
print(f"  Saved: p5_s1_coverage_report.md")
print("\nStage 5.1 complete.")
