"""Phase 5 Gap Fixes — addresses two structural weaknesses in Stage 5.4:

  GAP A — Reentrancy definition too narrow (only ETH-transfer .call.value):
    Extended to detect non-ETH external calls + ERC token callbacks +
    CEI violation proxy (state write after external call).

  GAP B — Single-file scanning misses inherited/imported vulnerabilities:
    Fix B1: Recover contracts where Slither=1 but we said DROP.
             Slither follows full compile tree; its signal overrides our regex DROP.
    Fix B2: Import follower — parse import statements, scan local imported files
             with same class patterns.
    Fix B3: When full-dataset Slither results available (p5_gap_full_slither_results.csv),
             apply the same Slither-recovery logic to ALL contracts.

Run from repo root:
    python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_gap_fixes.py

Re-run after full Slither completes for maximum recovery.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(".")
P4_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs"
P5_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"

# ─── Slither detector → SENTINEL class map ──────────────────────────────────
SLITHER_CLASS_MAP = {
    "reentrancy-eth":           "Class11:Reentrancy",
    "reentrancy-no-eth":        "Class11:Reentrancy",
    "reentrancy-benign":        "Class11:Reentrancy",
    "reentrancy-events":        "Class11:Reentrancy",
    "unused-return":            "Class06:UnusedReturn",
    "unchecked-transfer":       "Class03:MishandledException",
    "unchecked-send":           "Class03:MishandledException",
    "unchecked-lowlevel":       "Class03:MishandledException",
    "costly-loop":              "Class02:GasException",
    "calls-loop":               "Class09:DenialOfService",
    "timestamp":                "Class04:Timestamp",
    "tx-origin":                "Class01:ExternalBug",
    "suicidal":                 "Class01:ExternalBug",
    "controlled-delegatecall":  "Class08:CallToUnknown",
    "divide-before-multiply":   "Class10:IntegerUO",
}

ADERYN_CLASS_MAP = {
    "reentrancy-state-change":   "Class11:Reentrancy",
    "unchecked-return":          "Class03:MishandledException",
    "unchecked-send":            "Class03:MishandledException",
    "unchecked-low-level-call":  "Class03:MishandledException",
    "selfdestruct":              "Class01:ExternalBug",
    "centralization-risk":       "Class01:ExternalBug",
    "ecrecover":                 "Class01:ExternalBug",
    "unsafe-erc20-operation":    "Class03:MishandledException",
    "weak-randomness":           "Class04:Timestamp",
    "division-before-multiplication": "Class10:IntegerUO",
}

def slither_hits_to_classes(hits_json: str) -> set:
    """Convert a Slither hits_json string to a set of SENTINEL class names."""
    try:
        hits = json.loads(hits_json) if isinstance(hits_json, str) else hits_json
    except Exception:
        return set()
    classes = set()
    for h in hits:
        det = h if isinstance(h, str) else h.get("check", "")
        if det in SLITHER_CLASS_MAP:
            classes.add(SLITHER_CLASS_MAP[det])
    return classes

def aderyn_hits_to_classes(hits_json: str) -> set:
    try:
        hits = json.loads(hits_json) if isinstance(hits_json, str) else hits_json
    except Exception:
        return set()
    classes = set()
    for h in hits:
        det = h if isinstance(h, str) else h.get("check", "")
        if det in ADERYN_CLASS_MAP:
            classes.add(ADERYN_CLASS_MAP[det])
    return classes

# ─── Load datasets ───────────────────────────────────────────────────────────

print("Loading Stage 5.4 final verdicts...")
s4 = pd.read_csv(P5_OUT / "p5_s4_final_verdict.csv")
print(f"  {len(s4):,} verdict rows")

print("Loading base dataset + source paths...")
base = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv",
                   usecols=["id", "bccc_file_path"])
s4 = s4.merge(base, on="id", how="left")

print("Loading Phase 4 Slither results (10,693)...")
p4_slither = pd.read_csv(P4_OUT / "ws_p4_s1_slither_results.csv",
                         usecols=["id", "status", "hits_json"])
p4_slither["slither_ok"] = p4_slither["status"] == "OK"
p4_slither["slither_classes"] = p4_slither["hits_json"].apply(slither_hits_to_classes)

print("Loading Phase 4 Aderyn results (10,693)...")
p4_aderyn = pd.read_csv(P4_OUT / "ws_p4_s1_aderyn_results.csv",
                        usecols=["id", "status", "hits_json"])
p4_aderyn["aderyn_ok"] = p4_aderyn["status"] == "OK"
p4_aderyn["aderyn_classes"] = p4_aderyn["hits_json"].apply(aderyn_hits_to_classes)

# Load full-dataset Slither if available
FULL_SLITHER = P5_OUT / "p5_gap_full_slither_results.csv"
if FULL_SLITHER.exists():
    print(f"Loading full-dataset Slither results...")
    full_sl = pd.read_csv(FULL_SLITHER, usecols=["id", "status", "hits_json"])
    full_sl["slither_ok"] = full_sl["status"] == "OK"
    full_sl["slither_classes"] = full_sl["hits_json"].apply(slither_hits_to_classes)
    print(f"  {len(full_sl):,} rows; OK={full_sl['slither_ok'].sum():,}")
else:
    full_sl = None
    print("  Full Slither not yet available — using Phase 4 (10,693 only)")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def resolve_path(bcck: str) -> Path:
    return ROOT / str(bcck).replace("Source Codes/", "SourceCodes/")

def read_src(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

# ─── GAP A: Extended Reentrancy patterns ─────────────────────────────────────

RE_NONREENTRANT    = re.compile(r'nonReentrant|ReentrancyGuard', re.S)
RE_CALLVAL_PRE08   = re.compile(r'\.call\.value\s*\([^)]*\)\s*\(', re.S)
RE_CALLVAL_POST08  = re.compile(r'\.call\s*\{[^}]*value\s*:', re.S)
RE_CALL_DATA       = re.compile(r'\.\bcall\b\s*\(', re.S)          # .call(data) without value
RE_ERC_CALLBACK    = re.compile(
    r'\bonERC1155Received\b|\bonERC721Received\b|\btokenFallback\b'
    r'|\btokensReceived\b|\bonTokenTransfer\b',
    re.S)
# CEI violation proxy: state variable write AFTER external call in same block
# (simplified: mapping[addr] -= / += after a .call or .send or .transfer)
RE_BALANCE_STATE   = re.compile(
    r'(?:balances?|amounts?|deposits?|funds?|credits?|stakes?|rewards?)\s*\[[^\]]+\]\s*[+\-]?=',
    re.S | re.I)
# NOTE: .send() and .transfer() forward only 2300 gas — NOT reentrancy vectors.
# Only .call() allows re-entry. Do NOT use RE_EXT_CALL_SIMPLE here.

def reentrancy_extended_check(src: str) -> tuple:
    """
    Extended reentrancy check beyond .call.value().
    Returns (is_reentrancy_candidate, pattern_found).

    IMPORTANT: .transfer() and .send() are NOT reentrancy vectors (2300 gas limit).
    Only .call() (with or without value) enables reentrancy.
    """
    if bool(RE_NONREENTRANT.search(src)):
        return False, "has_reentrancy_guard"

    has_eth_call  = bool(RE_CALLVAL_PRE08.search(src)) or bool(RE_CALLVAL_POST08.search(src))
    has_data_call = bool(RE_CALL_DATA.search(src))
    has_erc_cb    = bool(RE_ERC_CALLBACK.search(src))
    has_state     = bool(RE_BALANCE_STATE.search(src))

    if has_eth_call:
        return True, "eth_call_value"
    if has_erc_cb and has_state:
        return True, "erc_callback_with_state_write"
    if has_data_call and has_state:
        # .call(data) without ETH value + state write — possible non-ETH reentrancy
        return True, "call_data_with_state_write"
    return False, "no_reentrancy_pattern"


# ─── GAP B-2: Import follower ─────────────────────────────────────────────────

RE_IMPORT = re.compile(
    r'''import\s+(?:['"]([^'"]+)['"]|[^'";\n]*\s+from\s+['"]([^'"]+)['"]);''',
    re.S)

def parse_imports(src: str) -> list:
    """Extract import paths from Solidity source."""
    imports = []
    for m in RE_IMPORT.finditer(src):
        path = m.group(1) or m.group(2)
        if path:
            imports.append(path)
    return imports

def resolve_import(import_path: str, contract_path: Path) -> Path | None:
    """Resolve an import path relative to the contract file."""
    if import_path.startswith(("http://", "https://", "@")):
        return None  # external/npm import — can't resolve
    # Resolve relative to contract directory
    base_dir = contract_path.parent
    candidate = (base_dir / import_path).resolve()
    if candidate.exists():
        return candidate
    # Try without leading ./
    clean = import_path.lstrip("./")
    candidate2 = (base_dir / clean).resolve()
    if candidate2.exists():
        return candidate2
    return None

# Class-specific pattern checks (used in import scanning)
CLASS_PATTERNS = {
    "Class11:Reentrancy": [
        re.compile(r'\.call\.value\s*\([^)]*\)\s*\(', re.S),
        re.compile(r'\.call\s*\{[^}]*value\s*:', re.S),
        re.compile(r'\.\bcall\b\s*\([^)]{0,60}abi\.encode', re.S),
    ],
    "Class08:CallToUnknown": [
        re.compile(r'\.call\s*[\(\{]|\.delegatecall\s*\(|\.staticcall\s*\(', re.S),
    ],
    "Class04:Timestamp": [
        re.compile(r'\bblock\.timestamp\b|\bnow\b', re.S),
    ],
    "Class01:ExternalBug": [
        re.compile(r'\bselfdestruct\s*\(|\bsuicide\s*\(|\btx\.origin\b|\.delegatecall\s*\(', re.S),
    ],
    "Class02:GasException": [
        re.compile(r'\bfor\s*\(|\bwhile\s*\(', re.S),
    ],
    "Class09:DenialOfService": [
        re.compile(r'require\s*\([^;]{0,80}\.(?:send|transfer)\s*\(', re.S),
    ],
}

def scan_imports_for_class(src: str, contract_path: Path, class_col: str) -> bool:
    """
    Recursively scan imported files for class-specific patterns.
    Returns True if any imported file has a matching pattern.
    """
    patterns = CLASS_PATTERNS.get(class_col, [])
    imports = parse_imports(src)
    seen = {str(contract_path)}
    queue = imports[:]
    base_dir = contract_path.parent

    while queue:
        imp_path = queue.pop(0)
        resolved = resolve_import(imp_path, contract_path)
        if not resolved or str(resolved) in seen:
            continue
        seen.add(str(resolved))
        imp_src = read_src(resolved)
        if not imp_src:
            continue
        for pat in patterns:
            if pat.search(imp_src):
                return True
        # Follow nested imports (1 level deep only to avoid cycles)
        if len(seen) < 10:
            for nested in parse_imports(imp_src):
                queue.append(nested)
    return False

# ─── Apply all gap fixes ──────────────────────────────────────────────────────

print("\n" + "="*65)
print("APPLYING GAP FIXES")
print("="*65)

recovered = 0
gap_a_count = 0
gap_b1_p4_count = 0
gap_b1_full_count = 0
gap_b2_count = 0

NOISY_CLASSES = [
    "Class11:Reentrancy", "Class08:CallToUnknown", "Class04:Timestamp",
    "Class01:ExternalBug", "Class02:GasException", "Class09:DenialOfService",
]

# Build lookup: id → Phase 4 slither classes
p4_slither_lookup = {row["id"]: row["slither_classes"]
                     for _, row in p4_slither.iterrows() if row["slither_ok"]}
p4_aderyn_lookup  = {row["id"]: row["aderyn_classes"]
                     for _, row in p4_aderyn.iterrows() if row["aderyn_ok"]}
full_sl_lookup    = ({row["id"]: row["slither_classes"]
                      for _, row in full_sl.iterrows() if row["slither_ok"]}
                     if full_sl is not None else {})

print(f"\n  Phase 4 Slither coverage: {len(p4_slither_lookup):,} contracts")
print(f"  Phase 4 Aderyn coverage:  {len(p4_aderyn_lookup):,} contracts")
print(f"  Full Slither coverage:    {len(full_sl_lookup):,} contracts")

# Process each noisy class
for cls in NOISY_CLASSES:
    cls_rows = s4[s4["class"] == cls].copy()
    drop_rows = cls_rows[cls_rows["verdict"] == "DROP"]
    print(f"\n{cls}: {len(cls_rows):,} total, {len(drop_rows):,} DROPs to re-examine")

    n_fix_a = n_fix_b1_p4 = n_fix_b1_full = n_fix_b2 = 0

    for idx, row in drop_rows.iterrows():
        cid = row["id"]
        path = resolve_path(row["bccc_file_path"])
        fixed = False

        # ── Gap B1: Slither/Aderyn confirmed (Phase 4 data) ──────────────────
        p4_sl_cls = p4_slither_lookup.get(cid, set())
        p4_ad_cls = p4_aderyn_lookup.get(cid, set())
        if cls in p4_sl_cls:
            s4.at[idx, "verdict"]    = "KEEP"
            s4.at[idx, "confidence"] = 0.78
            s4.at[idx, "notes"]      = "gap_b1:slither_p4_confirmed"
            fixed = True; n_fix_b1_p4 += 1

        elif cls in p4_ad_cls and not fixed:
            s4.at[idx, "verdict"]    = "KEEP"
            s4.at[idx, "confidence"] = 0.72
            s4.at[idx, "notes"]      = "gap_b1:aderyn_p4_confirmed"
            fixed = True; n_fix_b1_p4 += 1

        # ── Gap B1: Full Slither (new data) ───────────────────────────────────
        if not fixed and cid in full_sl_lookup:
            full_cls = full_sl_lookup[cid]
            if cls in full_cls:
                s4.at[idx, "verdict"]    = "KEEP"
                s4.at[idx, "confidence"] = 0.78
                s4.at[idx, "notes"]      = "gap_b1:slither_full_confirmed"
                fixed = True; n_fix_b1_full += 1

        # ── Gap A: Extended Reentrancy ────────────────────────────────────────
        if not fixed and cls == "Class11:Reentrancy":
            src = read_src(path)
            is_re, pattern = reentrancy_extended_check(src)
            if is_re and pattern != "has_reentrancy_guard":
                s4.at[idx, "verdict"]    = "KEEP"
                s4.at[idx, "confidence"] = 0.65
                s4.at[idx, "notes"]      = f"gap_a:extended_reentrancy:{pattern}"
                fixed = True; n_fix_a += 1

        # ── Gap B2: Import follower ───────────────────────────────────────────
        if not fixed:
            src = read_src(path) if cls != "Class11:Reentrancy" else read_src(path)
            src = read_src(path)
            if scan_imports_for_class(src, path, cls):
                s4.at[idx, "verdict"]    = "KEEP"
                s4.at[idx, "confidence"] = 0.65
                s4.at[idx, "notes"]      = "gap_b2:import_chain_pattern"
                fixed = True; n_fix_b2 += 1

    total_fixed = n_fix_b1_p4 + n_fix_b1_full + n_fix_a + n_fix_b2
    print(f"  Recovered: {total_fixed:,}  "
          f"(B1-P4={n_fix_b1_p4}, B1-Full={n_fix_b1_full}, "
          f"A-ExtRe={n_fix_a}, B2-Import={n_fix_b2})")
    gap_a_count     += n_fix_a
    gap_b1_p4_count += n_fix_b1_p4
    gap_b1_full_count += n_fix_b1_full
    gap_b2_count    += n_fix_b2
    recovered       += total_fixed

print(f"\n{'='*65}")
print(f"TOTAL RECOVERED: {recovered:,}")
print(f"  Gap A (extended reentrancy): {gap_a_count:,}")
print(f"  Gap B1 Phase4 Slither/Aderyn: {gap_b1_p4_count:,}")
print(f"  Gap B1 Full Slither:          {gap_b1_full_count:,}")
print(f"  Gap B2 Import follower:       {gap_b2_count:,}")

# ─── Save gap-fixed verdict ──────────────────────────────────────────────────

s4.drop(columns=["bccc_file_path"], inplace=True, errors="ignore")
s4.to_csv(P5_OUT / "p5_gap_fixed_verdict.csv", index=False)
print(f"\nSaved p5_gap_fixed_verdict.csv  ({len(s4):,} rows)")

# ─── Re-run synthesis ────────────────────────────────────────────────────────

print("\n" + "="*65)
print("RE-SYNTHESIZING contracts_clean_v1.4.csv")
print("="*65)

base_full = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv")
CLASS_COLS = [c for c in base_full.columns if c.startswith("Class")]
NV_COL = "Class12:NonVulnerable"
active_label_cols = [c for c in CLASS_COLS if c != NV_COL]

# Build lookup: (id, class) → verdict
verdict_lookup = {(r["id"], r["class"]): r for _, r in s4.iterrows()}

df = base_full.copy()
for cls in NOISY_CLASSES:
    for c in [f"p5_verdict_{cls}", f"p5_confidence_{cls}"]:
        df[c] = np.nan if "confidence" in c else "not_positive"

labels_before = {c: int(base_full[c].sum()) for c in CLASS_COLS}
dropped = kept = 0

for cls in NOISY_CLASSES:
    n_pos = int(df[cls].sum())
    n_drop = n_keep = 0
    for idx, row in df[df[cls] == 1].iterrows():
        cid = row["id"]
        key = (cid, cls)
        if key in verdict_lookup:
            v = verdict_lookup[key]
            df.at[idx, f"p5_verdict_{cls}"]    = v["verdict"]
            df.at[idx, f"p5_confidence_{cls}"] = v["confidence"]
            if v["verdict"] == "DROP":
                df.at[idx, cls] = 0
                n_drop += 1
            else:
                n_keep += 1
    dropped += n_drop; kept += n_keep
    print(f"  {cls}: {n_pos:,} → KEEP {n_keep:,} / DROP {n_drop:,}")

# All-labels-dropped → NonVulnerable
all_zero = (df[active_label_cols].sum(axis=1) == 0)
prev_had = (base_full[active_label_cols].sum(axis=1) > 0)
newly_nv = all_zero & prev_had
df.loc[newly_nv, NV_COL] = 1
print(f"\n  Newly NonVulnerable: {newly_nv.sum():,}")
print(f"  Labels dropped: {dropped:,}  |  kept: {kept:,}")

# D-I-11/12 check
for c in [c for c in NOISY_CLASSES if c in df.columns]:
    viol = (df[NV_COL] == 1) & (df[c] == 1)
    if viol.sum(): df.loc[viol, NV_COL] = 0

out_path = P5_OUT / "contracts_clean_v1.4.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(df):,} × {len(df.columns)} cols)")

# ─── Comparison: v1.3 vs v1.4 ────────────────────────────────────────────────
print("\n=== Class size: v1.3 → v1.4 (gap-fixed) ===")
v13 = pd.read_csv(P5_OUT / "contracts_clean_v1.3.csv", usecols=CLASS_COLS)
v14 = df[CLASS_COLS]

print(f"\n{'Class':<35} {'v1.3':>7} {'v1.4':>7} {'Δ':>7}")
print("-" * 55)
for c in CLASS_COLS:
    b = int(v13[c].sum())
    a = int(v14[c].sum())
    flag = " ← recovered" if a > b else ""
    print(f"{c:<35} {b:>7,} {a:>7,} {a-b:>+7,}{flag}")

print(f"\nContracts with ≥1 active label: "
      f"{(df[active_label_cols].sum(axis=1)>0).sum():,}  "
      f"(was {(v13[active_label_cols].sum(axis=1)>0).sum():,})")

print("\nGap fixes complete.")
