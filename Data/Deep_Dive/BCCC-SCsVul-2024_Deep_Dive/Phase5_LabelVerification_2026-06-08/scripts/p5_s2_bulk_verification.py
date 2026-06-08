"""Stage 5.2: Bulk Automated Verification
Per-class source-code regex + existing tool evidence to assign automated verdicts
to all BCCC-positive contracts in the 6 noisy classes.

Run from repo root:
    poetry run python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_s2_bulk_verification.py
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(".")
BCCC_ROOT = ROOT / "BCCC-SCsVul-2024"
P4_OUT    = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs"
P5_OUT    = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"
P5_EV     = P5_OUT / "p5_s1_evidence_table.csv"
P5_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading dataset + evidence table...")
df = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv",
                 usecols=["id", "bccc_file_path",
                          "Class01:ExternalBug", "Class02:GasException",
                          "Class03:MishandledException", "Class04:Timestamp",
                          "Class06:UnusedReturn", "Class08:CallToUnknown",
                          "Class09:DenialOfService", "Class10:IntegerUO",
                          "Class11:Reentrancy"])

# Load Stage 5.1 slither/aderyn per-contract signals
ev_cols = ["id",
           "slither_Class11_Reentrancy", "slither_Class08_CallToUnknown",
           "slither_Class01_ExternalBug", "slither_Class02_GasException",
           "slither_Class09_DenialOfService", "slither_Class04_Timestamp",
           "aderyn_Class11_Reentrancy", "aderyn_Class08_CallToUnknown",
           "aderyn_Class01_ExternalBug"]
ev = pd.read_csv(P5_EV, usecols=[c for c in ev_cols
                                  if c in pd.read_csv(P5_EV, nrows=0).columns])
df = df.merge(ev, on="id", how="left")
print(f"  Dataset: {len(df):,} contracts loaded")

def resolve_path(bccc_file_path: str) -> Path:
    """Convert bccc_file_path (with 'Source Codes') to actual disk path."""
    fixed = bccc_file_path.replace("Source Codes/", "SourceCodes/")
    return ROOT / fixed

def read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Per-class verification logic
# ---------------------------------------------------------------------------

# --- Reentrancy ---
RE_CALLVAL_PRE08  = re.compile(r'\.call\.value\s*\([^)]*\)\s*\(', re.S)
RE_CALLVAL_POST08 = re.compile(r'\.call\s*\{[^}]*value\s*:', re.S)
RE_TRANSFER       = re.compile(r'\.(transfer|send)\s*\(', re.S)
RE_ANY_EXTERNAL   = re.compile(r'\.(call|send|transfer)\s*[\(\{]', re.S)
RE_NONREENTRANT   = re.compile(r'nonReentrant|ReentrancyGuard', re.S)
RE_CEI_ZERO       = re.compile(r'=\s*0\s*;[^}]{0,200}\.call\.value', re.S)  # balance zeroed before call

def verify_reentrancy(src: str, sigs: dict) -> tuple:
    """Returns (verdict, confidence, notes)"""
    slither_re = sigs.get("slither_Class11_Reentrancy", np.nan)
    has_pre  = bool(RE_CALLVAL_PRE08.search(src))
    has_post = bool(RE_CALLVAL_POST08.search(src))
    has_callval = has_pre or has_post
    has_guard   = bool(RE_NONREENTRANT.search(src))

    if not has_callval:
        return "DROP", 0.85, "no_callvalue_pattern"
    if has_guard:
        return "DROP", 0.75, "reentrancy_guard_present"
    if slither_re == 1:
        return "KEEP", 0.80, "callvalue+slither_agree"
    return "UNCERTAIN", 0.50, "callvalue_no_slither"

# --- CallToUnknown ---
RE_LOWLEVEL  = re.compile(r'\.call\s*[\(\{]|\.delegatecall\s*\(|\.staticcall\s*\(', re.S)
RE_CALL_SELF = re.compile(r'address\(this\)\.call|this\.call', re.S)

def verify_callto_unknown(src: str, sigs: dict) -> tuple:
    slither_ctu = sigs.get("slither_Class08_CallToUnknown", np.nan)
    has_ll  = bool(RE_LOWLEVEL.search(src))
    has_self = bool(RE_CALL_SELF.search(src))
    if not has_ll:
        return "DROP", 0.85, "no_lowlevel_call"
    if has_ll and has_self and not bool(re.search(r'\.call\s*\((?!.*address\(this\))', src, re.S)):
        return "DROP", 0.70, "only_self_call"
    if slither_ctu == 1:
        return "KEEP", 0.78, "lowlevel+slither_agree"
    return "UNCERTAIN", 0.45, "lowlevel_unconfirmed_target"

# --- Timestamp ---
RE_TS_ANY     = re.compile(r'\bblock\.timestamp\b|\bnow\b', re.S)
RE_TS_BRANCH  = re.compile(
    r'(?:if|require|assert)\s*\([^;]{0,120}(?:block\.timestamp|now\b)',
    re.S | re.I)
RE_TS_ARITH   = re.compile(r'(?:block\.timestamp|now\b)\s*[+\-\*/%]', re.S)
RE_TS_MODULO  = re.compile(r'(?:block\.timestamp|now\b)\s*%', re.S)
RE_TS_KECCAK  = re.compile(r'keccak256\s*\([^)]{0,120}(?:block\.timestamp|now\b)', re.S)
RE_TS_ASSIGN  = re.compile(r'=\s*(?:block\.timestamp|now\b)', re.S)
RE_TS_LOG_ONLY= re.compile(r'emit\s+\w+\s*\([^;]{0,200}(?:block\.timestamp|now\b)', re.S)

def verify_timestamp(src: str, sigs: dict) -> tuple:
    slither_ts = sigs.get("slither_Class04_Timestamp", np.nan)
    if not bool(RE_TS_ANY.search(src)):
        return "DROP", 0.90, "no_timestamp_usage"
    critical = (bool(RE_TS_BRANCH.search(src)) or
                bool(RE_TS_MODULO.search(src)) or
                bool(RE_TS_KECCAK.search(src)))
    non_critical_only = bool(RE_TS_LOG_ONLY.search(src)) and not critical
    if non_critical_only and not bool(RE_TS_ASSIGN.search(src)):
        return "DROP", 0.70, "timestamp_in_logs_only"
    if critical:
        conf = 0.75 if slither_ts == 1 else 0.55
        return "KEEP", conf, "timestamp_in_critical_context"
    if bool(RE_TS_ASSIGN.search(src)):
        return "UNCERTAIN", 0.45, "timestamp_stored_not_in_branch"
    return "UNCERTAIN", 0.35, "timestamp_usage_not_critical"

# --- ExternalBug ---
RE_SELFDESTRUCT = re.compile(r'\bselfdestruct\s*\(|\bsuicide\s*\(', re.S)
RE_TX_ORIGIN    = re.compile(r'\btx\.origin\b', re.S)
RE_DELEGATECALL = re.compile(r'\.delegatecall\s*\(', re.S)
RE_ECRECOVER    = re.compile(r'\becrecover\s*\(', re.S)
RE_ONLYOWNER_SD = re.compile(r'onlyOwner[^}]{0,300}selfdestruct|selfdestruct[^}]{0,300}onlyOwner', re.S)
RE_TX_LOG       = re.compile(r'emit\s+\w+[^;]{0,100}tx\.origin|tx\.origin[^;]{0,100}==[^;]{0,100}0x0', re.S)

def verify_external_bug(src: str, sigs: dict) -> tuple:
    slither_eb = sigs.get("slither_Class01_ExternalBug", np.nan)
    aderyn_eb  = sigs.get("aderyn_Class01_ExternalBug", np.nan)
    has_sd    = bool(RE_SELFDESTRUCT.search(src))
    has_txo   = bool(RE_TX_ORIGIN.search(src))
    has_dc    = bool(RE_DELEGATECALL.search(src))
    has_ecr   = bool(RE_ECRECOVER.search(src))
    any_pattern = has_sd or has_txo or has_dc or has_ecr
    if not any_pattern:
        return "DROP", 0.85, "no_extbug_patterns"
    # selfdestruct behind onlyOwner is safe
    sd_safe = has_sd and bool(RE_ONLYOWNER_SD.search(src))
    # tx.origin only in logs/null-check is safe
    txo_safe = has_txo and bool(RE_TX_LOG.search(src)) and not bool(
        re.search(r'(?:require|if)\s*\([^;]{0,80}tx\.origin', src, re.S))
    exploitable = (has_sd and not sd_safe) or (has_txo and not txo_safe) or has_dc or has_ecr
    if not exploitable:
        return "DROP", 0.65, "patterns_present_but_safe"
    tool_agree = (slither_eb == 1 or aderyn_eb == 1)
    if tool_agree:
        return "KEEP", 0.80, "extbug_pattern+tool_agree"
    return "UNCERTAIN", 0.50, "extbug_pattern_unconfirmed"

# --- GasException ---
RE_LOOP       = re.compile(r'\bfor\s*\(|\bwhile\s*\(', re.S)
RE_PUSH       = re.compile(r'\.push\s*\(', re.S)
RE_LOOP_BOUND = re.compile(r'\bfor\s*\([^;]*;\s*\w+\s*<\s*(\d+)\s*;', re.S)  # bounded loop
RE_MAPPING_ITER = re.compile(r'mapping\s*\(', re.S)

def verify_gas_exception(src: str, sigs: dict) -> tuple:
    slither_ge = sigs.get("slither_Class02_GasException", np.nan)
    has_loop   = bool(RE_LOOP.search(src))
    has_push   = bool(RE_PUSH.search(src))
    # bounded constant loops — safe
    bounded_matches = RE_LOOP_BOUND.findall(src)
    all_bounded = has_loop and bounded_matches and not has_push
    if not has_loop:
        return "DROP", 0.80, "no_loop"
    if all_bounded:
        return "DROP", 0.65, "loop_bounded_constant"
    if has_loop and has_push:
        conf = 0.70 if slither_ge == 1 else 0.55
        return "KEEP", conf, "loop_with_push_pattern"
    if slither_ge == 1:
        return "KEEP", 0.72, "slither_costly_loop"
    return "UNCERTAIN", 0.40, "loop_no_push_no_slither"

# --- DenialOfService ---
RE_REQUIRE_SEND = re.compile(
    r'require\s*\([^;]{0,60}\.send\s*\(|require\s*\([^;]{0,60}\.call\s*[\(\{]',
    re.S)
RE_LOOP_SEND    = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,400}\.(?:send|call)\s*[\(\{]'
    r'|\.(?:send|call)\s*[\(\{][^}]{0,400}(?:for|while)\s*\(',
    re.S)
RE_PUSH_PATTERN = re.compile(
    r'\.push\s*\([^;]{0,60}msg\.sender'     # push msg.sender into array
    r'|\baddress\s*\[\s*\]\s*\w+\s*=\s*new',  # dynamic address array allocation
    re.S)

def verify_denial_of_service(src: str, sigs: dict) -> tuple:
    slither_dos = sigs.get("slither_Class09_DenialOfService", np.nan)
    has_loop    = bool(RE_LOOP.search(src))
    req_send    = bool(RE_REQUIRE_SEND.search(src))
    loop_send   = bool(RE_LOOP_SEND.search(src))
    push_pat    = bool(RE_PUSH_PATTERN.search(src))
    if not has_loop and not req_send:
        return "DROP", 0.65, "no_loop_no_require_send"
    if req_send and loop_send:
        conf = 0.72 if slither_dos == 1 else 0.58
        return "KEEP", conf, "loop+require_on_send"
    if push_pat and has_loop:
        return "KEEP", 0.55, "push_pattern_with_loop"
    if slither_dos == 1:
        return "KEEP", 0.68, "slither_calls_loop"
    return "UNCERTAIN", 0.38, "partial_dos_pattern"

# ---------------------------------------------------------------------------
# Helper: get tool signals for a contract (from merged ev columns)
# ---------------------------------------------------------------------------
def get_sig(row, col: str) -> float:
    v = row.get(col, np.nan)
    return float(v) if not pd.isna(v) else np.nan

# ---------------------------------------------------------------------------
# Run verification for one class
# ---------------------------------------------------------------------------
def run_class(class_col: str, verify_fn, signal_cols: list,
              df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    positives = df[df[class_col] == 1].copy()
    n = len(positives)
    print(f"\n  {class_col}: {n:,} BCCC-positive contracts")
    t0 = time.time()

    results = []

    def process_row(row):
        path = resolve_path(row["bccc_file_path"])
        src  = read_source(path)
        sigs = {c: get_sig(row, c) for c in signal_cols}
        verdict, conf, notes = verify_fn(src, sigs)
        return {"id": row["id"], "verdict": verdict, "confidence": conf, "notes": notes,
                "src_len": len(src), "path_ok": len(src) > 0}

    rows_iter = [r for _, r in positives.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_row, r): r["id"] for r in rows_iter}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 2000 == 0:
                print(f"    {done:,}/{n:,}  ({done/n*100:.0f}%)  "
                      f"{(time.time()-t0):.0f}s elapsed")

    out = pd.DataFrame(results)
    out["class"] = class_col
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # summary
    vc = out["verdict"].value_counts()
    for v, c in vc.items():
        print(f"    {v}: {c:,}  ({c/n*100:.1f}%)")
    path_fail = (~out["path_ok"]).sum()
    if path_fail:
        print(f"    ⚠️  {path_fail} files not found on disk")
    return out


# ---------------------------------------------------------------------------
# Run all 6 noisy classes
# ---------------------------------------------------------------------------
all_results = []

CLASS_SPECS = [
    ("Class11:Reentrancy",      verify_reentrancy,
     ["slither_Class11_Reentrancy", "aderyn_Class11_Reentrancy"]),
    ("Class08:CallToUnknown",   verify_callto_unknown,
     ["slither_Class08_CallToUnknown", "aderyn_Class08_CallToUnknown"]),
    ("Class04:Timestamp",       verify_timestamp,
     ["slither_Class04_Timestamp"]),
    ("Class01:ExternalBug",     verify_external_bug,
     ["slither_Class01_ExternalBug", "aderyn_Class01_ExternalBug"]),
    ("Class02:GasException",    verify_gas_exception,
     ["slither_Class02_GasException"]),
    ("Class09:DenialOfService", verify_denial_of_service,
     ["slither_Class09_DenialOfService"]),
]

for class_col, verify_fn, signal_cols in CLASS_SPECS:
    print("\n" + "="*60)
    print(f"CLASS: {class_col}")
    print("="*60)
    res = run_class(class_col, verify_fn, signal_cols, df)
    all_results.append(res)

# ---------------------------------------------------------------------------
# Merge all results and apply gates
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("STAGE 5.2 GATE RESULTS")
print("="*60)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(P5_OUT / "p5_s2_automated_verdict.csv", index=False)
print(f"\nSaved: p5_s2_automated_verdict.csv  ({len(combined):,} rows)")

gate_rows = []
for cls in combined["class"].unique():
    sub = combined[combined["class"] == cls]
    n = len(sub)
    n_keep  = (sub["verdict"] == "KEEP").sum()
    n_drop  = (sub["verdict"] == "DROP").sum()
    n_unc   = (sub["verdict"] == "UNCERTAIN").sum()
    n_no_src= (~sub["path_ok"]).sum()

    # Agreement = (KEEP with high conf + DROP with high conf) / total
    high_conf = (sub["confidence"] >= 0.75).sum()
    pct_agree = high_conf / n * 100

    # Gate thresholds from plan
    if pct_agree >= 95:
        gate = "VERIFIED"
    elif pct_agree >= 80:
        gate = "PROVISIONAL — edge cases to 5.3"
    else:
        gate = "UNVERIFIED → Stage 5.3"

    gate_rows.append({
        "class": cls, "n_total": n,
        "n_keep": int(n_keep), "pct_keep": round(n_keep/n*100,1),
        "n_drop": int(n_drop), "pct_drop": round(n_drop/n*100,1),
        "n_uncertain": int(n_unc), "pct_uncertain": round(n_unc/n*100,1),
        "n_no_src": int(n_no_src),
        "pct_high_conf": round(pct_agree,1),
        "gate": gate
    })

    print(f"\n{cls}  (n={n:,})")
    print(f"  KEEP:      {n_keep:5,}  ({n_keep/n*100:.1f}%)")
    print(f"  DROP:      {n_drop:5,}  ({n_drop/n*100:.1f}%)")
    print(f"  UNCERTAIN: {n_unc:5,}  ({n_unc/n*100:.1f}%)")
    print(f"  No source: {n_no_src:5,}")
    print(f"  High-conf decisions (≥0.75): {high_conf:,} / {n:,} = {pct_agree:.1f}%")
    print(f"  Gate: {gate}")

gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(P5_OUT / "p5_s2_gate_results.csv", index=False)

# Also save per-class dispute sets (for Stage 5.3)
print("\n--- Dispute sets saved for Stage 5.3 ---")
for cls in combined["class"].unique():
    sub = combined[combined["class"] == cls]
    disputed = sub[sub["verdict"] == "UNCERTAIN"]
    if len(disputed) > 0:
        fname = f"p5_s2_disputes_{cls.replace(':','_').lower()}.csv"
        disputed.to_csv(P5_OUT / fname, index=False)
        print(f"  {cls}: {len(disputed):,} disputed → {fname}")

print("\nStage 5.2 complete.")
