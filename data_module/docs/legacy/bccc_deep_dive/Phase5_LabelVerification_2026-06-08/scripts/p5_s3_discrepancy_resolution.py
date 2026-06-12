"""Stage 5.3: Discrepancy Resolution
Refines Stage 5.2 UNCERTAIN verdicts using:
  - NaN vs 0 slither split (NaN = tool not run; 0 = tool ran, didn't fire)
  - Stronger structural patterns for GasException / DenialOfService
  - Context-aware Timestamp analysis
  - Confidence bumping for unambiguous patterns

Run from repo root:
    python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_s3_discrepancy_resolution.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(".")
P4_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs"
P5_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"
P5_EV  = P5_OUT / "p5_s1_evidence_table.csv"

# -------------------------------------------------------------------------
# Load Stage 5.2 verdicts + evidence table (for NaN vs 0 tool signals)
# -------------------------------------------------------------------------
print("Loading Stage 5.2 verdicts...")
s2 = pd.read_csv(P5_OUT / "p5_s2_automated_verdict.csv")
print(f"  {len(s2):,} rows loaded")

# Load evidence table for slither/aderyn actual values (NaN = not run, 0 = ran+miss)
print("Loading evidence table for tool signal quality...")
ev_needed = ["id",
             "slither_Class11_Reentrancy", "slither_Class08_CallToUnknown",
             "slither_Class04_Timestamp",   "slither_Class01_ExternalBug",
             "slither_Class02_GasException","slither_Class09_DenialOfService",
             "aderyn_Class11_Reentrancy",   "aderyn_Class08_CallToUnknown",
             "aderyn_Class01_ExternalBug"]
ev_header = pd.read_csv(P5_EV, nrows=0).columns.tolist()
ev = pd.read_csv(P5_EV, usecols=[c for c in ev_needed if c in ev_header])
s2 = s2.merge(ev, on="id", how="left")

# Also need bccc_file_path for re-reading source on UNCERTAIN contracts
df_base = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv",
                      usecols=["id", "bccc_file_path"])
s2 = s2.merge(df_base, on="id", how="left")
print(f"  Merged: {len(s2):,} rows")

def resolve_path(bcck: str) -> Path:
    fixed = str(bcck).replace("Source Codes/", "SourceCodes/")
    return ROOT / fixed

def read_source(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

# -------------------------------------------------------------------------
# Stage 5.3 helpers
# -------------------------------------------------------------------------

def tool_status(val) -> str:
    """Classify tool column value: 'hit', 'miss', 'not_run'"""
    if pd.isna(val):
        return "not_run"
    return "hit" if int(val) == 1 else "miss"


# ---- Reentrancy refinement ----
RE_CALLVAL_PRE08  = re.compile(r'\.call\.value\s*\([^)]*\)\s*\(', re.S)
RE_CALLVAL_POST08 = re.compile(r'\.call\s*\{[^}]*value\s*:', re.S)
RE_NONREENTRANT   = re.compile(r'nonReentrant|ReentrancyGuard', re.S)

def refine_reentrancy(row) -> tuple:
    """Re-evaluate UNCERTAIN Reentrancy rows."""
    slither_ts = tool_status(row.get("slither_Class11_Reentrancy"))
    src = read_source(resolve_path(row["bccc_file_path"]))
    has_pre  = bool(RE_CALLVAL_PRE08.search(src))
    has_post = bool(RE_CALLVAL_POST08.search(src))
    has_guard = bool(RE_NONREENTRANT.search(src))

    if not (has_pre or has_post):
        # Stage 5.2 assigned UNCERTAIN; but on fresh read still no pattern
        return "DROP", 0.82, "s3:recheck_no_callvalue"
    if has_guard:
        return "DROP", 0.76, "s3:guard_present"
    if slither_ts == "not_run":
        # Pattern present, tool not run → trust the pattern
        return "KEEP", 0.72, "s3:callvalue_slither_not_run"
    if slither_ts == "miss":
        # Tool ran but didn't fire → slither has low recall; still likely vuln
        return "KEEP", 0.62, "s3:callvalue_slither_miss"
    return "KEEP", 0.80, "s3:callvalue_slither_hit"


# ---- CallToUnknown refinement ----
RE_LOWLEVEL    = re.compile(r'\.call\s*[\(\{]|\.delegatecall\s*\(|\.staticcall\s*\(', re.S)
RE_USER_ARG    = re.compile(
    r'(?:msg\.sender|_addr|_to|_target|recipient|user|owner|caller)\s*\.'
    r'(?:call|delegatecall)',
    re.S | re.I)
RE_PARAM_CALL  = re.compile(
    r'(?:address|addr)\s+\w*[Aa]ddr\w*[^;]{0,100}\.call\s*[\(\{]'
    r'|function\s+\w+\s*\([^)]*address[^)]*\)[^{]{0,200}\.call\s*[\(\{]',
    re.S)
RE_HARDCODED   = re.compile(r'0x[0-9a-fA-F]{40}\.call', re.S)

def refine_callto_unknown(row) -> tuple:
    slither_ts = tool_status(row.get("slither_Class08_CallToUnknown"))
    aderyn_ts  = tool_status(row.get("aderyn_Class08_CallToUnknown"))
    src = read_source(resolve_path(row["bccc_file_path"]))
    has_ll      = bool(RE_LOWLEVEL.search(src))
    has_user    = bool(RE_USER_ARG.search(src))
    has_param   = bool(RE_PARAM_CALL.search(src))
    has_hardcode= bool(RE_HARDCODED.search(src))

    if not has_ll:
        return "DROP", 0.82, "s3:no_lowlevel"
    if has_hardcode and not has_user and not has_param:
        return "DROP", 0.68, "s3:only_hardcoded_call"
    if has_user or has_param:
        tool_hit = (slither_ts == "hit" or aderyn_ts == "hit")
        if tool_hit:
            return "KEEP", 0.80, "s3:user_ctrl_addr+tool"
        not_run = (slither_ts == "not_run" and aderyn_ts == "not_run")
        if not_run:
            return "KEEP", 0.68, "s3:user_ctrl_addr_tools_not_run"
        return "KEEP", 0.60, "s3:user_ctrl_addr_tool_miss"
    # has_ll but no clear user-controlled indicator
    if slither_ts == "hit" or aderyn_ts == "hit":
        return "KEEP", 0.75, "s3:tool_confirms_ctu"
    return "UNCERTAIN", 0.42, "s3:lowlevel_target_unclear"


# ---- Timestamp refinement ----
RE_TS_ANY    = re.compile(r'\bblock\.timestamp\b|\bnow\b', re.S)
RE_TS_BRANCH = re.compile(
    r'(?:if|require|assert)\s*\([^;]{0,120}(?:block\.timestamp|now\b)', re.S | re.I)
RE_TS_MODULO = re.compile(r'(?:block\.timestamp|now\b)\s*%', re.S)
RE_TS_KECCAK = re.compile(r'keccak256\s*\([^)]{0,120}(?:block\.timestamp|now\b)', re.S)
RE_TS_SEED   = re.compile(
    r'(?:block\.timestamp|now\b)[^;]{0,60}(?:random|rand|seed|lottery|winner)',
    re.S | re.I)
RE_TS_EMIT_ONLY = re.compile(
    r'emit\s+\w+\s*\([^;]{0,200}(?:block\.timestamp|now\b)', re.S)
RE_TS_DEADLINE  = re.compile(
    r'(?:deadline|expiry|expiration|end[Tt]ime|startTime|lockTime)[^;]{0,60}'
    r'(?:block\.timestamp|now\b)'
    r'|(?:block\.timestamp|now\b)[^;]{0,60}(?:deadline|expiry|expiration|end[Tt]ime)',
    re.S)
RE_TS_REQUIRE_TS = re.compile(
    r'require\s*\([^;]{0,80}(?:block\.timestamp|now\b)[^;]{0,80}(?:deadline|expiry|end|start)',
    re.S | re.I)

def refine_timestamp(row) -> tuple:
    slither_ts = tool_status(row.get("slither_Class04_Timestamp"))
    src = read_source(resolve_path(row["bccc_file_path"]))
    has_ts     = bool(RE_TS_ANY.search(src))
    in_branch  = bool(RE_TS_BRANCH.search(src))
    in_modulo  = bool(RE_TS_MODULO.search(src))
    in_keccak  = bool(RE_TS_KECCAK.search(src))
    in_seed    = bool(RE_TS_SEED.search(src))
    emit_only  = bool(RE_TS_EMIT_ONLY.search(src))
    in_deadline= bool(RE_TS_DEADLINE.search(src))
    req_ts     = bool(RE_TS_REQUIRE_TS.search(src))
    critical   = in_branch or in_modulo or in_keccak or in_seed

    if not has_ts:
        return "DROP", 0.90, "s3:no_timestamp"
    # Strong gambling/randomness use = definitely vulnerable
    if in_seed or in_keccak:
        return "KEEP", 0.88, "s3:ts_in_randomness"
    # Deadline requires in financial logic — clear inclusion
    if in_deadline and req_ts:
        return "KEEP", 0.82, "s3:ts_financial_deadline"
    # General branch with timestamp + slither confirms
    if in_branch and slither_ts == "hit":
        return "KEEP", 0.85, "s3:ts_branch+slither"
    if in_branch and slither_ts == "not_run":
        return "KEEP", 0.72, "s3:ts_branch_slither_not_run"
    if in_branch and slither_ts == "miss":
        return "KEEP", 0.60, "s3:ts_branch_slither_miss"
    # emit-only → drop
    if emit_only and not critical:
        return "DROP", 0.72, "s3:ts_emit_only"
    # Deadline-stored but not in critical branch
    if in_deadline and not critical:
        return "UNCERTAIN", 0.48, "s3:ts_deadline_not_branch"
    return "UNCERTAIN", 0.38, "s3:ts_usage_non_critical"


# ---- ExternalBug refinement ----
RE_SELFDESTRUCT = re.compile(r'\bselfdestruct\s*\(|\bsuicide\s*\(', re.S)
RE_TX_ORIGIN    = re.compile(r'\btx\.origin\b', re.S)
RE_DELEGATECALL = re.compile(r'\.delegatecall\s*\(', re.S)
RE_ECRECOVER    = re.compile(r'\becrecover\s*\(', re.S)
RE_ONLY_OWNER   = re.compile(r'onlyOwner|require\s*\([^;]{0,60}msg\.sender\s*==\s*owner', re.S | re.I)
RE_TX_AUTH      = re.compile(r'require\s*\([^;]{0,80}tx\.origin', re.S)

def refine_external_bug(row) -> tuple:
    slither_ts = tool_status(row.get("slither_Class01_ExternalBug"))
    aderyn_ts  = tool_status(row.get("aderyn_Class01_ExternalBug"))
    src = read_source(resolve_path(row["bccc_file_path"]))
    has_sd  = bool(RE_SELFDESTRUCT.search(src))
    has_txo = bool(RE_TX_ORIGIN.search(src))
    has_dc  = bool(RE_DELEGATECALL.search(src))
    has_ecr = bool(RE_ECRECOVER.search(src))
    # tx.origin used in require() → auth vulnerability
    txo_in_auth = has_txo and bool(RE_TX_AUTH.search(src))
    # selfdestruct without owner-only guard
    sd_unguarded = has_sd and not bool(RE_ONLY_OWNER.search(src))

    tool_hit = (slither_ts == "hit" or aderyn_ts == "hit")
    tool_not_run = (slither_ts == "not_run" and aderyn_ts == "not_run")

    if txo_in_auth:
        conf = 0.85 if tool_hit else (0.72 if tool_not_run else 0.62)
        return "KEEP", conf, "s3:tx_origin_in_auth"
    if sd_unguarded:
        conf = 0.80 if tool_hit else (0.68 if tool_not_run else 0.60)
        return "KEEP", conf, "s3:selfdestruct_unguarded"
    if has_dc:
        conf = 0.78 if tool_hit else (0.65 if tool_not_run else 0.55)
        return "KEEP", conf, "s3:delegatecall_present"
    if has_ecr:
        conf = 0.72 if tool_hit else 0.58
        return "KEEP", conf, "s3:ecrecover_present"
    if tool_hit:
        return "KEEP", 0.75, "s3:tool_confirms_extbug"
    return "UNCERTAIN", 0.42, "s3:pattern_unclear"


# ---- GasException refinement ----
RE_LOOP         = re.compile(r'\bfor\s*\(|\bwhile\s*\(', re.S)
RE_STORAGE_ITER = re.compile(
    r'\.length\b[^;]{0,60}(?:for|while)'
    r'|(?:for|while)[^{]{0,80}\.length\b',
    re.S)
RE_MAP_ITER     = re.compile(r'mapping\s*\(', re.S)
RE_PUSH         = re.compile(r'\.push\s*\(', re.S)
RE_SSTORE_LOOP  = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,400}(?:storage\b|=\s*\w+\[)',
    re.S)
RE_BOUNDED_LOOP = re.compile(r'\bfor\s*\([^;]*;\s*\w+\s*<\s*(\d+)\s*;', re.S)
RE_GAS_LIMIT    = re.compile(r'\.call\.gas\s*\(|\.gas\s*\(', re.S)

def refine_gas_exception(row) -> tuple:
    slither_ge = tool_status(row.get("slither_Class02_GasException"))
    src = read_source(resolve_path(row["bccc_file_path"]))
    has_loop     = bool(RE_LOOP.search(src))
    has_store_it = bool(RE_STORAGE_ITER.search(src))
    has_push     = bool(RE_PUSH.search(src))
    has_sstore_l = bool(RE_SSTORE_LOOP.search(src))
    bounded      = RE_BOUNDED_LOOP.findall(src)
    all_bounded  = bool(bounded) and not has_push and not has_store_it
    has_gas_limit= bool(RE_GAS_LIMIT.search(src))

    if not has_loop:
        return "DROP", 0.82, "s3:no_loop"
    if all_bounded and not has_sstore_l:
        return "DROP", 0.70, "s3:bounded_no_storage_write"
    # Loop over storage array (array.length) — unbounded in practice
    if has_store_it or has_sstore_l:
        conf = 0.82 if slither_ge == "hit" else (0.70 if slither_ge == "not_run" else 0.60)
        return "KEEP", conf, "s3:loop_over_storage_array"
    if has_push and has_loop:
        conf = 0.75 if slither_ge == "hit" else 0.58
        return "KEEP", conf, "s3:push_into_iterated_array"
    if has_gas_limit:
        return "UNCERTAIN", 0.45, "s3:gas_limit_set_manually"
    if slither_ge == "hit":
        return "KEEP", 0.76, "s3:slither_costly_loop"
    return "UNCERTAIN", 0.40, "s3:loop_pattern_ambiguous"


# ---- DenialOfService refinement ----
RE_LOOP_SEND  = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,600}\.(?:send|transfer|call)\s*[\(\{]'
    r'|\.(?:send|transfer|call)\s*[\(\{][^}]{0,600}(?:for|while)\s*\(',
    re.S)
RE_REQUIRE_SEND = re.compile(
    r'require\s*\([^;]{0,80}\.(?:send|transfer)\s*\(', re.S)
RE_LOOP_PUSH    = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,400}\.push\s*\(', re.S)
RE_ADDR_ARRAY   = re.compile(
    r'address\s*(?:payable\s*)?\[\s*\]\s*(?:public\s*|private\s*|internal\s*)?\w+\s*;',
    re.S)
RE_LOOP_MAP     = re.compile(
    r'(?:for|while)[^{]{0,80}\{[^}]{0,400}mapping\s*\(', re.S)
RE_KING_PATTERN = re.compile(
    r'require\s*\([^;]{0,80}\.(?:send|transfer)\s*\([^)]*\)\s*\)',
    re.S)
RE_LOOP_EXTERNAL= re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,600}\.call\s*[\(\{]', re.S)

def refine_denial_of_service(row) -> tuple:
    slither_dos = tool_status(row.get("slither_Class09_DenialOfService"))
    src = read_source(resolve_path(row["bccc_file_path"]))
    has_loop       = bool(RE_LOOP.search(src))
    loop_send      = bool(RE_LOOP_SEND.search(src))
    req_send       = bool(RE_REQUIRE_SEND.search(src))
    loop_push      = bool(RE_LOOP_PUSH.search(src))
    addr_arr       = bool(RE_ADDR_ARRAY.search(src))
    king_pat       = bool(RE_KING_PATTERN.search(src))
    loop_external  = bool(RE_LOOP_EXTERNAL.search(src))

    # "King of the hill" pattern: require(msg.sender.transfer(...))
    if king_pat:
        conf = 0.88 if slither_dos == "hit" else (0.78 if slither_dos == "not_run" else 0.68)
        return "KEEP", conf, "s3:king_pattern"
    # Iterating over address array and sending to each
    if (loop_send or loop_external) and addr_arr:
        conf = 0.82 if slither_dos == "hit" else (0.72 if slither_dos == "not_run" else 0.62)
        return "KEEP", conf, "s3:loop_send_over_addr_array"
    # loop with require on external send (blocking if any fails)
    if req_send and has_loop:
        conf = 0.78 if slither_dos == "hit" else (0.65 if slither_dos == "not_run" else 0.55)
        return "KEEP", conf, "s3:require_send_in_loop"
    # Growing address array + loop (push pattern)
    if loop_push and addr_arr:
        conf = 0.72 if slither_dos == "hit" else 0.58
        return "KEEP", conf, "s3:push_to_addr_array"
    if slither_dos == "hit":
        return "KEEP", 0.75, "s3:slither_calls_loop"
    if not has_loop and not req_send:
        return "DROP", 0.72, "s3:no_loop_no_require_send"
    return "UNCERTAIN", 0.40, "s3:partial_dos_ambiguous"


# -------------------------------------------------------------------------
# Apply refinements
# -------------------------------------------------------------------------
REFINE_MAP = {
    "Class11:Reentrancy":      ("slither_Class11_Reentrancy",   refine_reentrancy),
    "Class08:CallToUnknown":   ("slither_Class08_CallToUnknown", refine_callto_unknown),
    "Class04:Timestamp":       ("slither_Class04_Timestamp",     refine_timestamp),
    "Class01:ExternalBug":     ("slither_Class01_ExternalBug",   refine_external_bug),
    "Class02:GasException":    ("slither_Class02_GasException",  refine_gas_exception),
    "Class09:DenialOfService": ("slither_Class09_DenialOfService", refine_denial_of_service),
}

all_refined = []

for cls, (_, refine_fn) in REFINE_MAP.items():
    cls_data = s2[s2["class"] == cls].copy()
    uncertain = cls_data[cls_data["verdict"] == "UNCERTAIN"].copy()
    confirmed = cls_data[cls_data["verdict"] != "UNCERTAIN"].copy()
    print(f"\n{cls}: {len(uncertain):,} UNCERTAIN to refine (of {len(cls_data):,} total)")

    refined_rows = []
    for _, row in uncertain.iterrows():
        v, c, n = refine_fn(row)
        refined_rows.append({
            "id": row["id"], "verdict": v, "confidence": c,
            "notes": n, "class": cls,
            "src_len": row.get("src_len", 0), "path_ok": row.get("path_ok", True)
        })
    refined_df = pd.DataFrame(refined_rows)

    # Keep only the columns needed for stacking
    keep_cols = ["id", "verdict", "confidence", "notes", "class", "src_len", "path_ok"]
    confirmed_slim = confirmed[keep_cols].copy()
    refined_slim   = refined_df[keep_cols].copy()
    cls_final = pd.concat([confirmed_slim, refined_slim], ignore_index=True)

    # Print summary
    vc = cls_final["verdict"].value_counts()
    n = len(cls_final)
    print(f"  After Stage 5.3:")
    for verd, cnt in vc.items():
        print(f"    {verd}: {cnt:,}  ({cnt/n*100:.1f}%)")
    hc = (cls_final["confidence"] >= 0.75).sum()
    print(f"  High-conf (≥0.75): {hc:,} / {n:,} = {hc/n*100:.1f}%")

    all_refined.append(cls_final)

# -------------------------------------------------------------------------
# Save + gate results
# -------------------------------------------------------------------------
combined = pd.concat(all_refined, ignore_index=True)
combined.to_csv(P5_OUT / "p5_s3_refined_verdict.csv", index=False)
print(f"\nSaved p5_s3_refined_verdict.csv  ({len(combined):,} rows)")

print("\n" + "="*65)
print("STAGE 5.3 GATE RESULTS")
print("="*65)

gate_rows = []
for cls in combined["class"].unique():
    sub = combined[combined["class"] == cls]
    n        = len(sub)
    n_keep   = (sub["verdict"] == "KEEP").sum()
    n_drop   = (sub["verdict"] == "DROP").sum()
    n_unc    = (sub["verdict"] == "UNCERTAIN").sum()
    hc       = (sub["confidence"] >= 0.75).sum()
    pct_hc   = hc / n * 100

    if pct_hc >= 95:
        gate = "VERIFIED"
    elif pct_hc >= 80:
        gate = "PROVISIONAL — residual UNCERTAIN to Stage 5.4"
    else:
        gate = "UNVERIFIED → Stage 5.4"

    gate_rows.append({
        "class": cls, "n_total": n,
        "n_keep": int(n_keep), "pct_keep": round(n_keep/n*100, 1),
        "n_drop": int(n_drop), "pct_drop": round(n_drop/n*100, 1),
        "n_uncertain": int(n_unc), "pct_uncertain": round(n_unc/n*100, 1),
        "pct_high_conf": round(pct_hc, 1), "gate": gate,
    })

    print(f"\n{cls}  (n={n:,})")
    print(f"  KEEP:      {n_keep:5,}  ({n_keep/n*100:.1f}%)")
    print(f"  DROP:      {n_drop:5,}  ({n_drop/n*100:.1f}%)")
    print(f"  UNCERTAIN: {n_unc:5,}  ({n_unc/n*100:.1f}%)")
    print(f"  High-conf (≥0.75): {hc:,}/{n:,} = {pct_hc:.1f}%  → {gate}")

gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(P5_OUT / "p5_s3_gate_results.csv", index=False)

# Save remaining UNCERTAIN per class for Stage 5.4 manual review
print("\n--- Residual UNCERTAIN sets saved for Stage 5.4 ---")
for cls in combined["class"].unique():
    sub = combined[(combined["class"] == cls) & (combined["verdict"] == "UNCERTAIN")]
    if len(sub) > 0:
        fname = f"p5_s3_residual_{cls.replace(':','_').lower()}.csv"
        sub.to_csv(P5_OUT / fname, index=False)
        print(f"  {cls}: {len(sub):,} residual → {fname}")

print("\nStage 5.3 complete.")
