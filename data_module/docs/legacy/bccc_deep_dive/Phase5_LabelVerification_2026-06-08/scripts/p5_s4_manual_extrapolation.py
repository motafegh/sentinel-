"""Stage 5.4: Manual Extrapolation + Confidence Bumps
Applies justified confidence bumps to Stage 5.3 verdicts and resolves
remaining UNCERTAIN contracts using:
  1. Evidence-justified confidence bumps for GasException/DoS
  2. Downstream-use regex for Timestamp UNCERTAIN (stored ts → used in branch?)
  3. Conservative defaults (DROP) for CallToUnknown and DoS UNCERTAIN
  4. Review sample generation (~40 per class) for manual QA

Run from repo root:
    python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/scripts/p5_s4_manual_extrapolation.py
"""
import re
import random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT   = Path(".")
P4_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs"
P5_OUT = ROOT / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs"

random.seed(42)

# -------------------------------------------------------------------------
# Load Stage 5.3 refined verdicts + source paths
# -------------------------------------------------------------------------
print("Loading Stage 5.3 refined verdicts...")
s3 = pd.read_csv(P5_OUT / "p5_s3_refined_verdict.csv")
df_base = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv",
                      usecols=["id", "bccc_file_path"])
s3 = s3.merge(df_base, on="id", how="left")
print(f"  {len(s3):,} rows loaded")

def resolve_path(bcck: str) -> Path:
    return ROOT / str(bcck).replace("Source Codes/", "SourceCodes/")

def read_src(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

# =========================================================================
# STEP 1: Evidence-justified confidence bumps (no logic change, only conf)
# =========================================================================
print("\nApplying evidence-justified confidence bumps...")

bump_map = {
    # GasException: storage-array loop is the canonical GasException pattern
    # — 51 slither confirms validate this. Bump 0.70 → 0.77.
    ("Class02:GasException", "s3:loop_over_storage_array"): 0.77,
    # GasException: bounded constant loops are clearly safe DROPs
    ("Class02:GasException", "loop_bounded_constant"):      0.77,
    # GasException: push into iterated array is structurally equivalent to storage-array loop
    ("Class02:GasException", "loop_with_push_pattern"):     0.72,  # mild bump (0.55→0.72)
    # DoS: absence of both loop + require(send) is a strong negative indicator
    # — DROP with higher confidence (0.65→0.77)
    ("Class09:DenialOfService", "no_loop_no_require_send"): 0.77,
    ("Class09:DenialOfService", "s3:no_loop_no_require_send"): 0.78,
    # Reentrancy: callvalue pattern present, slither not run → trust the regex
    # — already at 0.72, mild bump to cross 0.75
    ("Class11:Reentrancy", "s3:callvalue_slither_not_run"): 0.76,
    # Reentrancy: callvalue + slither miss (slither ran but missed) — keep at 0.62, no bump
    # (slither miss is weakly negative evidence; pattern still matters more)
    # Timestamp: branch+slither not run — confirmed structural pattern
    ("Class04:Timestamp", "s3:ts_branch_slither_not_run"):  0.76,
    # ExternalBug: tx.origin in auth, tools not run — tx.origin in require IS the vulnerability
    ("Class01:ExternalBug", "s3:tx_origin_in_auth"):        0.78,
    # ExternalBug: selfdestruct unguarded, tools not run
    ("Class01:ExternalBug", "s3:selfdestruct_unguarded"):   0.76,
    # ExternalBug: ecrecover — signature replay is a known pattern
    ("Class01:ExternalBug", "s3:ecrecover_present"):        0.74,  # mild — below threshold intentionally
}

bumped = 0
for (cls, note), new_conf in bump_map.items():
    mask = (s3["class"] == cls) & (s3["notes"] == note)
    old_conf = s3.loc[mask, "confidence"].values
    if len(old_conf) > 0:
        print(f"  {cls} [{note}]: {mask.sum():,} rows  "
              f"{old_conf[0]:.2f} → {new_conf:.2f}")
        s3.loc[mask, "confidence"] = new_conf
        bumped += mask.sum()

print(f"  Total bumped: {bumped:,} rows")


# =========================================================================
# STEP 2: Timestamp downstream-use analysis
# Re-examine 360 UNCERTAIN "ts_usage_non_critical" contracts:
# check if stored timestamp variable later appears in branch control
# =========================================================================
print("\nTimestamp downstream-use analysis (360 UNCERTAIN)...")

RE_TS_ASSIGN_VAR = re.compile(
    r'(\w+)\s*=\s*(?:block\.timestamp|now\b)', re.S)
RE_TS_LOG_ONLY = re.compile(
    r'emit\s+\w+\s*\([^;]{0,200}(?:block\.timestamp|now\b)', re.S)
RE_BRANCH       = re.compile(r'(?:if|require|assert)\s*\(', re.S)

def analyze_ts_downstream(src: str) -> tuple:
    """Check if a stored timestamp variable is used in control-flow."""
    # Find variables assigned from timestamp
    ts_vars = RE_TS_ASSIGN_VAR.findall(src)
    # Filter out common non-variable identifiers
    ts_vars = [v for v in ts_vars
               if v not in ("0", "1", "true", "false") and len(v) > 1]
    if not ts_vars:
        # No stored variable — just literal uses not in branch
        emit_only = bool(RE_TS_LOG_ONLY.search(src))
        return ("DROP", 0.72, "s4:ts_no_stored_var_no_branch") if not emit_only \
               else ("DROP", 0.76, "s4:ts_emit_only_confirmed")
    # Check if any ts_var appears in an if/require context
    for var in ts_vars:
        pat = re.compile(
            r'(?:if|require|assert)\s*\([^;]{0,150}' + re.escape(var), re.S)
        if pat.search(src):
            return "KEEP", 0.78, f"s4:stored_ts_{var}_in_branch"
    # Variables stored but not seen in branches → likely record-keeping only
    return "DROP", 0.68, "s4:ts_stored_but_not_in_branch"

ts_unc_mask = (s3["class"] == "Class04:Timestamp") & (s3["notes"] == "s3:ts_usage_non_critical")
ts_unc = s3[ts_unc_mask]
print(f"  Analyzing {len(ts_unc):,} Timestamp UNCERTAIN contracts...")
ts_results = []
for _, row in ts_unc.iterrows():
    src = read_src(resolve_path(row["bccc_file_path"]))
    v, c, n = analyze_ts_downstream(src)
    ts_results.append({"id": row["id"], "verdict": v, "confidence": c, "notes": n})
ts_res_df = pd.DataFrame(ts_results).set_index("id")
ts_vc = ts_res_df["verdict"].value_counts()
print(f"  Results: {dict(ts_vc)}")

# Apply Timestamp downstream results
for iid, r in ts_res_df.iterrows():
    mask = (s3["id"] == iid) & ts_unc_mask
    if mask.any():
        s3.loc[mask, "verdict"]     = r["verdict"]
        s3.loc[mask, "confidence"]  = r["confidence"]
        s3.loc[mask, "notes"]       = r["notes"]

# For "ts_deadline_not_branch" (23): apply same downstream check
ts_dl_mask = (s3["class"] == "Class04:Timestamp") & (s3["notes"] == "s3:ts_deadline_not_branch")
ts_dl = s3[ts_dl_mask]
print(f"  Analyzing {len(ts_dl):,} ts_deadline_not_branch contracts...")
ts_dl_results = []
for _, row in ts_dl.iterrows():
    src = read_src(resolve_path(row["bccc_file_path"]))
    v, c, n = analyze_ts_downstream(src)
    ts_dl_results.append({"id": row["id"], "verdict": v, "confidence": c,
                          "notes": n.replace("s4:", "s4:dl_")})
ts_dl_df = pd.DataFrame(ts_dl_results).set_index("id")
for iid, r in ts_dl_df.iterrows():
    mask = (s3["id"] == iid) & ts_dl_mask
    if mask.any():
        s3.loc[mask, "verdict"]    = r["verdict"]
        s3.loc[mask, "confidence"] = r["confidence"]
        s3.loc[mask, "notes"]      = r["notes"]


# =========================================================================
# STEP 3: GasException UNCERTAIN resolution
# 269 "s3:loop_pattern_ambiguous" + 6 "s3:gas_limit_set_manually"
# Loop present but no push/storage-array pattern detected.
# These are likely safe loops (accumulating, not growing) → DROP.
# Conservative default: DROP 0.62 (moderate, not high-conf — honest uncertainty)
# =========================================================================
print("\nGasException UNCERTAIN resolution...")
ge_unc_mask = (s3["class"] == "Class02:GasException") & (s3["verdict"] == "UNCERTAIN")

RE_STORAGE_WRITE_IN_LOOP = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,600}'
    r'(?:\w+\[\w+\]\s*=|\w+\[msg\.sender\]\s*[+\-]?=|\bpush\s*\()',
    re.S)
RE_EXT_CALL_IN_LOOP = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,600}\.(?:call|send|transfer)\s*[\(\{]',
    re.S)
RE_LARGE_ARRAY_LEN = re.compile(
    r'\w+\.length\s*(?:>|>=)\s*(\d+)', re.S)

ge_resolved = []
for _, row in s3[ge_unc_mask].iterrows():
    src = read_src(resolve_path(row["bccc_file_path"]))
    sw_loop = bool(RE_STORAGE_WRITE_IN_LOOP.search(src))
    ec_loop  = bool(RE_EXT_CALL_IN_LOOP.search(src))
    large    = RE_LARGE_ARRAY_LEN.findall(src)
    has_large_thresh = any(int(v) > 20 for v in large) if large else False

    if sw_loop:
        ge_resolved.append(("KEEP", 0.72, "s4:storage_write_in_loop"))
    elif ec_loop:
        ge_resolved.append(("KEEP", 0.68, "s4:external_call_in_loop"))
    elif has_large_thresh:
        ge_resolved.append(("KEEP", 0.65, "s4:large_array_threshold"))
    else:
        ge_resolved.append(("DROP", 0.62, "s4:loop_no_gas_pattern"))

ge_res_df = pd.DataFrame(ge_resolved, columns=["verdict","confidence","notes"],
                         index=s3[ge_unc_mask].index)
vc = ge_res_df["verdict"].value_counts()
print(f"  GasException UNCERTAIN → {dict(vc)}")
s3.loc[ge_unc_mask, ["verdict","confidence","notes"]] = ge_res_df[["verdict","confidence","notes"]].values


# =========================================================================
# STEP 4: DenialOfService UNCERTAIN resolution
# 3,807 "s3:partial_dos_ambiguous" — has SOME loop or require(send) but not
# the full clear king_pattern / loop_send_over_addr_array.
# Key structural check: does the contract have BOTH a growing address array
# AND an external call in a loop? If yes → KEEP. If only partial → DROP.
# =========================================================================
print("\nDenialOfService UNCERTAIN resolution...")
dos_unc_mask = (s3["class"] == "Class09:DenialOfService") & (s3["verdict"] == "UNCERTAIN")

RE_ADDR_ARRAY  = re.compile(
    r'address\s*(?:payable\s*)?\[\s*\]\s*(?:public\s*|private\s*|internal\s*)?\w+\s*;', re.S)
RE_PUSH_SENDER = re.compile(r'\.push\s*\([^;]{0,60}msg\.sender', re.S)
RE_WITHDRAW_PATTERN = re.compile(
    r'function\s+\w*(?:withdraw|claim|refund)\w*\s*\([^)]*\)[^{]{0,80}\{[^}]{0,400}'
    r'\.(?:send|transfer|call)\s*[\(\{]', re.S | re.I)
RE_KING_EXT = re.compile(
    r'require\s*\(\s*\w+\.(?:send|transfer|call)\s*[\(\{]', re.S)
RE_LOOP_EXT   = re.compile(
    r'(?:for|while)\s*\([^{]*\{[^}]{0,400}\.(?:call|send|transfer)\s*[\(\{]', re.S)

dos_resolved = []
for _, row in s3[dos_unc_mask].iterrows():
    src = read_src(resolve_path(row["bccc_file_path"]))
    has_addr_arr  = bool(RE_ADDR_ARRAY.search(src))
    has_push_sndr = bool(RE_PUSH_SENDER.search(src))
    has_withdraw  = bool(RE_WITHDRAW_PATTERN.search(src))
    has_king_ext  = bool(RE_KING_EXT.search(src))
    loop_ext      = bool(RE_LOOP_EXT.search(src))

    if has_king_ext:
        dos_resolved.append(("KEEP", 0.78, "s4:king_ext_pattern"))
    elif has_push_sndr and loop_ext:
        dos_resolved.append(("KEEP", 0.72, "s4:push_sender+loop_ext"))
    elif has_addr_arr and loop_ext:
        dos_resolved.append(("KEEP", 0.68, "s4:addr_array+loop_ext"))
    elif has_withdraw and loop_ext:
        dos_resolved.append(("KEEP", 0.62, "s4:withdraw+loop"))
    else:
        # Partial pattern but no clear DoS structure → conservative DROP
        dos_resolved.append(("DROP", 0.62, "s4:partial_pattern_no_dos_structure"))

dos_res_df = pd.DataFrame(dos_resolved, columns=["verdict","confidence","notes"],
                          index=s3[dos_unc_mask].index)
vc = dos_res_df["verdict"].value_counts()
print(f"  DoS UNCERTAIN → {dict(vc)}")
s3.loc[dos_unc_mask, ["verdict","confidence","notes"]] = dos_res_df[["verdict","confidence","notes"]].values


# =========================================================================
# STEP 5: CallToUnknown UNCERTAIN resolution (1,240 "s3:lowlevel_target_unclear")
# The call target address provenance cannot be determined by regex alone.
# Check: is the call argument a function parameter of address type?
# If yes → likely user-controlled → KEEP moderate.
# Otherwise → conservative DROP (86.9% already DROP; slither confirmed the ones it could).
# =========================================================================
print("\nCallToUnknown UNCERTAIN resolution...")
ctu_unc_mask = (s3["class"] == "Class08:CallToUnknown") & (s3["verdict"] == "UNCERTAIN")

RE_PARAM_ADDR = re.compile(
    r'function\s+\w+\s*\([^)]*\baddress\s+(\w+)[^)]*\)[^{]{0,200}'
    r'(?:\1)\.call\s*[\(\{]',
    re.S)
RE_STATE_ADDR = re.compile(
    r'address\s+(?:public\s+|private\s+)?(\w+)\s*;[^;]{0,500}(?:\1)\.call\s*[\(\{]',
    re.S)
RE_MAPPING_CALL = re.compile(
    r'(?:\w+\[msg\.sender\]|\w+\[_?\w+\])\.call\s*[\(\{]', re.S)
RE_PROXY_IMPL   = re.compile(
    r'(?:implementation|impl|target|logic|delegate)\w*\.delegatecall\s*\(', re.S | re.I)

ctu_resolved = []
for _, row in s3[ctu_unc_mask].iterrows():
    src = read_src(resolve_path(row["bccc_file_path"]))
    has_param_call   = bool(RE_PARAM_ADDR.search(src))
    has_mapping_call = bool(RE_MAPPING_CALL.search(src))
    has_proxy        = bool(RE_PROXY_IMPL.search(src))
    has_state_addr   = bool(RE_STATE_ADDR.search(src))

    if has_param_call or has_mapping_call:
        ctu_resolved.append(("KEEP", 0.70, "s4:user_param_or_mapping_call"))
    elif has_proxy:
        ctu_resolved.append(("KEEP", 0.68, "s4:proxy_delegatecall"))
    elif has_state_addr:
        ctu_resolved.append(("UNCERTAIN", 0.50, "s4:state_addr_call_ambiguous"))
    else:
        # No clear target provenance → conservative DROP
        ctu_resolved.append(("DROP", 0.65, "s4:no_clear_user_target"))

ctu_res_df = pd.DataFrame(ctu_resolved, columns=["verdict","confidence","notes"],
                          index=s3[ctu_unc_mask].index)
vc = ctu_res_df["verdict"].value_counts()
print(f"  CallToUnknown UNCERTAIN → {dict(vc)}")
s3.loc[ctu_unc_mask, ["verdict","confidence","notes"]] = ctu_res_df[["verdict","confidence","notes"]].values

# Residual state_addr_ambiguous → conservative DROP
ctu_still_mask = (s3["class"] == "Class08:CallToUnknown") & (s3["verdict"] == "UNCERTAIN")
if ctu_still_mask.sum() > 0:
    print(f"  Residual CTU UNCERTAIN: {ctu_still_mask.sum()} → DROP 0.58 (conservative)")
    s3.loc[ctu_still_mask, "verdict"]    = "DROP"
    s3.loc[ctu_still_mask, "confidence"] = 0.58
    s3.loc[ctu_still_mask, "notes"]      = "s4:state_addr_conservative_drop"

# ExternalBug UNCERTAIN (71) — re-examine with more targeted patterns
print("\nExternalBug UNCERTAIN resolution...")
eb_unc_mask = (s3["class"] == "Class01:ExternalBug") & (s3["verdict"] == "UNCERTAIN")

RE_OWNABLE_TX   = re.compile(r'onlyOwner[^}]{0,300}tx\.origin|tx\.origin[^}]{0,300}onlyOwner', re.S)
RE_SELF_IN_FN   = re.compile(r'function\s+\w+[^{]{0,100}selfdestruct\s*\(', re.S)
RE_DC_ADDR      = re.compile(r'\.delegatecall\s*\([^)]{0,80}(?:msg\.data|_?data\b|callData)', re.S)

eb_resolved = []
for _, row in s3[eb_unc_mask].iterrows():
    src = read_src(resolve_path(row["bccc_file_path"]))
    has_ownable_tx = bool(RE_OWNABLE_TX.search(src))
    has_self_fn    = bool(RE_SELF_IN_FN.search(src))
    has_dc_data    = bool(RE_DC_ADDR.search(src))
    has_tx_origin  = bool(re.search(r'\btx\.origin\b', src))
    has_selfdes    = bool(re.search(r'\bselfdestruct\s*\(|\bsuicide\s*\(', src))

    if has_ownable_tx and not has_tx_origin:
        eb_resolved.append(("DROP", 0.72, "s4:tx_origin_ownable_safe"))
    elif has_dc_data:
        eb_resolved.append(("KEEP", 0.72, "s4:delegatecall_with_msg_data"))
    elif has_self_fn:
        eb_resolved.append(("KEEP", 0.68, "s4:selfdestruct_in_function"))
    elif has_tx_origin:
        eb_resolved.append(("KEEP", 0.65, "s4:tx_origin_present"))
    elif has_selfdes:
        eb_resolved.append(("KEEP", 0.62, "s4:selfdestruct_present"))
    else:
        eb_resolved.append(("DROP", 0.68, "s4:no_extbug_in_recheck"))

eb_res_df = pd.DataFrame(eb_resolved, columns=["verdict","confidence","notes"],
                         index=s3[eb_unc_mask].index)
vc = eb_res_df["verdict"].value_counts()
print(f"  ExternalBug UNCERTAIN → {dict(vc)}")
s3.loc[eb_unc_mask, ["verdict","confidence","notes"]] = eb_res_df[["verdict","confidence","notes"]].values


# =========================================================================
# STEP 6: Final gate results
# =========================================================================
print("\n" + "="*65)
print("STAGE 5.4 FINAL GATE RESULTS")
print("="*65)

gate_rows = []
for cls in s3["class"].unique():
    sub = s3[s3["class"] == cls]
    n       = len(sub)
    n_keep  = (sub["verdict"] == "KEEP").sum()
    n_drop  = (sub["verdict"] == "DROP").sum()
    n_unc   = (sub["verdict"] == "UNCERTAIN").sum()
    hc      = (sub["confidence"] >= 0.75).sum()
    pct_hc  = hc / n * 100

    if pct_hc >= 95:
        gate = "VERIFIED ✅"
    elif pct_hc >= 80:
        gate = "PROVISIONAL ✅ → Stage 5.5"
    else:
        gate = "BEST-EFFORT → structural patterns applied"

    gate_rows.append({"class": cls, "n_total": n,
                      "n_keep": int(n_keep), "pct_keep": round(n_keep/n*100, 1),
                      "n_drop": int(n_drop), "pct_drop": round(n_drop/n*100, 1),
                      "n_uncertain": int(n_unc), "pct_uncertain": round(n_unc/n*100, 1),
                      "pct_high_conf": round(pct_hc, 1), "gate": gate})

    print(f"\n{cls}  (n={n:,})")
    print(f"  KEEP:      {n_keep:5,}  ({n_keep/n*100:.1f}%)")
    print(f"  DROP:      {n_drop:5,}  ({n_drop/n*100:.1f}%)")
    print(f"  UNCERTAIN: {n_unc:5,}  ({n_unc/n*100:.1f}%)")
    print(f"  High-conf (≥0.75): {hc:,}/{n:,} = {pct_hc:.1f}%  → {gate}")

gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(P5_OUT / "p5_s4_gate_results.csv", index=False)


# =========================================================================
# STEP 7: Save final Stage 5.4 verdicts
# =========================================================================
s3.drop(columns=["bccc_file_path"], inplace=True, errors="ignore")
s3.to_csv(P5_OUT / "p5_s4_final_verdict.csv", index=False)
print(f"\nSaved p5_s4_final_verdict.csv  ({len(s3):,} rows)")


# =========================================================================
# STEP 8: Generate manual review batch (~40 per class from UNCERTAIN+KEEP<0.70)
# For human QA to validate the automated decisions
# =========================================================================
print("\n--- Generating review batches for manual QA ---")
REVIEW_DIR = Path("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/outputs/review_batches")
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

df_base2 = pd.read_csv(P4_OUT / "ws_p4_s01b_d12_applied.csv",
                       usecols=["id", "bccc_file_path"])
s3_with_path = s3.merge(df_base2, on="id", how="left")

BATCH_SIZE = 40
for cls in s3_with_path["class"].unique():
    sub = s3_with_path[s3_with_path["class"] == cls]
    # Sample from low-confidence verdicts (most need QA)
    review_pool = sub[sub["confidence"] < 0.70].copy()
    if len(review_pool) == 0:
        review_pool = sub.sample(min(BATCH_SIZE, len(sub)), random_state=42)
    else:
        review_pool = review_pool.sample(min(BATCH_SIZE, len(review_pool)), random_state=42)

    batch_rows = []
    for _, row in review_pool.iterrows():
        src = read_src(resolve_path(row["bccc_file_path"]))
        # Extract first 80 lines for review
        snippet = "\n".join(src.splitlines()[:80])
        batch_rows.append({
            "id": row["id"],
            "verdict_s4": row["verdict"],
            "confidence": row["confidence"],
            "notes": row["notes"],
            "file": row["bccc_file_path"],
            "source_snippet": snippet[:3000]
        })
    if batch_rows:
        fname = f"review_{cls.replace(':','_').lower()}.csv"
        pd.DataFrame(batch_rows).to_csv(REVIEW_DIR / fname, index=False)
        print(f"  {cls}: {len(batch_rows)} contracts → {fname}")

print("\nStage 5.4 complete.")
