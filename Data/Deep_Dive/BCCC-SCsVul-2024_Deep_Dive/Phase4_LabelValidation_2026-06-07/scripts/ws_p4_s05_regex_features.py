"""
Stage 0.5 — Compute 31 regex features on all 67,311 contracts in v1.1.

Features are simple string-presence or count-based, computed by one-pass
scan over the source. Output: ws_p4_s05_regex_features.csv (67311 x 31).

The 31 features (categorized):
[Pragma]    f01_pragma_04, f02_pragma_05, f03_pragma_06, f04_pragma_07, f05_pragma_08
[Timestamp] f06_block_timestamp, f07_now
[Tx]        f08_tx_origin
[Calls]     f09_delegatecall, f10_callvalue, f11_transfer, f12_send,
            f13_lowlevel_call, f14_ecrecover
[Crypto]    f15_keccak256, f16_sha3
[Control]   f17_selfdestruct, f18_assembly, f19_unchecked_block, f20_try_catch
[Safety]    f21_require, f22_assert, f23_revert
[Types]     f24_uint8_decl, f25_address_payable, f26_payable_modifier
[Funcs]     f27_fallback, f28_receive_eth, f29_constructor
[Mods]      f30_modifier_decl, f31_event_decl
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
V11 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_applied.csv"
SRC = REPO / "BCCC-SCsVul-2024/SourceCodes"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s05_regex_features.csv"

FEATURE_NAMES = [
    "f01_pragma_04", "f02_pragma_05", "f03_pragma_06", "f04_pragma_07", "f05_pragma_08",
    "f06_block_timestamp", "f07_now",
    "f08_tx_origin",
    "f09_delegatecall", "f10_callvalue", "f11_transfer", "f12_send",
    "f13_lowlevel_call", "f14_ecrecover",
    "f15_keccak256", "f16_sha3",
    "f17_selfdestruct", "f18_assembly", "f19_unchecked_block", "f20_try_catch",
    "f21_require", "f22_assert", "f23_revert",
    "f24_uint8_decl", "f25_address_payable", "f26_payable_modifier",
    "f27_fallback", "f28_receive_eth", "f29_constructor",
    "f30_modifier_decl", "f31_event_decl",
]

PATTERNS = {
    "f01_pragma_04": re.compile(r"pragma\s+solidity\s+\^?0\.4\.|pragma\s+solidity\s+0\.4\."),
    "f02_pragma_05": re.compile(r"pragma\s+solidity\s+\^?0\.5\.|pragma\s+solidity\s+0\.5\."),
    "f03_pragma_06": re.compile(r"pragma\s+solidity\s+\^?0\.6\.|pragma\s+solidity\s+0\.6\."),
    "f04_pragma_07": re.compile(r"pragma\s+solidity\s+\^?0\.7\.|pragma\s+solidity\s+0\.7\."),
    "f05_pragma_08": re.compile(r"pragma\s+solidity\s+\^?0\.8\.|pragma\s+solidity\s+0\.8\."),
    "f06_block_timestamp": re.compile(r"\bblock\.timestamp\b"),
    "f07_now": re.compile(r"\bnow\b"),
    "f08_tx_origin": re.compile(r"\btx\.origin\b"),
    "f09_delegatecall": re.compile(r"\.delegatecall\s*\("),
    "f10_callvalue": re.compile(r"\.call\s*\{[^}]*value\s*:"),
    "f11_transfer": re.compile(r"\.transfer\s*\("),
    "f12_send": re.compile(r"\.send\s*\("),
    "f13_lowlevel_call": re.compile(r"\.call\s*\("),
    "f14_ecrecover": re.compile(r"\becrecover\s*\("),
    "f15_keccak256": re.compile(r"\bkeccak256\s*\("),
    "f16_sha3": re.compile(r"\bsha3\s*\("),
    "f17_selfdestruct": re.compile(r"\bselfdestruct\s*\(|\bsuicide\s*\("),
    "f18_assembly": re.compile(r"\bassembly\s*(?:\(|[\{\"])"),
    "f19_unchecked_block": re.compile(r"\bunchecked\s*\{"),
    "f20_try_catch": re.compile(r"\btry\s+[a-zA-Z_]"),
    "f21_require": re.compile(r"\brequire\s*\("),
    "f22_assert": re.compile(r"\bassert\s*\("),
    "f23_revert": re.compile(r"\brevert\s*\("),
    "f24_uint8_decl": re.compile(r"\buint8\b|\bint8\b|\buint\d+\b|\bint\d+\b"),
    "f25_address_payable": re.compile(r"\baddress\s+payable\b"),
    "f26_payable_modifier": re.compile(r"\bpayable\b"),
    "f27_fallback": re.compile(r"\bfunction\s*\(\s*\)\s*(?:public\s+|external\s+)*(?:payable\s+)?[\{\"]"),
    "f28_receive_eth": re.compile(r"\breceive\s*\(\s*\)\s*(?:external\s+)*payable"),
    "f29_constructor": re.compile(r"\bconstructor\s*\("),
    "f30_modifier_decl": re.compile(r"\bmodifier\s+[a-zA-Z_]\w*\s*\("),
    "f31_event_decl": re.compile(r"\bevent\s+[a-zA-Z_]\w*\s*\("),
}


def extract_features(src: str) -> dict:
    return {k: 1 if p.search(src) else 0 for k, p in PATTERNS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=V11)
    ap.add_argument("--src-root", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} contracts")

    out_fieldnames = ["id"] + FEATURE_NAMES
    out_rows = []
    n_missing = 0
    for i, row in enumerate(rows):
        rel = row["bccc_file_path"]
        rel_clean = rel
        if rel_clean.startswith("BCCC-SCsVul-2024/Source Codes/"):
            rel_clean = rel_clean.replace("BCCC-SCsVul-2024/Source Codes/", "")
        elif rel_clean.startswith("BCCC-SCsVul-2024/SourceCodes/"):
            rel_clean = rel_clean.replace("BCCC-SCsVul-2024/SourceCodes/", "")
        p = args.src_root / rel_clean
        if not p.exists():
            n_missing += 1
            feats = {k: 0 for k in FEATURE_NAMES}
        else:
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                n_missing += 1
                feats = {k: 0 for k in FEATURE_NAMES}
            else:
                feats = extract_features(src)
        out_rows.append({"id": row["id"], **feats})
        if (i + 1) % 5000 == 0:
            print(f"  features computed: {i+1}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {args.out} ({len(out_rows)} rows, {n_missing} missing source)")

    n_all_zero = 0
    n_all_one = 0
    for r in out_rows:
        vals = [int(r[k]) for k in FEATURE_NAMES]
        if sum(vals) == 0:
            n_all_zero += 1
        elif sum(vals) == len(FEATURE_NAMES):
            n_all_one += 1
    print(f"Rows with all-zero features: {n_all_zero}")
    print(f"Rows with all-one features: {n_all_one}")
    from collections import Counter
    freq = Counter()
    for r in out_rows:
        for k in FEATURE_NAMES:
            if int(r[k]) == 1:
                freq[k] += 1
    print(f"Per-feature frequency (top 10):")
    for k, v in freq.most_common(10):
        print(f"  {k}: {v} ({100*v/len(out_rows):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
