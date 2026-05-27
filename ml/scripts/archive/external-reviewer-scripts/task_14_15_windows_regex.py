"""
Task 14: Window Sub-sampling Vulnerability Coverage
Task 15: in_unchecked Regex False Positive Check
-------------------------------------------------
Run:
    python task_14_15_windows_regex.py [--task 14|15|both]
"""
import sys
import re
import random
import tempfile
from pathlib import Path
from common import (
    get_dirs, load_csv, load_graph, load_token,
    sol_from_graph, LABEL_COLS, print_header
)

VULN_PATTERNS = {
    "low_level_call": re.compile(r'\.call\s*[\({]'),
    "block_timestamp": re.compile(r'block\.timestamp|block\.number|\bnow\b'),
    "unchecked_arith": re.compile(r'unchecked\s*\{'),
    "send_transfer": re.compile(r'\.(send|transfer)\s*\('),
    "reentrancy": re.compile(r'\.call\s*\{.*?value'),
}

# ════════════════════════════════════════════════════════════════════════════
# TASK 14
# ════════════════════════════════════════════════════════════════════════════

def task_14():
    print_header(14, "Window Sub-sampling Vulnerability Coverage")
    _, _, graphs_dir, tokens_dir, _, bccc_dir = get_dirs()
    df = load_csv()

    print("  Scanning for contracts with num_windows > 4 (>~2048 tokens)…")
    all_stems = df["md5_stem"].astype(str).tolist()
    random.seed(42)
    sample = random.sample(all_stems, min(3000, len(all_stems)))

    multi_window = []
    for stem in sample:
        tpath = tokens_dir / f"{stem}.pt"
        if not tpath.exists(): continue
        t = load_token(tpath)
        nw = t.get("num_windows") if isinstance(t, dict) else getattr(t, "num_windows", None)
        # "more than 4" means the contract needed more than max_windows; check num_tokens
        ntok = t.get("num_tokens") if isinstance(t, dict) else getattr(t, "num_tokens", None)
        if ntok and int(ntok) > 2000:
            multi_window.append(stem)

    print(f"  Contracts with >2000 tokens: {len(multi_window)} of {len(sample)} sampled")

    if not multi_window:
        print("  [CONFIRMED] All contracts fit within 4 windows — sub-sampling never triggers.")
        print("  This is expected: most Solidity contracts are short.")
        return

    inspect = random.sample(multi_window, min(10, len(multi_window)))

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    except ImportError:
        print("  [SKIP] transformers not available for Task 14.")
        return

    print(f"\n  {'Stem':<16} {'W_total':>8} {'Vuln_wins':>9} {'Sub_wins':>8} {'Coverage':>9}")
    print("  " + "-" * 56)

    coverage_rates = []
    for stem in inspect:
        gpath = graphs_dir / f"{stem}.pt"
        tpath = tokens_dir / f"{stem}.pt"
        if not gpath.exists() or not tpath.exists(): continue

        try:
            g = load_graph(gpath)
            t = load_token(tpath)
        except Exception as e:
            print(f"  {stem}: [ERROR] {e}")
            continue

        sol = sol_from_graph(g, bccc_dir)
        if sol is None:
            print(f"  {stem}: [no .sol]")
            continue
        source = sol.read_text(errors="replace")

        # Tokenize FULL source with return_overflowing_tokens
        enc = tokenizer(
            source,
            max_length=512,
            stride=256,
            return_overflowing_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        total_windows = enc["input_ids"].shape[0]

        # Decode each window and check for vuln patterns
        vuln_window_indices = []
        for wi in range(total_windows):
            decoded = tokenizer.decode(enc["input_ids"][wi].tolist(), skip_special_tokens=True)
            if any(p.search(decoded) for p in VULN_PATTERNS.values()):
                vuln_window_indices.append(wi)

        # Sub-sampled windows from disk (linspace of 4)
        import numpy as np
        if total_windows <= 4:
            sub_indices = list(range(total_windows))
        else:
            sub_indices = list(set(
                int(round(i)) for i in np.linspace(0, total_windows - 1, 4)
            ))

        # Coverage: what fraction of vuln windows are in sub-sample?
        if vuln_window_indices:
            covered = sum(1 for wi in vuln_window_indices if wi in sub_indices)
            cov_rate = covered / len(vuln_window_indices)
            coverage_rates.append(cov_rate)
            cov_str = f"{cov_rate*100:.0f}%"
        else:
            cov_str = "N/A"

        print(f"  {stem:<16} {total_windows:>8} {len(vuln_window_indices):>9} "
              f"{len(sub_indices):>8} {cov_str:>9}")

    if coverage_rates:
        print(f"\n  Mean vulnerability coverage by linspace sub-sampling: "
              f"{sum(coverage_rates)/len(coverage_rates)*100:.1f}%")
        if min(coverage_rates) < 0.5:
            print("  [BUG] Some contracts lose >50% of vulnerability-relevant windows.")
        else:
            print("  [CONFIRMED] Linspace sub-sampling retains most vulnerability windows.")

# ════════════════════════════════════════════════════════════════════════════
# TASK 15 — in_unchecked regex false positive
# ════════════════════════════════════════════════════════════════════════════

UNCHECKED_REGEX = re.compile(r'\bunchecked\s*\{')

FP_TEST_CASES = [
    ("comment_fp",   "// unchecked { this is a comment }\nfunction f() public {}"),
    ("string_fp",    'string memory s = "unchecked {"; function f() public {}'),
    ("real_positive", "function f() public { unchecked { uint x = a + b; } }"),
    ("no_unchecked",  "function f() public { uint x = a + b; }"),
]

def task_15():
    print_header(15, "in_unchecked Regex False Positive Check")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()
    df = load_csv()

    # 1. Check if any real contracts have in_unchecked=1.0
    print("  Scanning 500 graphs for in_unchecked [9] == 1.0…")
    paths = list(graphs_dir.glob("*.pt"))
    random.seed(42)
    sample = random.sample(paths, min(500, len(paths)))
    fires_count = 0
    fire_examples = []
    for p in sample:
        try:
            g = load_graph(p)
            x = g.x.numpy()
            if (x[:, 9] == 1.0).any():
                fires_count += 1
                fire_examples.append(p.stem)
        except Exception:
            continue

    print(f"  Graphs with in_unchecked=1.0: {fires_count} / {len(sample)}")
    if fires_count == 0:
        print("  [CONFIRMED] in_unchecked is dead in practice (BUG-5).")
    else:
        print(f"  [NEW FINDING] in_unchecked fires in {fires_count} graphs!")
        for s in fire_examples[:5]:
            print(f"    {s}")

    # 2. Regex false positive test with synthetic sources
    print("\n  Regex false positive tests:")
    print(f"  {'Test case':<20} {'Regex hits':>11} {'Expected':>10}")
    print("  " + "-" * 46)
    for name, source in FP_TEST_CASES:
        hits = len(UNCHECKED_REGEX.findall(source))
        expected = "1" if "real_positive" in name else "0"
        status = "OK" if (hits > 0) == ("real_positive" in name) else "FALSE POSITIVE/NEG"
        print(f"  {name:<20} {hits:>11}  {expected:>9}  {status}")

    # 3. Try running extract_contract_graph on synthetic sources
    print("\n  Attempting live extraction test on synthetic 0.8.x contract…")
    try:
        sys.path.insert(0, str(get_dirs()[1] / "src"))
        from preprocessing.graph_extractor import extract_contract_graph

        comment_src = """
pragma solidity ^0.8.0;
contract TestComment {
    function f(uint a, uint b) public returns (uint) {
        // unchecked { this should not fire }
        return a + b;
    }
}
"""
        real_src = """
pragma solidity ^0.8.0;
contract TestReal {
    function f(uint a, uint b) public returns (uint) {
        unchecked { return a + b; }
    }
}
"""
        for label, src in [("comment_src", comment_src), ("real_unchecked", real_src)]:
            with tempfile.NamedTemporaryFile(suffix=".sol", mode="w", delete=False) as f:
                f.write(src)
                fname = f.name
            try:
                g = extract_contract_graph(fname)
                if g:
                    x = g.x.numpy()
                    fires = int((x[:, 9] == 1.0).any())
                    print(f"    {label}: in_unchecked fires = {fires}")
            except Exception as e:
                print(f"    {label}: [ERROR] {e}")
            finally:
                Path(fname).unlink(missing_ok=True)
    except ImportError as e:
        print(f"  [SKIP] Cannot import graph_extractor: {e}")

    print("\n── Summary Task 15 ──────────────────────────────────────────────────────")
    print("  Check regex output above for false positives on comment/string cases.")


def main():
    mode = "both"
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        mode = sys.argv[idx + 1]
    if mode in ("14", "both"): task_14()
    if mode in ("15", "both"): task_15()

if __name__ == "__main__":
    main()
