"""
Task 10 + Task 24: Graph-Token Contract Hash Alignment
-------------------------------------------------------
Verifies that for 100 random stems, the graph .pt and token .pt refer to the
same .sol source (matching contract_hash, contract_path, and decoded tokens).
Tasks 10 and 24 are functionally identical — this script covers both.

Run:
    python task_10_24_hash_alignment.py
"""
import hashlib
import random
from pathlib import Path
from common import get_dirs, load_csv, load_graph, load_token, sol_from_graph, print_header

N_STEMS   = 100   # stems to check for hash/path alignment
N_DECODE  = 10    # stems to also decode tokens and compare to source

def compute_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def main():
    print_header("10 + 24", "Graph-Token Contract Hash Alignment")
    _, _, graphs_dir, tokens_dir, _, bccc_dir = get_dirs()

    df = load_csv()
    all_stems = df["md5_stem"].astype(str).tolist()

    # Only keep stems where BOTH files exist
    paired = [s for s in all_stems
              if (graphs_dir / f"{s}.pt").exists() and (tokens_dir / f"{s}.pt").exists()]
    print(f"  Stems with both graph and token: {len(paired):,} / {len(all_stems):,}")

    random.seed(42)
    sample = random.sample(paired, min(N_STEMS, len(paired)))

    hash_matches      = 0
    path_matches      = 0
    hash_mismatches   = []
    path_mismatches   = []
    missing_hash_g    = 0
    missing_hash_t    = 0
    missing_path_g    = 0
    missing_path_t    = 0
    sol_md5_matches   = 0
    sol_md5_checked   = 0

    for stem in sample:
        g = load_graph(graphs_dir / f"{stem}.pt")
        t = load_token(tokens_dir / f"{stem}.pt")

        g_hash = getattr(g, "contract_hash", None)
        t_hash = t.get("contract_hash") if isinstance(t, dict) else getattr(t, "contract_hash", None)
        g_path = getattr(g, "contract_path", None)
        t_path = t.get("contract_path") if isinstance(t, dict) else getattr(t, "contract_path", None)

        if g_hash is None: missing_hash_g += 1
        if t_hash is None: missing_hash_t += 1
        if g_path is None: missing_path_g += 1
        if t_path is None: missing_path_t += 1

        # Hash comparison
        if g_hash and t_hash:
            if g_hash == t_hash:
                hash_matches += 1
            else:
                hash_mismatches.append((stem, g_hash, t_hash))

        # Path comparison
        if g_path and t_path:
            gp = Path(g_path).name
            tp = Path(t_path).name
            if gp == tp:
                path_matches += 1
            else:
                path_mismatches.append((stem, g_path, t_path))

        # .sol MD5 vs stem
        sol = sol_from_graph(g, bccc_dir)
        if sol:
            actual_md5 = compute_md5(sol)
            sol_md5_checked += 1
            if actual_md5 == stem:
                sol_md5_matches += 1

    hash_total  = len(sample) - missing_hash_g - missing_hash_t
    path_total  = len(sample) - missing_path_g - missing_path_t

    print(f"""
Hash Alignment Report (n={len(sample)})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hash matches (g==t):   {hash_matches:>4} / {hash_total:>4}  ({hash_matches/max(hash_total,1)*100:.1f}%)
Path matches (g==t):   {path_matches:>4} / {path_total:>4}  ({path_matches/max(path_total,1)*100:.1f}%)
.sol MD5 == stem:      {sol_md5_matches:>4} / {sol_md5_checked:>4}  ({sol_md5_matches/max(sol_md5_checked,1)*100:.1f}%)

Missing contract_hash in graphs: {missing_hash_g}
Missing contract_hash in tokens: {missing_hash_t}
Missing contract_path in graphs: {missing_path_g}
Missing contract_path in tokens: {missing_path_t}
""")

    if hash_mismatches:
        print(f"[BUG] Hash mismatches ({len(hash_mismatches)}):")
        for stem, gh, th in hash_mismatches[:10]:
            print(f"  {stem}  graph={gh[:16]}…  token={th[:16]}…")
    else:
        print("[CONFIRMED] No hash mismatches found.")

    if path_mismatches:
        print(f"\n[BUG] Path mismatches ({len(path_mismatches)}):")
        for stem, gp, tp in path_mismatches[:10]:
            print(f"  {stem}")
            print(f"    graph: {gp}")
            print(f"    token: {tp}")
    else:
        print("[CONFIRMED] No path mismatches found.")

    # ── Token decode spot-check ───────────────────────────────────────────────
    print(f"\n── Token decode spot-check (n={N_DECODE}) ──────────────────────────")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    except Exception as e:
        print(f"  [SKIP] Cannot load CodeBERT tokenizer: {e}")
        return

    decode_sample = random.sample(paired, min(N_DECODE, len(paired)))
    decode_ok = 0
    for stem in decode_sample:
        g = load_graph(graphs_dir / f"{stem}.pt")
        t = load_token(tokens_dir / f"{stem}.pt")
        sol = sol_from_graph(g, bccc_dir)
        if sol is None:
            print(f"  {stem}: [SKIP] .sol file not found")
            continue
        sol_text = sol.read_text(errors="replace")[:2000]

        # Decode first window
        ids = t["input_ids"] if isinstance(t, dict) else t.input_ids
        decoded = tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
        first_200 = decoded[:500]

        # Compare by looking for a distinctive substring from the source
        # (exact match is impossible due to tokenisation artefacts)
        source_fragment = sol_text[:200].strip()[:50]
        match = source_fragment.replace(" ", "") in decoded.replace(" ", "")
        status = "OK" if match else "MISMATCH"
        if match:
            decode_ok += 1
        print(f"  {stem}: [{status}]")
        if not match:
            print(f"    Source starts: {repr(source_fragment[:60])}")
            print(f"    Decoded starts: {repr(first_200[:80])}")

    print(f"\n  Decode match: {decode_ok}/{min(N_DECODE, len(decode_sample))}")

if __name__ == "__main__":
    main()
