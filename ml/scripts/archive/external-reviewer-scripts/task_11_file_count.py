"""
Task 11: File Count Triple-Alignment
--------------------------------------
Counts rows in CSV, .pt files in graphs/ and tokens_windowed/.
Computes all set intersections and lists first 20 of each missing category.

Run:
    python task_11_file_count.py
"""
import json
from pathlib import Path
from common import get_dirs, load_csv, print_header

def main():
    print_header(11, "File Count Triple-Alignment")
    _, _, graphs_dir, tokens_dir, _, _ = get_dirs()

    # ── Load sets ────────────────────────────────────────────────────────────
    df = load_csv()
    csv_stems   = set(df["md5_stem"].astype(str).tolist())
    graph_stems = {p.stem for p in graphs_dir.glob("*.pt")}
    token_stems = {p.stem for p in tokens_dir.glob("*.pt")}

    # ── Set math ─────────────────────────────────────────────────────────────
    csv_and_graphs        = csv_stems & graph_stems
    csv_and_tokens        = csv_stems & token_stems
    graphs_and_tokens     = graph_stems & token_stems
    trainable             = csv_stems & graph_stems & token_stems

    in_csv_not_graphs     = csv_stems - graph_stems
    in_csv_not_tokens     = csv_stems - token_stems
    orphan_graphs         = graph_stems - csv_stems
    orphan_tokens         = token_stems - csv_stems
    graphs_no_tokens      = graph_stems - token_stems
    tokens_no_graphs      = token_stems - graph_stems

    # ── Report ────────────────────────────────────────────────────────────────
    pct = lambda n, d: f"{n/d*100:.1f}%" if d else "N/A"

    print(f"""
File Alignment Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSV rows:              {len(csv_stems):>7,}
Graph .pt files:       {len(graph_stems):>7,}
Token .pt files:       {len(token_stems):>7,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSV ∩ graphs:          {len(csv_and_graphs):>7,}  ({pct(len(csv_and_graphs), len(csv_stems))} of CSV)
CSV ∩ tokens:          {len(csv_and_tokens):>7,}  ({pct(len(csv_and_tokens), len(csv_stems))} of CSV)
graphs ∩ tokens:       {len(graphs_and_tokens):>7,}
CSV ∩ graphs ∩ tokens: {len(trainable):>7,}  ← TRAINABLE SET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Missing from graphs:   {len(in_csv_not_graphs):>7,}  ({pct(len(in_csv_not_graphs), len(csv_stems))} of CSV)
Missing from tokens:   {len(in_csv_not_tokens):>7,}  ({pct(len(in_csv_not_tokens), len(csv_stems))} of CSV)
Orphan graphs:         {len(orphan_graphs):>7,}
Orphan tokens:         {len(orphan_tokens):>7,}
Graphs without tokens: {len(graphs_no_tokens):>7,}
Tokens without graphs: {len(tokens_no_graphs):>7,}
""")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    ckpt_path = tokens_dir / "checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
            print(f"Retokenization checkpoint: completed={ckpt.get('completed', '?')}")
            print(f"  Keys: {list(ckpt.keys())}")
        except Exception as e:
            print(f"[WARN] Could not parse checkpoint.json: {e}")
    else:
        print("Retokenization checkpoint.json: NOT FOUND")

    # ── First 20 of each missing category ────────────────────────────────────
    def show_sample(label, items, n=20):
        lst = sorted(items)[:n]
        print(f"\n{label} (first {min(n, len(items))} of {len(items)}):")
        for s in lst:
            print(f"  {s}")

    show_sample("In CSV but NOT in graphs", in_csv_not_graphs)
    show_sample("In graphs but NOT in tokens", graphs_no_tokens)
    show_sample("In tokens but NOT in graphs", tokens_no_graphs)
    show_sample("Orphan graphs (not in CSV)", orphan_graphs)
    show_sample("Orphan tokens (not in CSV)", orphan_tokens)

    # ── Finding ───────────────────────────────────────────────────────────────
    print("\n── Finding ──────────────────────────────────────────────")
    if len(in_csv_not_graphs) == 0 and len(in_csv_not_tokens) == 0:
        print("  [CONFIRMED] CSV, graphs, and tokens are fully aligned.")
    else:
        if in_csv_not_graphs:
            print(f"  [BUG] {len(in_csv_not_graphs)} CSV rows have no graph — "
                  "model cannot train on these labels.")
        if in_csv_not_tokens:
            print(f"  [BUG] {len(in_csv_not_tokens)} CSV rows have no token file — "
                  "dual-path dataset will fail for these rows.")

if __name__ == "__main__":
    main()
