"""
diag_per_eye_solidifi.py — Per-eye breakdown of SENTINEL Run 9 on SolidiFI

For each SolidiFI contract, runs a forward pass with return_aux=True to capture
the four independent eye predictions:
  - gnn_eye        (structural opinion from Phase-3 aggregated function nodes)
  - transformer_eye (semantic opinion from GraphCodeBERT + LoRA)
  - fused_eye      (joint structural+semantic from CrossAttentionFusion)
  - cfg_eye        (Phase-2 CFG-specific opinion)
  - combined       (final 4-eye classifier output)

Outputs:
  /tmp/sentinel_solidifi_per_eye.json  — per-contract per-eye probability tables
  Console                              — summary tables + notable contracts

Usage (repo root, venv active):
    python -m ml.scripts.diag_per_eye_solidifi [--verbose]
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch
from torch_geometric.data import Batch

REPO_ROOT    = Path(__file__).resolve().parents[2]
SOLIDIFI_DIR = REPO_ROOT / "Data" / "data" / "raw" / "solidifi" / "repo" / "buggy_contracts"
DEFAULT_CKPT = REPO_ROOT / "ml" / "checkpoints" / "GCB-P1-Run9-v11-20260606_best.pt"
OUT_JSON     = Path("/tmp/sentinel_solidifi_per_eye.json")

NEAR_DUP_STEMS = {
    "buggy_1", "buggy_11", "buggy_13", "buggy_16", "buggy_25",
    "buggy_27", "buggy_35", "buggy_37", "buggy_43",
}

SOLIDIFI_TO_SENTINEL: dict[str, str] = {
    "Re-entrancy":          "Reentrancy",
    "Overflow-Underflow":   "IntegerUO",
    "TOD":                  "TransactionOrderDependence",
    "Timestamp-Dependency": "Timestamp",
    "Unchecked-Send":       "CallToUnknown",
    "Unhandled-Exceptions": "MishandledException",
}

_TRAINING_MAX_WINDOWS = 4


def _build_windowed_tensors(windows: list[dict], device: torch.device):
    """Replicate the exact padding logic from Predictor._score_windowed."""
    selected = windows[:_TRAINING_MAX_WINDOWS]
    pad_ids  = torch.zeros(1, 512, dtype=torch.long, device=device)
    pad_mask = torch.zeros(1, 512, dtype=torch.long, device=device)
    padded = list(selected)
    while len(padded) < _TRAINING_MAX_WINDOWS:
        padded.append({"input_ids": pad_ids, "attention_mask": pad_mask})
    stacked_ids  = torch.cat([w["input_ids"].to(device)      for w in padded], dim=0).unsqueeze(0)
    stacked_mask = torch.cat([w["attention_mask"].to(device) for w in padded], dim=0).unsqueeze(0)
    return stacked_ids, stacked_mask  # [1, W, 512]


def run(checkpoint: Path, verbose: bool) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from ml.src.inference.predictor import Predictor
    from ml.src.training.trainer import CLASS_NAMES

    print(f"\n{'='*70}")
    print("SENTINEL Run 9 — Per-Eye Diagnostic on SolidiFI")
    print(f"{'='*70}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Classes    : {CLASS_NAMES}")
    print()

    predictor = Predictor(checkpoint=str(checkpoint))
    model     = predictor.model
    device    = predictor.device
    model.eval()

    # ── Collect contracts ────────────────────────────────────────────────────
    contracts: list[tuple[str, Path, bool]] = []
    for cat_dir in sorted(SOLIDIFI_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            near_dup = (sol.stem in NEAR_DUP_STEMS and cat_dir.name == "Unchecked-Send")
            contracts.append((cat_dir.name, sol, near_dup))

    total    = len(contracts)
    all_recs: list[dict] = []
    errors   = 0

    print(f"Processing {total} contracts ...\n")

    for i, (category, sol_path, near_dup) in enumerate(contracts, 1):
        tag = f"[{i:3d}/{total}] {category}/{sol_path.name}"
        try:
            source = sol_path.read_text(encoding="utf-8", errors="replace")
            graph, windows = predictor.preprocessor.process_source_windowed(source)

            batch        = Batch.from_data_list([graph]).to(device)
            ids, mask    = _build_windowed_tensors(windows, device)

            with torch.no_grad():
                logits, aux = model(batch, ids, mask, return_aux=True)

            # Main combined probabilities
            combined_probs = torch.sigmoid(logits.float()).squeeze(0).cpu().tolist()

            # Per-eye probabilities via auxiliary heads
            eye_probs: dict[str, list[float]] = {}
            for eye_name in ("gnn", "transformer", "fused", "phase2"):
                raw = aux[eye_name]
                eye_probs[eye_name] = torch.sigmoid(raw.float()).squeeze(0).cpu().tolist()

            # Build per-class dicts
            def to_dict(prob_list: list[float]) -> dict[str, float]:
                return {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, prob_list)}

            rec = {
                "category":    category,
                "file":        sol_path.name,
                "near_dup":    near_dup,
                "num_nodes":   int(graph.num_nodes),
                "num_edges":   int(graph.num_edges),
                "combined":    to_dict(combined_probs),
                "eye_gnn":     to_dict(eye_probs["gnn"]),
                "eye_tf":      to_dict(eye_probs["transformer"]),
                "eye_fused":   to_dict(eye_probs["fused"]),
                "eye_phase2":  to_dict(eye_probs["phase2"]),
                "error":       None,
            }

            # Compute rank of correct class in combined output
            sentinel_cls = SOLIDIFI_TO_SENTINEL.get(category)
            if sentinel_cls:
                correct_idx = CLASS_NAMES.index(sentinel_cls)
                correct_p_raw = combined_probs[correct_idx]      # raw float — no rounding
                sorted_desc   = sorted(combined_probs, reverse=True)
                rank          = sorted_desc.index(correct_p_raw) + 1  # 1-based
                rec["correct_class"] = sentinel_cls
                rec["correct_prob"]  = round(correct_p_raw, 4)
                rec["correct_rank"]  = rank
            else:
                rec["correct_class"] = None
                rec["correct_rank"]  = None

            all_recs.append(rec)

            if verbose:
                rank_str = f"rank={rank}" if sentinel_cls else "unmapped"
                print(f"{tag}  nodes={graph.num_nodes:3d}  {rank_str}  "
                      f"combined={correct_p:.3f}" if sentinel_cls else
                      f"{tag}  nodes={graph.num_nodes:3d}")

        except Exception as exc:
            errors += 1
            all_recs.append({
                "category": category, "file": sol_path.name,
                "near_dup": near_dup, "error": str(exc),
            })
            print(f"{tag}  ERROR: {exc}")
            if verbose:
                traceback.print_exc()

    print(f"\nDone. {total - errors}/{total} succeeded, {errors} errors.")

    # ── Save JSON ────────────────────────────────────────────────────────────
    OUT_JSON.write_text(json.dumps(all_recs, indent=2))
    print(f"Results saved → {OUT_JSON}\n")

    # ── Summary tables ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOP-K RANK ANALYSIS (combined 4-eye output)")
    print(f"{'='*70}")
    _print_topk_table(all_recs, CLASS_NAMES)

    print(f"\n{'='*70}")
    print("PER-EYE AVERAGE PROBABILITY (correct class vs others)")
    print(f"{'='*70}")
    _print_per_eye_table(all_recs, CLASS_NAMES)

    print(f"\n{'='*70}")
    print("NOTABLE CONTRACTS (Top-1 misses + top-1 hits for each category)")
    print(f"{'='*70}")
    _print_notable(all_recs)


def _print_topk_table(recs: list[dict], class_names: list[str]) -> None:
    cats = sorted(SOLIDIFI_TO_SENTINEL.keys())
    print(f"  {'Category':<26} {'Class':<28} {'N':>4} {'Top-1%':>7} {'Top-2%':>7} {'Top-3%':>7}")
    print("  " + "-" * 82)
    for cat in cats:
        cls = SOLIDIFI_TO_SENTINEL[cat]
        ok  = [r for r in recs if r["category"] == cat and not r.get("error") and not r.get("near_dup")]
        n   = len(ok)
        if n == 0:
            continue
        t1 = sum(1 for r in ok if isinstance(r.get("correct_rank"), int) and r["correct_rank"] == 1)
        t2 = sum(1 for r in ok if isinstance(r.get("correct_rank"), int) and 1 <= r["correct_rank"] <= 2)
        t3 = sum(1 for r in ok if isinstance(r.get("correct_rank"), int) and 1 <= r["correct_rank"] <= 3)
        print(f"  {cat:<26} {cls:<28} {n:>4} {t1/n*100:>6.0f}% {t2/n*100:>6.0f}% {t3/n*100:>6.0f}%")


def _print_per_eye_table(recs: list[dict], class_names: list[str]) -> None:
    eyes  = ["eye_gnn", "eye_tf", "eye_fused", "eye_phase2", "combined"]
    elabels = ["GNN", "Transformer", "Fused", "Phase2/CFG", "Combined"]
    cats  = sorted(SOLIDIFI_TO_SENTINEL.keys())

    print(f"  {'Category':<26}  " + "  ".join(f"{l:<12}" for l in elabels))
    print("  " + "-" * (26 + 2 + len(elabels) * 14))
    for cat in cats:
        cls = SOLIDIFI_TO_SENTINEL[cat]
        ok  = [r for r in recs if r["category"] == cat and not r.get("error") and not r.get("near_dup")]
        if not ok:
            continue
        row = f"  {cat:<26}  "
        for eye in eyes:
            vals = [r[eye][cls] for r in ok if eye in r and cls in r[eye]]
            avg  = sum(vals) / len(vals) if vals else 0
            row += f"  {avg:.4f}      "
        print(row)

    print()
    print("  → Each cell = average P(correct_class) for that eye across all contracts in category.")
    print("  → Values close to 0.5 = random. Values > 0.65 = eye is learning this class.")


def _print_notable(recs: list[dict]) -> None:
    cats = sorted(SOLIDIFI_TO_SENTINEL.keys())
    for cat in cats:
        cls = SOLIDIFI_TO_SENTINEL[cat]
        ok  = [r for r in recs if r["category"] == cat and not r.get("error") and not r.get("near_dup")]
        if not ok:
            continue
        by_rank = sorted(ok, key=lambda r: (r.get("correct_rank", 99), -r.get("correct_prob", 0)))
        print(f"\n  [{cat}] → SENTINEL class: {cls}")
        print(f"  {'File':<20} {'Rank':>5} {'P(correct)':>11} {'P(top-1 class)':>16} {'Top-1 class':<28}")
        print("  " + "-" * 86)
        # Show best 3 and worst 3
        show = by_rank[:3] + (by_rank[-3:] if len(by_rank) > 6 else [])
        prev = -1
        for r in show:
            if prev >= 0 and by_rank.index(r) > prev + 1:
                print("  ...")
            prev = by_rank.index(r)
            rank = r.get("correct_rank", "?")
            cp   = r.get("correct_prob", 0)
            # top-1 prediction
            comb = r.get("combined", {})
            top1_cls = max(comb, key=comb.get) if comb else "?"
            top1_p   = comb.get(top1_cls, 0) if comb else 0
            print(f"  {r['file']:<20} {rank:>5} {cp:>11.4f} {top1_p:>16.4f} {top1_cls:<28}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    run(args.checkpoint, args.verbose)


if __name__ == "__main__":
    main()
