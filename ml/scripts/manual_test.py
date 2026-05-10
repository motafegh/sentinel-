"""
manual_test.py — Run the v4 predictor against hand-crafted contracts.

Usage:
    TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/manual_test.py \
        --checkpoint ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt \
        --contracts ml/scripts/test_contracts/

Output: per-contract table of predicted classes, probabilities, and
        whether the expected class was detected above threshold.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.inference.predictor import Predictor

CLASS_NAMES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]

GREEN = "\033[92m"
RED   = "\033[91m"
AMBER = "\033[93m"
RESET = "\033[0m"
PASS  = f"{GREEN}✓{RESET}"
FAIL  = f"{RED}✗{RESET}"
WARN  = f"{AMBER}~{RESET}"


def get_all_probs(predictor: Predictor, source: str) -> tuple[list[float], dict]:
    """
    Run inference and return (all_class_probs, result_dict).
    predict_source only returns above-threshold classes; we also need raw probs
    for the full table, so we call the preprocessor + model directly.
    """
    graph, windows = predictor.preprocessor.process_source_windowed(source)
    predictor.model.eval()

    with torch.no_grad():
        batch = Batch.from_data_list([graph]).to(predictor.device)
        if len(windows) == 1:
            tokens = windows[0]
            input_ids     = tokens["input_ids"].to(predictor.device)
            attention_mask = tokens["attention_mask"].to(predictor.device)
            logits = predictor.model(batch, input_ids, attention_mask)
            probs  = torch.sigmoid(logits.float()).squeeze(0).cpu()
        else:
            # windowed: take max across windows (same as predictor._aggregate_window_predictions)
            all_probs = []
            for tokens in windows:
                input_ids      = tokens["input_ids"].to(predictor.device)
                attention_mask = tokens["attention_mask"].to(predictor.device)
                logits = predictor.model(batch, input_ids, attention_mask)
                all_probs.append(torch.sigmoid(logits.float()).squeeze(0).cpu())
            probs = torch.stack(all_probs).max(dim=0).values

    all_probs_list = probs.tolist()
    # also get the structured result for label / vulnerability list
    result = predictor.predict_source(source)
    return all_probs_list, result


def run(checkpoint: Path, contracts_dir: Path) -> None:
    print(f"\nLoading checkpoint: {checkpoint}")
    predictor = Predictor(checkpoint=str(checkpoint), device="cuda")
    thresholds = predictor.thresholds.cpu().tolist()
    print("Model loaded.\n")

    sol_files = sorted(contracts_dir.glob("*.sol"))
    if not sol_files:
        print(f"No .sol files found in {contracts_dir}")
        return

    rows = []
    for sol_path in sol_files:
        source = sol_path.read_text()
        first_line = source.strip().splitlines()[0] if source.strip() else ""
        expected: list[str] = []
        if first_line.startswith("// expect:"):
            raw = first_line.replace("// expect:", "").strip()
            expected = [c.strip() for c in raw.split(",") if c.strip()]

        all_probs, result = get_all_probs(predictor, source)
        detected = {v["vulnerability_class"] for v in result.get("vulnerabilities", [])}
        rows.append((sol_path.stem, expected, all_probs, detected))

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{'Contract':<35} {'Expected':<28} {'Detected (≥threshold)'}")
    print("=" * 80)

    total_expected = 0
    total_hit      = 0
    safe_contracts = 0
    safe_correct   = 0

    for name, expected, _, detected in rows:
        det_str = ", ".join(sorted(detected)) if detected else "(none)"
        exp_str = ", ".join(expected)         if expected else "(safe)"

        if expected:
            hits = sum(1 for e in expected if e in detected)
            total_expected += len(expected)
            total_hit      += hits
            icon = PASS if hits == len(expected) else (WARN if hits > 0 else FAIL)
        else:
            safe_contracts += 1
            is_clean = len(detected) == 0
            safe_correct += int(is_clean)
            icon = PASS if is_clean else WARN

        print(f"  {icon} {name:<33} {exp_str:<28} {det_str}")

    print("=" * 80)

    # ── Probability table ──────────────────────────────────────────────────
    short = [c[:5] for c in CLASS_NAMES]
    header = f"\n{'Contract':<35}" + "".join(f"{s:>7}" for s in short)
    print(header)
    print("-" * (35 + 7 * len(CLASS_NAMES)))

    for name, expected, all_probs, detected in rows:
        row = f"{name:<35}"
        for i, (cls, prob, thr) in enumerate(zip(CLASS_NAMES, all_probs, thresholds)):
            is_detected = cls in detected
            is_expected = cls in expected
            cell = f"{prob:.3f}"
            if is_detected and is_expected:
                cell = f"{GREEN}{cell}{RESET}"   # hit — green
            elif is_detected and not is_expected:
                cell = f"{AMBER}{cell}{RESET}"   # false positive — amber
            elif not is_detected and is_expected:
                cell = f"{RED}{cell}{RESET}"     # miss — red
            row += f"{cell:>7}"
        print(row)

    print(f"\n  {GREEN}green{RESET}=hit  {AMBER}amber{RESET}=false-pos  {RED}red{RESET}=miss  plain=correct-negative")
    print(f"\n  Thresholds: " + " ".join(f"{c[:5]}={t:.2f}" for c, t in zip(CLASS_NAMES, thresholds)))

    # ── Score ──────────────────────────────────────────────────────────────
    print()
    if total_expected > 0:
        pct = 100 * total_hit // total_expected
        print(f"  Vulnerability detection: {total_hit}/{total_expected} ({pct}%)")
    if safe_contracts > 0:
        print(f"  Safe-contract specificity: {safe_correct}/{safe_contracts} correctly clean")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt",
    )
    parser.add_argument(
        "--contracts",
        default="ml/scripts/test_contracts",
    )
    args = parser.parse_args()
    run(Path(args.checkpoint), Path(args.contracts))


if __name__ == "__main__":
    main()
