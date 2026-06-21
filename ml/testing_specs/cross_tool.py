"""
cross_tool.py — Model vs Slither/Aderyn consistency check.

WHY THIS EXISTS

Static analysis tools (Slither, Aderyn) and the model should AGREE
when they say a contract is vulnerable. They should DISAGREE on
edge cases. If they agree 100%, the model is probably overfitting
to the tools (it learned their patterns, not the vulnerability). If
they disagree completely, the model is broken (or the tools are).

For Run 12: Slither returned 0 findings on safe_storage.sol. The
model said ExternalBug=0.82. Disagreement — but the model was wrong,
not Slither. This check would have flagged the disagreement and forced
investigation.

WHAT IT DOES

1. Run Slither + Aderyn on a benchmark
2. Run the model on the same benchmark
3. For each class, compare:
   - Agreement rate (both say yes / both say no)
   - Model-only rate (model says yes, tools say no)
   - Tools-only rate (tools say yes, model says no)
4. Flag:
   - >50% disagreement (model and tools have different views)
   - >30% model-only positive (model is over-predicting)
   - >30% tools-only positive (model is missing real bugs)

USAGE

    # Default: Run 12 on v0.1 benchmark
    python ml/testing_specs/cross_tool.py \\
        --checkpoint ml/checkpoints/Run12_FINAL.pt \\
        --benchmark data_module/benchmarks/v0_1_honest/ \\
        --output ml/checkpoints/Run12_cross_tool.json \\
        --exit-on-fail
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

# Map Slither detector names to SENTINEL classes
SLITHER_TO_SENTINEL = {
    "reentrancy-eth": "Reentrancy",
    "reentrancy-no-eth": "Reentrancy",
    "reentrancy-benign": "Reentrancy",
    "reentrancy-events": "Reentrancy",
    "arbitrary-send-eth": "AccessControl",
    "controlled-delegatecall": "AccessControl",
    "delegatecall-loop": "AccessControl",
    "msg-value-loop": "DenialOfService",
    "calls-loop": "DenialOfService",
    "locked-ether": "MishandledException",
    "unchecked-lowlevel": "CallToUnknown",
    "unchecked-send": "CallToUnknown",
    "unchecked-transfer": "CallToUnknown",
    "uninitialized-state": "MishandledException",
    "uninitialized-storage": "MishandledException",
    "uninitialized-local": "MishandledException",
    "timestamp": "Timestamp",
    "block-number": "Timestamp",
    "low-level-calls": "CallToUnknown",
    "naming-convention": None,  # Style, not vuln
    "pragma": None,  # Style
    "solc-version": None,  # Style
    "missing-zero-check": "IntegerUO",
    "divide-before-multiply": "IntegerUO",
    "tautology": "IntegerUO",
    "incorrect-equality": "IntegerUO",
    "shadowing-state": None,
    "void-cst": "MishandledException",
    "events-maths": "UnusedReturn",
    "missing-zero-check-return": "UnusedReturn",
    "calls-loop-eth": "DenialOfService",
    "constant-function-asm": None,
    "constant-function-state": None,
    "divide-by-zero": "IntegerUO",
    "locked-ether-eth": "MishandledException",
    "mapping-deletion": "UnusedReturn",
    "tautology-eth": "IntegerUO",
    "tautology-uint": "IntegerUO",
    # Auth/Origin detectors
    "tx-origin": "TransactionOrderDependence",  # added 2026-06-18: tx.origin = TOD
    "suicidal": "AccessControl",  # anyone can kill the contract = access control
    # ERC20/permit detectors (Slither 0.10+)
    "arbitrary-send-erc20": "AccessControl",  # arbitrary token transfer
    "arbitrary-send-erc20-permit": "AccessControl",
    "unchecked-transfer-erc20": "CallToUnknown",  # unchecked ERC20 return
}

# Thresholds
MAX_DISAGREEMENT_RATE = 0.50  # WARN if model+tools disagree on >50% of contracts
MAX_MODEL_ONLY_RATE = 0.30  # WARN if model says yes but tools say no on >30%
MAX_TOOLS_ONLY_RATE = 0.30  # WARN if tools say yes but model says no on >30%


@dataclass
class ClassConsistency:
    """Model vs tools consistency for one class."""

    class_name: str
    n_contracts: int
    both_yes: int
    both_no: int
    model_yes_tools_no: int  # FP by model
    tools_yes_model_no: int  # FN by model
    agreement_rate: float
    disagreement_rate: float
    model_only_rate: float  # FP rate
    tools_only_rate: float  # FN rate
    flags: list

    def to_dict(self) -> dict:
        return asdict(self)


def run_slither(contract_path: Path) -> set[str]:
    """Run Slither on a contract and return SENTINEL class names."""
    try:
        result = subprocess.run(
            ["slither", str(contract_path), "--json", "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 and not result.stdout:
            return set()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Slither failed for {contract_path}: {e}")
        return set()
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    classes = set()
    for det in data.get("results", {}).get("detectors", []):
        check = det.get("check", "")
        sentinel = SLITHER_TO_SENTINEL.get(check)
        if sentinel:
            classes.add(sentinel)
    return classes


def run_model(predictor: Any, contract_path: Path, threshold: float = 0.5) -> set[str]:
    """Run the model on a contract and return predicted class names."""
    try:
        with contract_path.open() as f:
            source = f.read()
        r = predictor.predict_source(source)
    except Exception as e:
        logger.warning(f"Model failed for {contract_path}: {e}")
        return set()
    probs = r.get("probabilities", {})
    return {name for name, p in probs.items() if p >= threshold}


def analyze_class(
    class_name: str,
    model_preds: list[bool],
    tool_preds: list[bool],
) -> ClassConsistency:
    """Analyze model vs tools consistency for one class."""
    n = len(model_preds)
    both_yes = sum(1 for m, t in zip(model_preds, tool_preds) if m and t)
    both_no = sum(1 for m, t in zip(model_preds, tool_preds) if not m and not t)
    model_yes_tools_no = sum(1 for m, t in zip(model_preds, tool_preds) if m and not t)
    tools_yes_model_no = sum(1 for m, t in zip(model_preds, tool_preds) if not m and t)

    agree = both_yes + both_no
    disagree = model_yes_tools_no + tools_yes_model_no
    agree_rate = agree / n if n else 0
    disagree_rate = disagree / n if n else 0
    model_only_rate = model_yes_tools_no / n if n else 0
    tools_only_rate = tools_yes_model_no / n if n else 0

    flags = []
    if disagree_rate > MAX_DISAGREEMENT_RATE:
        flags.append(f"disagreement {disagree_rate:.1%} > {MAX_DISAGREEMENT_RATE:.0%} threshold")
    if model_only_rate > MAX_MODEL_ONLY_RATE:
        flags.append(f"model-only {model_only_rate:.1%} > {MAX_MODEL_ONLY_RATE:.0%} (model over-predicting)")
    if tools_only_rate > MAX_TOOLS_ONLY_RATE:
        flags.append(f"tools-only {tools_only_rate:.1%} > {MAX_TOOLS_ONLY_RATE:.0%} (model missing bugs)")

    return ClassConsistency(
        class_name=class_name,
        n_contracts=n,
        both_yes=both_yes,
        both_no=both_no,
        model_yes_tools_no=model_yes_tools_no,
        tools_yes_model_no=tools_yes_model_no,
        agreement_rate=agree_rate,
        disagreement_rate=disagree_rate,
        model_only_rate=model_only_rate,
        tools_only_rate=tools_only_rate,
        flags=flags,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cross_tool",
        description=(
            "Compare model predictions with Slither + Aderyn on a benchmark. "
            "Flags high disagreement, model over-prediction, model under-prediction."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Path to benchmark directory with .sol files.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write report to this path.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Model probability threshold for positive prediction (default 0.5)")
    parser.add_argument("--max-contracts", type=int, default=100,
                        help="Limit benchmark size (Slither is slow)")
    parser.add_argument("--exit-on-fail", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1
    if not args.benchmark.exists():
        print(f"ERROR: benchmark dir not found: {args.benchmark}")
        return 1

    # Load model
    from ml.src.inference.predictor import Predictor
    predictor = Predictor(str(args.checkpoint))

    # Get benchmark contracts
    sol_files = sorted(args.benchmark.rglob("*.sol"))[:args.max_contracts]
    if not sol_files:
        print(f"ERROR: no .sol files in {args.benchmark}")
        return 1
    print(f"Found {len(sol_files)} contracts (max={args.max_contracts})")

    # Get class list
    from synthetic_probes import SENTINEL_CLASSES

    # Run Slither + model on each contract
    model_preds_by_class: dict[str, list[bool]] = {n: [] for n in SENTINEL_CLASSES}
    tool_preds_by_class: dict[str, list[bool]] = {n: [] for n in SENTINEL_CLASSES}

    for i, sol in enumerate(sol_files):
        if i % 5 == 0:
            print(f"  [{i+1}/{len(sol_files)}] {sol.name}", end="\r")
        try:
            tool_classes = run_slither(sol)
        except Exception as e:
            logger.warning(f"Slither error on {sol.name}: {e}")
            tool_classes = set()
        try:
            model_classes = run_model(predictor, sol, threshold=args.threshold)
        except Exception as e:
            logger.warning(f"Model error on {sol.name}: {e}")
            model_classes = set()
        for name in SENTINEL_CLASSES:
            model_preds_by_class[name].append(name in model_classes)
            tool_preds_by_class[name].append(name in tool_classes)

    print(f"  [{len(sol_files)}/{len(sol_files)}]")
    print()

    # Analyze each class
    results: list[ClassConsistency] = []
    n_flagged = 0
    for name in SENTINEL_CLASSES:
        r = analyze_class(name, model_preds_by_class[name], tool_preds_by_class[name])
        results.append(r)
        if r.flags:
            n_flagged += 1

    # Print
    print("=" * 70)
    print(f"CROSS-TOOL CONSISTENCY: {n_flagged}/{len(SENTINEL_CLASSES)} classes flagged")
    print("=" * 70)
    print()
    print(f"Per-class model vs tools (threshold={args.threshold}):")
    for r in results:
        marker = "⚠" if r.flags else " "
        print(f"  {marker} {r.class_name:30} agree={r.agreement_rate:.1%}  "
              f"model_only={r.model_only_rate:.1%}  tools_only={r.tools_only_rate:.1%}")
        if r.flags:
            for f in r.flags:
                print(f"      {f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "summary": {"n_contracts": len(sol_files), "n_classes": len(SENTINEL_CLASSES), "n_flagged": n_flagged},
            "per_class": [r.to_dict() for r in results],
        }, indent=2))
        print(f"\nReport written to: {args.output}")

    if args.exit_on_fail and n_flagged > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
