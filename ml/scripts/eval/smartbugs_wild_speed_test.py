"""smartbugs_wild_speed_test.py — Speed + prediction distribution test on SmartBugs Wild.

Loads Run 12 checkpoint and runs inference on a sample of N real-world
Ethereum mainnet contracts. Measures:
  - Time per prediction (mean, p50, p95, p99)
  - Throughput (predictions/sec)
  - Per-class prediction distribution (which classes the model triggers most)
  - Top-N contracts by max confidence (the "most confident" predictions)
  - Per-class mean confidence (calibration check on real data)

Output: console report + ml/reports/Run12_smartbugs_wild_speed_N{N}.json
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
sys.path.insert(0, str(REPO_ROOT))

from ml.src.inference.predictor import Predictor

WILD_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-wild"
CONTRACTS_DIR = WILD_DIR / "contracts"


def sample_contracts(n: int, seed: int = 42) -> list:
    """Random sample of N .sol files from the wild dataset."""
    import random
    rng = random.Random(seed)
    all_files = sorted(CONTRACTS_DIR.glob("*.sol"))
    rng.shuffle(all_files)
    return all_files[:n]


def run_speed_test(n: int, batch_size: int = 1) -> dict:
    print(f"\n=== SMARTBUGS WILD SPEED TEST (N={n}, batch_size={batch_size}) ===")
    print(f"Source: {CONTRACTS_DIR}")
    print(f"Total wild contracts: {sum(1 for _ in CONTRACTS_DIR.glob('*.sol'))}")

    # Sample
    print(f"\nSampling {n} contracts...")
    files = sample_contracts(n)
    print(f"  sampled: {len(files)}")

    # Load model
    print("\nInitialising Predictor (with warmup)...")
    t0 = time.time()
    predictor = Predictor(
        checkpoint="ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt",
        threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    warmup_time = time.time() - t0
    print(f"  ready in {warmup_time:.2f}s (includes warmup)")
    print(f"  architecture: {getattr(predictor, 'architecture', 'N/A')}")

    # Per-class info (for distribution)
    from data_module.sentinel_data.representation.graph_schema import CLASS_NAMES

    # Predict
    print(f"\nRunning inference on {n} contracts...")
    times = []
    errors = []
    class_distribution = Counter()  # class -> number of times it's the top prediction
    class_confidence_sum = defaultdict(float)  # class -> sum of max probs
    class_confidence_count = defaultdict(int)
    highest_confidence = []  # (prob, class, name)
    triggered_counts = []  # how many classes have >= 0.5 per contract
    n_with_any_trigger = 0

    for i, fpath in enumerate(files):
        try:
            source = fpath.read_text()
            t_start = time.time()
            result = predictor.predict_source(source, name=fpath.name)
            t_elapsed = time.time() - t_start
            times.append(t_elapsed)
            probs = result["probabilities"]

            # Top class
            top_class = max(probs, key=probs.get)
            top_prob = probs[top_class]
            class_distribution[top_class] += 1
            class_confidence_sum[top_class] += top_prob
            class_confidence_count[top_class] += 1
            highest_confidence.append((top_prob, top_class, fpath.name))

            # How many classes have >= 0.5
            triggered = sum(1 for p in probs.values() if p >= 0.5)
            triggered_counts.append(triggered)
            if triggered > 0:
                n_with_any_trigger += 1
        except Exception as e:
            errors.append((fpath.name, str(e)[:200]))
            continue
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n}")

    # Stats
    if not times:
        print(f"  NO successful predictions ({len(errors)} errors)")
        return {"n_contracts": n, "n_successful": 0, "n_errors": len(errors), "errors": errors}

    times_arr = np.array(times)
    mean_class_conf = {c: class_confidence_sum[c] / class_confidence_count[c] for c in class_confidence_sum}

    # Report
    print(f"\n=== SPEED STATS ===")
    print(f"  Successful: {len(times)} / {n} ({100*len(times)/n:.1f}%)")
    print(f"  Errors: {len(errors)}")
    print(f"  Time per prediction (excluding warmup):")
    print(f"    mean:  {times_arr.mean()*1000:.1f} ms")
    print(f"    p50:   {np.percentile(times_arr, 50)*1000:.1f} ms")
    print(f"    p95:   {np.percentile(times_arr, 95)*1000:.1f} ms")
    print(f"    p99:   {np.percentile(times_arr, 99)*1000:.1f} ms")
    print(f"    total: {times_arr.sum():.1f}s")
    print(f"  Throughput: {len(times)/times_arr.sum():.2f} predictions/sec")
    print(f"  Extrapolated to 47,398 contracts: {47398 * times_arr.mean() / 60:.1f} minutes")

    print(f"\n=== PREDICTION DISTRIBUTION (top class) ===")
    for cls, n_count in class_distribution.most_common():
        pct = 100 * n_count / len(times)
        mean_conf = mean_class_conf[cls]
        print(f"  {cls:30s}: {n_count:4d} ({pct:.1f}%)  mean conf={mean_conf:.3f}")

    print(f"\n=== TRIGGER STATS (classes with prob >= 0.5) ===")
    print(f"  Contracts with >=1 trigger: {n_with_any_trigger}/{len(times)} ({100*n_with_any_trigger/len(times):.1f}%)")
    if triggered_counts:
        tc = np.array(triggered_counts)
        print(f"  Mean triggers per contract: {tc.mean():.2f}")
        print(f"  Max triggers: {tc.max()}")
        print(f"  p50: {np.percentile(tc, 50):.0f}, p95: {np.percentile(tc, 95):.0f}")

    print(f"\n=== TOP 10 HIGHEST CONFIDENCE PREDICTIONS ===")
    highest_confidence.sort(reverse=True)
    for prob, cls, name in highest_confidence[:10]:
        print(f"  {prob:.3f}  {cls:30s}  {name}")

    # Save report
    report = {
        "n_contracts": n,
        "n_successful": len(times),
        "n_errors": len(errors),
        "warmup_time_sec": warmup_time,
        "time_per_pred_ms": {
            "mean": float(times_arr.mean() * 1000),
            "p50": float(np.percentile(times_arr, 50) * 1000),
            "p95": float(np.percentile(times_arr, 95) * 1000),
            "p99": float(np.percentile(times_arr, 99) * 1000),
            "total": float(times_arr.sum()),
        },
        "throughput_per_sec": float(len(times) / times_arr.sum()),
        "extrapolated_full_dataset_minutes": float(47398 * times_arr.mean() / 60),
        "class_distribution": dict(class_distribution),
        "class_mean_confidence": mean_class_conf,
        "n_with_any_trigger": n_with_any_trigger,
        "trigger_stats": {
            "mean_per_contract": float(np.mean(triggered_counts)) if triggered_counts else 0,
            "max_per_contract": int(np.max(triggered_counts)) if triggered_counts else 0,
            "p50": float(np.percentile(triggered_counts, 50)) if triggered_counts else 0,
            "p95": float(np.percentile(triggered_counts, 95)) if triggered_counts else 0,
        },
        "top_10_highest_confidence": [
            {"prob": prob, "class": cls, "name": name}
            for prob, cls, name in highest_confidence[:10]
        ],
        "errors_sample": errors[:10],
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of contracts to predict on")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="ml/reports/Run12_smartbugs_wild_speed_N100.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path = Path(str(output_path).replace("N100", f"N{args.n}"))
    report = run_speed_test(args.n, args.batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written → {output_path}")


if __name__ == "__main__":
    main()
