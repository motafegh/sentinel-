"""Re-analyze SmartBugs Wild eval results on OOD-only contracts.

Joins the per-contract eval predictions (ml/data/smartbugs_wild_eval_state.json)
with the contamination index (ml/reports/Run12_smartbugs_wild_contamination_index.json)
and recomputes all distribution/trigger stats separately for:

  - OOD-only   (not in v3 train/val/test — the honest deployment signal)
  - Seen-train (in v3 train split — expected to be better)
  - Seen-val/test
  - Full 47K   (for comparison)

Outputs:
  ml/reports/Run12_smartbugs_wild_ood_analysis.json
  ml/reports/Run12_smartbugs_wild_ood_analysis_summary.md

Usage:
    ml/.venv/bin/python ml/scripts/audit/analyze_wild_ood.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
EVAL_STATE = REPO_ROOT / "ml" / "data" / "smartbugs_wild_eval_state.json"
CONTAM_INDEX = REPO_ROOT / "ml" / "reports" / "Run12_smartbugs_wild_contamination_index.json"
REPORTS_DIR = REPO_ROOT / "ml" / "reports"

CLASSES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]


def compute_stats(records: list[dict]) -> dict:
    """Compute distribution and trigger stats over a list of eval records."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    class_counts: dict[str, int] = defaultdict(int)
    class_prob_sum: dict[str, float] = defaultdict(float)
    n_any_trigger = 0
    triggered_counts: list[int] = []
    time_secs: list[float] = []

    for r in records:
        top = r.get("top_class")
        if top:
            class_counts[top] += 1
        probs = r.get("all_probs", {})
        for cls, p in probs.items():
            class_prob_sum[cls] += p
        nt = r.get("n_triggered_tuned", 0)
        if nt > 0:
            n_any_trigger += 1
        triggered_counts.append(nt)
        time_secs.append(r.get("time_sec", 0))

    triggered_counts.sort()
    time_secs.sort()

    def pct(lst, p):
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst) - 1)]

    class_dist = {
        cls: class_counts.get(cls, 0) for cls in CLASSES
    }
    class_pct = {
        cls: round(100 * class_counts.get(cls, 0) / n, 2) for cls in CLASSES
    }
    class_mean_conf = {
        cls: round(class_prob_sum.get(cls, 0) / n, 4) for cls in CLASSES
    }

    return {
        "n": n,
        "n_any_trigger": n_any_trigger,
        "pct_any_trigger": round(100 * n_any_trigger / n, 2),
        "mean_triggers_per_contract": round(sum(triggered_counts) / n, 3),
        "p50_triggers": pct(triggered_counts, 50),
        "p95_triggers": pct(triggered_counts, 95),
        "max_triggers": max(triggered_counts) if triggered_counts else 0,
        "class_distribution": class_dist,
        "class_pct": class_pct,
        "class_mean_confidence": class_mean_conf,
        "time_mean_ms": round(1000 * sum(time_secs) / n, 1),
        "time_p50_ms": round(1000 * pct(time_secs, 50), 1),
        "time_p95_ms": round(1000 * pct(time_secs, 95), 1),
    }


def render_markdown(result: dict) -> str:
    ts = result["generated_at"]
    s = result["stats"]

    def table_row(label, ood, seen_tr, full):
        return f"| {label:<28} | {ood:>8} | {seen_tr:>10} | {full:>7} |"

    header = (
        f"# SmartBugs Wild Eval — OOD vs Seen Analysis\n\n"
        f"**Generated:** {ts}\n\n"
        f"## Key Numbers\n\n"
        f"| Subset | N | Contaminated | Source |\n"
        f"|--------|---|---|---|\n"
        f"| Full Wild | {result['n_full']:,} | — | all 47K |\n"
        f"| OOD (not in v3) | {result['n_ood']:,} | 0% | honest signal |\n"
        f"| Seen — train | {result['n_seen_train']:,} | 100% | in v3 train |\n"
        f"| Seen — val/test | {result['n_seen_valtest']:,} | 100% | in v3 val/test |\n"
        f"| Eval errors | {result['n_errors']:,} | — | Slither parse fail |\n\n"
    )

    # Trigger stats table
    ood = s["ood"]
    seen_tr = s["seen_train"]
    full = s["full"]

    trigger_block = (
        f"## Trigger Stats (n_triggered_tuned)\n\n"
        f"| Metric | OOD ({ood['n']:,}) | Seen-train ({seen_tr['n']:,}) | Full ({full['n']:,}) |\n"
        f"|--------|---------|------------|------|\n"
        f"| Any trigger % | {ood['pct_any_trigger']}% | {seen_tr['pct_any_trigger']}% | {full['pct_any_trigger']}% |\n"
        f"| Mean triggers | {ood['mean_triggers_per_contract']} | {seen_tr['mean_triggers_per_contract']} | {full['mean_triggers_per_contract']} |\n"
        f"| p50 triggers | {ood['p50_triggers']} | {seen_tr['p50_triggers']} | {full['p50_triggers']} |\n"
        f"| p95 triggers | {ood['p95_triggers']} | {seen_tr['p95_triggers']} | {full['p95_triggers']} |\n"
        f"| Max triggers | {ood['max_triggers']} | {seen_tr['max_triggers']} | {full['max_triggers']} |\n\n"
    )

    # Class distribution table
    dist_header = (
        f"## Top-class Distribution (% of subset)\n\n"
        f"| Class | OOD % | Seen-train % | Full % | Δ OOD vs Full |\n"
        f"|-------|-------|-------------|--------|---------------|\n"
    )
    dist_rows = ""
    for cls in CLASSES:
        ood_p = ood["class_pct"].get(cls, 0)
        st_p = seen_tr["class_pct"].get(cls, 0)
        fu_p = full["class_pct"].get(cls, 0)
        delta = round(ood_p - fu_p, 2)
        sign = "+" if delta >= 0 else ""
        dist_rows += f"| {cls:<26} | {ood_p:>5}% | {st_p:>11}% | {fu_p:>5}% | {sign}{delta:>5}pp |\n"

    conf_header = (
        f"\n## Mean Confidence per Class\n\n"
        f"| Class | OOD | Seen-train | Full |\n"
        f"|-------|-----|-----------|------|\n"
    )
    conf_rows = ""
    for cls in CLASSES:
        ood_c = ood["class_mean_confidence"].get(cls, 0)
        st_c = seen_tr["class_mean_confidence"].get(cls, 0)
        fu_c = full["class_mean_confidence"].get(cls, 0)
        conf_rows += f"| {cls:<26} | {ood_c:.4f} | {st_c:.4f} | {fu_c:.4f} |\n"

    interp = (
        f"\n## Interpretation\n\n"
        f"- **{ood['pct_any_trigger']}%** of truly OOD contracts triggered at least one vulnerability class\n"
        f"  (vs {full['pct_any_trigger']}% over the full 47K)\n"
        f"- Seen-train contracts triggered at {seen_tr['pct_any_trigger']}% — "
        f"{'higher' if seen_tr['pct_any_trigger'] > ood['pct_any_trigger'] else 'lower or equal'} "
        f"than OOD, expected if model has memorised those contracts\n"
        f"- **{result['n_ood']:,} contracts** ({round(100*result['n_ood']/result['n_full'],1)}% of Wild) "
        f"are genuine OOD — this is the honest benchmark pool\n"
        f"- Contamination: {result['n_seen_train']:,} in train, "
        f"{result['n_seen_valtest']:,} in val/test\n"
    )

    return header + trigger_block + dist_header + dist_rows + conf_header + conf_rows + interp


def main() -> None:
    print("Loading eval state...")
    with open(EVAL_STATE) as f:
        state = json.load(f)
    processed = state["processed_set"]
    print(f"  Eval records: {len(processed):,}")

    print("Loading contamination index...")
    with open(CONTAM_INDEX) as f:
        contam = json.load(f)
    contam_map = {r["address"]: r for r in contam}
    print(f"  Contamination records: {len(contam_map):,}")

    # Partition
    ood_records = []
    seen_train_records = []
    seen_valtest_records = []
    error_addresses = []

    for addr, pred in processed.items():
        # Skip eval errors (no top_class means prediction failed)
        if not pred.get("top_class"):
            error_addresses.append(addr)
            continue
        c = contam_map.get(addr, {})
        tier_hit = c.get("tier_hit", 0)
        split = c.get("v3_split")
        if tier_hit == 0:
            ood_records.append(pred)
        elif split == "train":
            seen_train_records.append(pred)
        else:
            seen_valtest_records.append(pred)

    all_successful = ood_records + seen_train_records + seen_valtest_records

    print(f"\n  OOD:           {len(ood_records):,}")
    print(f"  Seen-train:    {len(seen_train_records):,}")
    print(f"  Seen-val/test: {len(seen_valtest_records):,}")
    print(f"  Eval errors:   {len(error_addresses):,}")

    print("\nComputing stats...")
    stats = {
        "ood": compute_stats(ood_records),
        "seen_train": compute_stats(seen_train_records),
        "seen_valtest": compute_stats(seen_valtest_records),
        "full": compute_stats(all_successful),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_full": len(processed),
        "n_successful": len(all_successful),
        "n_errors": len(error_addresses),
        "n_ood": len(ood_records),
        "n_seen_train": len(seen_train_records),
        "n_seen_valtest": len(seen_valtest_records),
        "contamination_rate_pct": round(
            100 * (len(seen_train_records) + len(seen_valtest_records)) / len(all_successful), 2
        ) if all_successful else 0,
        "stats": stats,
    }

    # Write JSON
    json_path = REPORTS_DIR / "Run12_smartbugs_wild_ood_analysis.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Write markdown
    md = render_markdown(result)
    md_path = REPORTS_DIR / "Run12_smartbugs_wild_ood_analysis_summary.md"
    with open(md_path, "w") as f:
        f.write(md)

    # Print summary
    ood = stats["ood"]
    seen = stats["seen_train"]
    full = stats["full"]
    print(f"\n{'='*65}")
    print("OOD vs SEEN — KEY STATS")
    print(f"{'='*65}")
    print(f"{'Metric':<30} {'OOD':>8} {'Seen-train':>12} {'Full':>8}")
    print(f"{'-'*65}")
    print(f"{'N contracts':<30} {ood['n']:>8,} {seen['n']:>12,} {full['n']:>8,}")
    print(f"{'Any trigger %':<30} {ood['pct_any_trigger']:>7}% {seen['pct_any_trigger']:>11}% {full['pct_any_trigger']:>7}%")
    print(f"{'Mean triggers':<30} {ood['mean_triggers_per_contract']:>8} {seen['mean_triggers_per_contract']:>12} {full['mean_triggers_per_contract']:>8}")
    print(f"\nTop-class distribution (OOD vs Full):")
    for cls in CLASSES:
        ood_p = ood["class_pct"].get(cls, 0)
        full_p = full["class_pct"].get(cls, 0)
        delta = round(ood_p - full_p, 2)
        bar = ("▲" if delta > 0 else "▼") + f"{abs(delta):.1f}pp"
        if ood_p > 0:
            print(f"  {cls:<26} OOD={ood_p:>5}%  Full={full_p:>5}%  {bar}")
    print(f"\nJSON    → {json_path}")
    print(f"Markdown→ {md_path}\n")


if __name__ == "__main__":
    main()
