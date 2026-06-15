"""
smartbugs_wild_full_eval.py — Run 12 evaluation on the FULL SmartBugs Wild dataset.

DESIGN PRINCIPLES:
  1. RESUMABLE — atomic state file written every --checkpoint contracts.
     If the process dies (SIGKILL, OOM, network), re-run resumes from the
     last checkpoint. No work is lost.
  2. LIVE LOGS — timestamped progress written to ml/logs/smartbugs_wild_eval_<date>.log
     every --log-every seconds. tail -f friendly.
  3. GRACEFUL SHUTDOWN — SIGTERM handler saves state + flushes logs + exits 0.
     Use `kill -TERM <pid>` to stop cleanly. Use `kill -9` only as last resort.
  4. INCREMENTAL REPORTS — every --report-every contracts, save a JSON snapshot
     so progress is observable without waiting for completion.
  5. FINAL REPORT — docs/reports/Run12/2026-06-14_smartbugs_wild_full_eval_47K_in_progress/Run12_smartbugs_wild_FULL_<date>.json + .md
     with full per-class distribution, throughput, error analysis.

USAGE:
  # Full 47K (estimated 4.5 hours)
  ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py

  # Resume from saved state
  ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py --resume

  # Test with N=100 first
  ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py --max 100

  # Custom checkpoint interval
  ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py --checkpoint 200

  # Background with live tail
  nohup ml/.venv/bin/python ml/scripts/smartbugs_wild_full_eval.py > /dev/null 2>&1 &
  tail -f ml/logs/smartbugs_wild_eval_$(date +%Y%m%d).log

ARTIFACTS:
  ml/data/smartbugs_wild_eval_state.json      # Resumable state (atomic write)
  ml/data/smartbugs_wild_eval_state.json.tmp  # Temp during write
  ml/logs/smartbugs_wild_eval_<date>.log      # Live progress (append)
  docs/reports/Run12/2026-06-14_smartbugs_wild_full_eval_47K_in_progress/Run12_smartbugs_wild_FULL_<date>.json  # Final per-contract results
  docs/reports/Run12/2026-06-14_smartbugs_wild_full_eval_47K_in_progress/Run12_smartbugs_wild_FULL_<date>_summary.md  # Human-readable summary
  docs/reports/Run12/2026-06-14_smartbugs_wild_full_eval_47K_in_progress/Run12_smartbugs_wild_FULL_<date>_incremental_<n>.json  # Periodic snapshots
"""
import argparse
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
sys.path.insert(0, str(REPO_ROOT))

from ml.src.inference.predictor import Predictor

WILD_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-wild"
CONTRACTS_DIR = WILD_DIR / "contracts"
STATE_PATH = REPO_ROOT / "ml" / "data" / "smartbugs_wild_eval_state.json"
STATE_TMP = STATE_PATH.with_suffix(".json.tmp")
LOGS_DIR = REPO_ROOT / "ml" / "logs"
REPORTS_DIR = REPO_ROOT / "docs" / "reports" / "Run12" / "2026-06-14_smartbugs_wild_full_eval_47K_in_progress"

CHECKPOINT_DEFAULT = 500
LOG_EVERY_DEFAULT = 30
REPORT_EVERY_DEFAULT = 2000


class FullEvaluator:
    """Robust, resumable evaluator with live logging and graceful shutdown."""

    def __init__(self, args):
        self.args = args
        self.log_path = LOGS_DIR / f"smartbugs_wild_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_f = open(self.log_path, "a", buffering=1)  # line-buffered
        self.should_stop = False
        self.state = self._load_state() if args.resume else self._init_state()
        self.start_time = time.time()
        self.last_log_time = 0
        self.n_processed = len(self.state["processed_set"])
        self.n_errors = sum(1 for r in self.state["processed_set"].values() if r.get("error"))
        self.n_in_state = self.n_processed  # alias for clarity

        # Setup signal handler
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _init_state(self) -> dict:
        """Fresh state."""
        all_files = sorted(CONTRACTS_DIR.glob("*.sol"))
        return {
            "checkpoint_version": 1,
            "started_at": datetime.now().isoformat(),
            "last_updated": None,
            "config": {
                "n_total_contracts": len(all_files),
                "checkpoint_interval": self.args.checkpoint,
                "log_every_sec": self.args.log_every,
                "report_every": self.args.report_every,
                "max_contracts": self.args.max,
            },
            "processed_set": {},  # addr -> result dict
            "stats": {
                "total_time_sec": 0,
                "n_errors": 0,
                "n_skipped_resume": 0,
            },
        }

    def _load_state(self) -> dict:
        """Load existing state if present."""
        if not STATE_PATH.exists():
            self._log(f"No existing state at {STATE_PATH}, starting fresh")
            return self._init_state()
        try:
            state = json.loads(STATE_PATH.read_text())
            n_processed = len(state.get("processed_set", {}))
            self._log(f"Resumed from state: {n_processed} contracts already processed")
            return state
        except Exception as e:
            self._log(f"ERROR loading state from {STATE_PATH}: {e}")
            self._log("Starting fresh (old state may be corrupt; check)")
            return self._init_state()

    def _save_state(self):
        """Atomic state save: write to .tmp, then rename."""
        self.state["last_updated"] = datetime.now().isoformat()
        self.state["stats"]["total_time_sec"] = time.time() - self.start_time + self.state["stats"].get("baseline_time", 0)
        try:
            STATE_TMP.write_text(json.dumps(self.state, indent=2))
            STATE_TMP.replace(STATE_PATH)  # atomic on POSIX
        except Exception as e:
            self._log(f"ERROR saving state: {e}")

    def _on_signal(self, signum, frame):
        """Graceful shutdown handler — save state and exit."""
        signame = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self._log(f"\n[{signame}] received — saving state and exiting cleanly...")
        self.should_stop = True
        self._save_state()
        self._log(f"[{signame}] state saved. Resume with --resume flag.")
        self._log(f"[{signame}] log: {self.log_path}")
        self._log(f"[{signame}] state: {STATE_PATH}")
        self.log_f.close()
        sys.exit(0)

    def _log(self, msg: str):
        """Timestamped log to file (and stdout)."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def _maybe_log_progress(self, force: bool = False):
        """Periodic progress log every --log-every seconds."""
        now = time.time()
        if not force and (now - self.last_log_time) < self.args.log_every:
            return
        self.last_log_time = now

        elapsed = now - self.start_time + self.state["stats"].get("baseline_time", 0)
        n = self.n_processed
        total = self.args.max if self.args.max else self.state["config"]["n_total_contracts"]
        rate = n / elapsed if elapsed > 0 else 0
        eta_sec = (total - n) / rate if rate > 0 else 0
        eta_str = self._fmt_duration(eta_sec)
        elapsed_str = self._fmt_duration(elapsed)

        pct = 100 * n / total if total > 0 else 0
        self._log(
            f"PROGRESS: {n}/{total} ({pct:.1f}%) | "
            f"elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"rate: {rate:.2f}/s | errors: {self.n_errors}"
        )

    def _fmt_duration(self, sec: float) -> str:
        if sec < 0 or sec != sec:  # NaN check
            return "—"
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m}m{s:02d}s"

    def _save_incremental_report(self, snapshot_n: int):
        """Save incremental summary every --report-every contracts."""
        results = list(self.state["processed_set"].values())
        if not results:
            return
        snapshot_path = REPORTS_DIR / f"Run12_smartbugs_wild_FULL_{datetime.now().strftime('%Y%m%d')}_incremental_N{snapshot_n}.json"
        snapshot = self._compute_summary(results)
        snapshot["snapshot_n"] = snapshot_n
        snapshot["snapshot_at"] = datetime.now().isoformat()
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
        self._log(f"INCREMENTAL: snapshot saved → {snapshot_path.name}")

    def _compute_summary(self, results: list) -> dict:
        """Compute summary statistics from results list."""
        successful = [r for r in results if not r.get("error")]
        errors = [r for r in results if r.get("error")]

        if successful:
            times = [r["time_sec"] for r in successful]
            throughput = len(successful) / sum(times) if sum(times) > 0 else 0
        else:
            times = []
            throughput = 0

        # Per-class distribution
        class_top = Counter()
        class_conf_sum = defaultdict(float)
        class_conf_count = defaultdict(int)
        for r in successful:
            top = r["top_class"]
            class_top[top] += 1
            class_conf_sum[top] += r["top_prob"]
            class_conf_count[top] += 1
        class_mean_conf = {c: class_conf_sum[c] / class_conf_count[c] for c in class_conf_sum}

        # Trigger stats
        triggered = [r.get("n_triggered_tuned", 0) for r in successful]
        any_trigger = sum(1 for t in triggered if t > 0)
        return {
            "n_total": len(results),
            "n_successful": len(successful),
            "n_errors": len(errors),
            "elapsed_sec": time.time() - self.start_time + self.state["stats"].get("baseline_time", 0),
            "throughput_per_sec": throughput,
            "time_per_pred_ms": {
                "mean": float(np.mean(times)) * 1000 if times else 0,
                "p50": float(np.percentile(times, 50)) * 1000 if times else 0,
                "p95": float(np.percentile(times, 95)) * 1000 if times else 0,
                "p99": float(np.percentile(times, 99)) * 1000 if times else 0,
            },
            "class_distribution": dict(class_top.most_common()),
            "class_mean_confidence": class_mean_conf,
            "n_with_any_trigger": any_trigger,
            "trigger_stats": {
                "mean_per_contract": float(np.mean(triggered)) if triggered else 0,
                "p50": float(np.percentile(triggered, 50)) if triggered else 0,
                "p95": float(np.percentile(triggered, 95)) if triggered else 0,
                "max": int(np.max(triggered)) if triggered else 0,
            },
            "error_samples": [
                {"address": r["address"], "error": r["error"][:200]}
                for r in errors[:10]
            ],
        }

    def _save_final_report(self):
        """Save final per-contract + summary reports."""
        results = list(self.state["processed_set"].values())
        date_str = datetime.now().strftime("%Y%m%d")
        ts = datetime.now().strftime("%H%M%S")

        # Per-contract (could be large, 47K * ~500 bytes = ~25 MB)
        per_contract_path = REPORTS_DIR / f"Run12_smartbugs_wild_FULL_{date_str}_per_contract.json"
        per_contract_path.write_text(json.dumps(self.state["processed_set"], indent=2))
        self._log(f"FINAL: per-contract report → {per_contract_path.name} ({per_contract_path.stat().st_size/1e6:.1f} MB)")

        # Summary
        summary = self._compute_summary(results)
        summary["checkpoint_state_path"] = str(STATE_PATH)
        summary["log_path"] = str(self.log_path)
        summary["completed_at"] = datetime.now().isoformat()
        summary["total_elapsed_sec"] = time.time() - self.start_time + self.state["stats"].get("baseline_time", 0)

        summary_path = REPORTS_DIR / f"Run12_smartbugs_wild_FULL_{date_str}_{ts}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        self._log(f"FINAL: summary → {summary_path.name}")

        # Human-readable markdown
        md_path = REPORTS_DIR / f"Run12_smartbugs_wild_FULL_{date_str}_{ts}_summary.md"
        md = self._format_summary_md(summary)
        md_path.write_text(md)
        self._log(f"FINAL: markdown summary → {md_path.name}")

        return summary_path, md_path

    def _format_summary_md(self, s: dict) -> str:
        md = f"""# Run 12 SmartBugs Wild Full Evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview

- **Total contracts processed:** {s['n_total']:,}
- **Successful:** {s['n_successful']:,} ({100*s['n_successful']/max(1,s['n_total']):.1f}%)
- **Errors:** {s['n_errors']:,} ({100*s['n_errors']/max(1,s['n_total']):.1f}%)
- **Total elapsed:** {self._fmt_duration(s['total_elapsed_sec'])}
- **Throughput:** {s['throughput_per_sec']:.2f} predictions/sec

## Speed (excluding warmup)

| Percentile | Time (ms) |
|---|---|
| mean | {s['time_per_pred_ms']['mean']:.1f} |
| p50 | {s['time_per_pred_ms']['p50']:.1f} |
| p95 | {s['time_per_pred_ms']['p95']:.1f} |
| p99 | {s['time_per_pred_ms']['p99']:.1f} |

## Per-Class Distribution (top class)

| Class | Count | Pct | Mean conf |
|---|---|---|---|
"""
        for cls, n in sorted(s['class_distribution'].items(), key=lambda x: -x[1]):
            pct = 100 * n / max(1, s['n_successful'])
            conf = s['class_mean_confidence'].get(cls, 0)
            md += f"| {cls} | {n:,} | {pct:.1f}% | {conf:.3f} |\n"
        md += f"""

## Trigger Stats (>= 0.5 tuned threshold)

- Contracts with >=1 trigger: {s['n_with_any_trigger']:,} / {s['n_successful']:,} ({100*s['n_with_any_trigger']/max(1,s['n_successful']):.1f}%)
- Mean triggers per contract: {s['trigger_stats']['mean_per_contract']:.2f}
- p50: {s['trigger_stats']['p50']:.0f}, p95: {s['trigger_stats']['p95']:.0f}, max: {s['trigger_stats']['max']}

## Error Samples (first 10)

"""
        for e in s['error_samples']:
            md += f"- `{e['address']}`: {e['error'][:100]}...\n"
        if not s['error_samples']:
            md += "(none)\n"
        md += f"""

## Artifacts

- State: `{s['checkpoint_state_path']}`
- Log: `{s['log_path']}`
- Per-contract: `ml/reports/Run12_smartbugs_wild_FULL_<date>_per_contract.json`
- This summary: `ml/reports/Run12_smartbugs_wild_FULL_<date>_<time>_summary.md`

Generated: {s['completed_at']}
"""
        return md

    def run(self):
        self._log(f"=== SMARTBUGS WILD FULL EVALUATION — Run 12 ===")
        self._log(f"Started: {self.state['started_at']}")
        self._log(f"Checkpoint interval: {self.args.checkpoint} contracts")
        self._log(f"Log interval: {self.args.log_every} sec")
        self._log(f"Report interval: {self.args.report_every} contracts")
        self._log(f"Max contracts: {self.args.max or 'ALL'}")
        self._log(f"Total wild contracts: {self.state['config']['n_total_contracts']}")
        self._log(f"Already processed (resumed): {self.n_processed}")
        self._log(f"Log file: {self.log_path}")
        self._log(f"State file: {STATE_PATH}")

        # Load model
        self._log("\nInitialising Predictor (with warmup)...")
        t0 = time.time()
        predictor = Predictor(
            checkpoint="ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt",
            threshold=0.5,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self._log(f"  ready in {time.time()-t0:.2f}s")
        self._log(f"  architecture: {getattr(predictor, 'architecture', 'N/A')}")

        # Load tuned thresholds + temperatures
        thresholds = json.loads(Path("ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json").read_text())["thresholds"]
        temperatures = json.loads(Path("ml/calibration/temperatures_run12.json").read_text())

        def apply_temperature(probs):
            cal = {}
            for cls, p in probs.items():
                T = temperatures.get(cls, 1.0)
                if T == 1.0 or p in (0.0, 1.0):
                    cal[cls] = p; continue
                eps = 1e-7
                p = max(eps, min(1 - eps, p))
                logit = np.log(p / (1 - p))
                cal[cls] = 1.0 / (1.0 + np.exp(-logit / T))
            return cal

        # Enumerate all contracts (sorted, deterministic)
        all_files = sorted(CONTRACTS_DIR.glob("*.sol"))
        if self.args.max:
            all_files = all_files[:self.args.max]
        total = len(all_files)
        self._log(f"\nWill process {total} contracts (after max filter)")

        # Resume from last index
        start_idx = self.n_processed
        self._log(f"Starting from index {start_idx}/{total}")

        # Process loop
        last_checkpoint = start_idx
        last_report = start_idx
        try:
            for i in range(start_idx, total):
                if self.should_stop:
                    break

                fpath = all_files[i]
                addr = fpath.stem
                result = {"address": addr, "started_at": datetime.now().isoformat()}

                try:
                    t_start = time.time()
                    source = fpath.read_text()
                    res = predictor.predict_source(source, name=addr)
                    probs = apply_temperature(res["probabilities"])
                    t_elapsed = time.time() - t_start

                    # Top class
                    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
                    top_class = sorted_probs[0][0]
                    top_prob = sorted_probs[0][1]
                    top3 = sorted_probs[:3]

                    # Tuned predictions
                    tuned_pred = [c for c, p in probs.items() if p >= thresholds.get(c, 0.5)]
                    n_triggered = len(tuned_pred)

                    result.update({
                        "time_sec": t_elapsed,
                        "top_class": top_class,
                        "top_prob": top_prob,
                        "top_3": top3,
                        "n_triggered_tuned": n_triggered,
                        "n_triggered_tier": sum(1 for p in probs.values() if p >= predictor.TIER_SUSPICIOUS_THRESHOLD),
                        "all_probs": probs,
                        "completed_at": datetime.now().isoformat(),
                    })
                except Exception as e:
                    result.update({
                        "time_sec": time.time() - t_start if 't_start' in locals() else 0,
                        "error": str(e)[:500],
                        "completed_at": datetime.now().isoformat(),
                    })
                    self.n_errors += 1

                # Save to state
                self.state["processed_set"][addr] = result
                self.n_processed = len(self.state["processed_set"])
                self._maybe_log_progress()

                # Checkpoint
                if (self.n_processed - last_checkpoint) >= self.args.checkpoint:
                    self._save_state()
                    last_checkpoint = self.n_processed
                    self._log(f"CHECKPOINT: saved at {self.n_processed} contracts")

                # Incremental report
                if (self.n_processed - last_report) >= self.args.report_every:
                    self._save_incremental_report(self.n_processed)
                    last_report = self.n_processed

        except KeyboardInterrupt:
            self._log("\nKeyboardInterrupt — saving state and exiting")
            self._save_state()
            self.log_f.close()
            sys.exit(0)
        except Exception as e:
            self._log(f"\nUNEXPECTED ERROR: {e}")
            self._log(traceback.format_exc() if "traceback" in dir() else "")
            self._save_state()
            self.log_f.close()
            raise

        # Final report
        self._log("\n=== EVALUATION COMPLETE ===")
        self._log(f"Total processed: {self.n_processed}")
        self._log(f"Total errors: {self.n_errors}")
        self._log(f"Total time: {self._fmt_duration(time.time() - self.start_time + self.state['stats'].get('baseline_time', 0))}")
        summary_path, md_path = self._save_final_report()
        self._log(f"\nFinal summary: {summary_path}")
        self._log(f"Final markdown: {md_path}")
        self._save_state()
        self.log_f.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SmartBugs Wild full evaluation with resume support")
    parser.add_argument("--checkpoint", type=int, default=CHECKPOINT_DEFAULT,
                        help="Save state every N contracts (default 500)")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT,
                        help="Log progress every N seconds (default 30)")
    parser.add_argument("--report-every", type=int, default=REPORT_EVERY_DEFAULT,
                        help="Save incremental report every N contracts (default 2000)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max contracts to process (default: ALL)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state if available")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    evaluator = FullEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    import traceback
    main()
