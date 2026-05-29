# Preprocessing — Chunk 6: The Batch Pipeline (ast_extractor.py)

> **File:** `ml/src/data_extraction/ast_extractor.py`
> **What you'll learn:** Multiprocessing with `mp.Pool`, checkpoint/resume patterns, solc binary management, content-addressed file naming, and how to process 40K+ files at scale.
> **Time:** ~25 minutes
> **Interview relevance:** MLOps (data pipelines), ML (feature extraction at scale)

---

## 1. The Thin Wrapper Pattern

The file header says it best:
```
"This file is the orchestration layer only. Graph construction logic has been
extracted to graph_schema.py and graph_extractor.py."
```

`ast_extractor.py` has exactly two responsibilities:
1. **Orchestration**: read CSV, spawn workers, track progress, write files
2. **Offline-only concerns**: solc binary resolution, multiprocessing, checkpoint/resume

It does NOT duplicate any feature engineering code. That's all in `graph_extractor.py`.

This is the **Single Responsibility Principle** applied to a data pipeline: orchestration ≠ feature engineering.

---

## 2. The `ASTExtractorV4` Class

```python
class ASTExtractorV4:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = get_project_root()
    
    def contract_to_pyg(self, contract_path, solc_binary=None, solc_version=None, label=0) -> Data | None:
        ...
    
    def extract_batch_with_checkpoint(self, df, n_workers=11, chunksize=50, ...) -> List[Data]:
        ...
```

**`contract_to_pyg`** is the single-contract worker function. It:
1. Builds a `GraphExtractionConfig` with solc settings
2. Calls `extract_contract_graph()`
3. Attaches offline-specific metadata: `contract_hash`, `contract_path`, `y`
4. Returns `None` on any `GraphExtractionError` (batch skip policy)

Note the **two types of metadata**: the shared `extract_contract_graph()` returns the graph + `contract_name`. The batch caller attaches `contract_hash` and `y` (label). This is separation of concerns — the shared function doesn't know about labels or hash-based filenames.

---

## 3. Multiprocessing with `mp.Pool`

```python
with mp.Pool(processes=n_workers) as pool:
    for result in pool.imap(worker, contract_paths, chunksize=chunksize):
        if result is None:
            continue
        # Process result...
```

**`multiprocessing.Pool`** creates a pool of worker processes. Each worker runs `contract_to_pyg()` independently.

**Why multiprocessing instead of threading?**
Python's **GIL (Global Interpreter Lock)** prevents true parallel execution of CPU-bound Python code in threads. Multiprocessing bypasses the GIL by creating separate processes with separate memory spaces.

Feature extraction (Slither parsing + AST walking) is **CPU-bound**. Threading would give no speedup. Multiprocessing gives near-linear speedup up to CPU count.

**`pool.imap` vs `pool.map`:**
- `pool.map(fn, items)` — blocks until ALL results are ready, then returns list
- `pool.imap(fn, items)` — returns an **iterator** that yields results as they complete

For 40K files, `map` would hold all results in memory simultaneously. `imap` processes them one by one, writing each to disk immediately — constant memory usage regardless of dataset size.

**`chunksize=50`:** Instead of sending items one by one (high IPC overhead), send batches of 50. Each worker process gets 50 contracts to process before checking in. Reduces process-switching overhead.

**`partial(self.contract_to_pyg, ...)`:**
`mp.Pool.imap` takes a callable with ONE argument (the item from the iterable). But `contract_to_pyg` takes `(path, solc_binary, solc_version, label)`. `functools.partial` pre-fills the fixed arguments:
```python
worker = partial(self.contract_to_pyg, solc_binary=solc_bin, solc_version=version, label=0)
# Now worker(path) works
```

---

## 4. The Checkpoint/Resume Pattern — MLOps Essential

```python
checkpoint_file = output_dir / "checkpoint.json"

# Load checkpoint
processed_hashes: set = set()
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
        processed_hashes = set(checkpoint.get("processed", []))
    print(f"Found {len(processed_hashes):,} already processed contracts")

# Filter already-processed
if processed_hashes:
    df["_temp_hash"] = df["contract_path"].apply(get_contract_hash)
    df = df[~df["_temp_hash"].isin(processed_hashes)]
```

**Why checkpoints?**

Processing 41,576 contracts takes hours. If the process is interrupted (power outage, OOM kill, SIGTERM from job scheduler), you don't want to restart from scratch.

The checkpoint JSON:
```json
{
    "processed": ["abc123", "def456", ...],
    "total": 15000,
    "timestamp": "2026-05-18T14:23:45",
    "completed": false
}
```

On restart: load the checkpoint → skip hashes already in `processed` → resume from where you left off.

**Write checkpoint every N contracts:**
```python
if total_processed % checkpoint_every == 0:
    with open(checkpoint_file, "w") as f:
        json.dump({
            "processed": list(processed_hashes),
            "total": total_processed,
            "timestamp": datetime.now().isoformat(),
            "completed": False,
        }, f, indent=2)
```

`checkpoint_every=500` means at most 500 contracts are re-processed if the job fails. Smaller = more frequent disk writes = slower but safer.

> 🎯 **INTERVIEW FOCUS:** "How do you make a long-running data processing job resumable?" — Checkpoint file with set of processed item IDs. On restart, filter out already-processed items. Write checkpoint periodically.

---

## 5. Content-Addressed File Naming

```python
from src.utils.hash_utils import get_contract_hash, get_filename_from_hash

contract_hash = get_contract_hash(contract_path)
filename = get_filename_from_hash(contract_hash)   # e.g., "abc123def456.pt"
graph_file = output_dir / filename
torch.save(result, graph_file)
```

**Content-addressed storage**: The filename is derived from the content, not from an arbitrary sequential ID.

**Benefits:**
1. **Idempotent writes**: same contract → same hash → same filename. Re-running extraction overwrites the exact same file. No stale files from old runs.
2. **Duplicate detection**: two contracts with identical content have the same hash → one file (deduplication).
3. **Cache key**: the tokenizer uses the same hash to name token files. Matching by hash finds the corresponding graph for a token file.

**The `processed_hashes` set stores these hashes** — so the checkpoint check is: "have we already written `{hash}.pt`?" A `set` is O(1) for membership check vs O(N) for a list.

---

## 6. Solc Binary Management

```python
def get_solc_binary(version: str) -> Optional[str]:
    venv_path = Path.cwd() / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
    candidates = [
        venv_path / f"solc-{version}",
        venv_path / "solc",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None
```

**Why version-pinned solc binaries?**

Solidity has had breaking changes between versions. A contract written for `solc 0.4.18` won't compile with `solc 0.8.19`. The BCCC dataset contains contracts spanning versions 0.4.1 to 0.8.x.

`solc-select` is a tool that installs multiple solc versions side by side. `ast_extractor.py` resolves the right binary for each version group by looking for the binary in the `.solc-select` artifacts directory.

```python
def solc_supports_allow_paths(version: str) -> bool:
    major, minor, _ = parse_solc_version(version)
    return (major, minor) >= (0, 5)
```

The `--allow-paths` flag (used for import resolution) was introduced in solc 0.5.0. Passing it to solc 0.4.x would cause a startup error. This guard prevents that.

**Version grouping:**
```python
groups = df.groupby("detected_version")
for version, group in tqdm(groups, desc="Version groups"):
    solc_bin = group.iloc[0]["solc_binary"]
    worker = partial(self.contract_to_pyg, solc_binary=solc_bin, solc_version=version, label=0)
```

Group by version → resolve binary once per version → run all contracts in that group with the same binary. This is efficient: one binary resolution, not one per contract.

---

## 7. The Error Policy Hierarchy

```python
def contract_to_pyg(self, contract_path, ...):
    try:
        graph = extract_contract_graph(Path(contract_path), config)
    except RuntimeError:
        raise                  # Slither not installed — ABORT EVERYTHING
    except GraphExtractionError as exc:
        if self.verbose:
            print(f"Skipped {Path(contract_path).name}: {exc}")
        return None            # Expected failure — skip this contract
    except Exception as exc:
        if self.verbose:
            print(f"Unexpected error: {exc}")
        return None            # Unexpected failure — skip but log
    ...
```

Three tiers:
1. **Fatal** (`RuntimeError`): re-raise immediately. If Slither isn't installed, every contract will fail — no point continuing.
2. **Expected** (`GraphExtractionError`): skip. Compilation failures and Slither parse errors are expected for some contracts in a large dataset.
3. **Unexpected** (any other `Exception`): skip but log. A bug in one contract shouldn't crash the whole batch run.

> 🎯 **INTERVIEW FOCUS:** "How do you design fault tolerance in a batch ML data processing pipeline?" — Three-tier error handling: fatal errors re-raise (infrastructure issues), expected errors skip (data quality issues), unexpected errors log+skip (defensive against unknown bugs).

---

## 8. The CLI Entry Point

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--input",  default="ml/data/processed/contracts_metadata.parquet")
    parser.add_argument("--output", default="ml/data/graphs")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()
    
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive")
```

`if __name__ == "__main__"` — this block only runs when the file is executed directly (`python ast_extractor.py`), not when imported. This is the standard Python CLI pattern.

`mp.cpu_count() - 1` — leave one core for the OS. Using all cores can cause system unresponsiveness.

**Mutually exclusive flags guard:**
`--force` (restart from scratch) and `--resume` (continue from checkpoint) are mutually exclusive. `parser.error()` prints a usage message and exits with code 2.

**The `--force` behavior:**
```python
if args.force:
    checkpoint_file.unlink()  # delete checkpoint
    # extraction starts fresh
```

---

## 9. The Full Processing Architecture

```
contracts_metadata.parquet
         ↓ pd.read_parquet()
     DataFrame [contract_path, detected_version, ...]
         ↓ groupby("detected_version")
  Version "0.4.26": 12,000 contracts
  Version "0.5.17":  3,000 contracts
  ...
         ↓ mp.Pool(11 workers)
  Worker 1: contract_a.sol → graph_abc123.pt
  Worker 2: contract_b.sol → graph_def456.pt
  ...
  (11 workers in parallel)
         ↓
  graph_*.pt files in ml/data/graphs/
         ↓ checkpoint.json updated every 500
```

**Memory-efficient streaming:** `pool.imap` + immediate `torch.save` means only 11 graphs (one per worker) are in memory at any time. 41,576 graphs × ~50KB each = ~2GB if held in memory simultaneously — streaming avoids this.

---

## 10. Summary — MLOps Patterns in This File

| Pattern | Implementation | Use Case |
|---------|---------------|----------|
| Checkpoint/resume | JSON with processed hash set | Long-running batch jobs |
| Content-addressed storage | MD5 hash → filename | Idempotent writes, deduplication |
| Tiered error handling | RuntimeError → raise; GraphError → skip | Fault-tolerant batch processing |
| Version-pinned dependencies | solc-select binary resolution | Reproducible builds across Solidity versions |
| Streaming with imap | `pool.imap` + immediate writes | Constant memory for large datasets |
| Thin wrapper | Orchestration only, logic in imported modules | Separation of concerns |
| Progress monitoring | tqdm + failure rate logging | Operational visibility |

---

## Interview Questions

1. **"How would you process 40,000 files in parallel in Python?"**
   → `multiprocessing.Pool` to bypass the GIL for CPU-bound work. Use `pool.imap` (not `pool.map`) to stream results and write to disk immediately rather than holding all results in memory. Use `functools.partial` to pre-fill fixed arguments.

2. **"How do you make a data processing job resumable after interruption?"**
   → Checkpoint file (JSON or SQLite) tracking completed item IDs (content-addressed hashes, not sequential indices). Filter out completed items on restart. Write checkpoints periodically (every N items), not after every item.

3. **"Why use different solc compiler versions instead of always using the latest?"**
   → Smart contracts embed their intended compiler version in the `pragma` directive. Compiling with a different version can fail (breaking changes) or produce different ASTs (different feature extraction). Version pinning ensures the AST produced by Slither matches what the contract author intended.

4. **"A worker in your multiprocessing pool crashes. What happens to your job?"**
   → With `pool.imap`, a crashed worker is detected when the main process tries to read its result — Python raises an exception. The checkpoint pattern means only the current batch (up to 500 contracts) needs to be reprocessed. With the tiered error handling, unexpected errors return `None` (logged, not crashed) rather than killing the worker.

---

## Cross-Module Summary: The Full Preprocessing Stack

You've now learned the complete preprocessing stack:

```
graph_schema.py          → Constants, types, schema version
       ↓ imported by
graph_extractor.py       → Feature engineering, graph building
       ↓ imported by
ast_extractor.py         → Batch orchestration (this chunk)
preprocess.py            → Online inference (Module 6)
```

**The key invariant:** One function (`extract_contract_graph`), one schema (`v8`), two callers. Train/inference skew is structurally impossible.

---

**Preprocessing module complete!** ✅

Check your understanding before moving on:
- Can you name all 13 node types and why CFG subtypes exist?
- Can you explain why `external_call_count` includes Transfer/Send?
- Can you explain the `most_derived` heuristic and when it fails?
- Can you explain why `return_ignored` uses a sequential scan (IMP-D1)?
- Can you describe the checkpoint/resume pattern for batch processing?

**Next module:** `DataExtraction/01_tokenizer_windowed_approach.md`
