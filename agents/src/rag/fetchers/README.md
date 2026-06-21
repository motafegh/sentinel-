# Fetchers — RAG Data Sources

Strategy-pattern fetchers that supply documents to the RAG ingestion pipeline. Each fetcher implements `BaseFetcher` — the pipeline talks to the abstract interface, never to concrete implementations. Swap data sources without changing pipeline code.

## Files

| File | Purpose |
|------|---------|
| `base_fetcher.py` | Abstract `BaseFetcher` + `Document` dataclass |
| `github_fetcher.py` | `DeFiHackLabsFetcher` — .sol exploit PoC parser (726 docs) |
| `json_corpus_fetcher.py` | **(A.5, 2026-06-21)** Shared base for curated JSON-backed corpora |
| `code4rena_fetcher.py` | **(A.5)** Code4rena contest findings |
| `sherlock_fetcher.py` | **(A.5)** Sherlock contest findings (oracle/MEV-heavy) |
| `solodit_fetcher.py` | **(A.5)** Solodit aggregated findings |
| `immunefi_fetcher.py` | **(A.5)** Immunefi bug-bounty disclosures |
| `swc_registry_fetcher.py` | **(A.5)** SWC weakness-classification registry (static reference) |

## `base_fetcher.py` — BaseFetcher

### Document Dataclass

```python
@dataclass
class Document:
    content:   str    # raw text to embed
    source:    str    # where it came from
    doc_id:    str    # unique identifier for dedup
    metadata:  dict   # protocol, date, vuln_type, severity, loss_usd, chain, url
```

Every piece of knowledge in SENTINEL's RAG system is a Document. The `metadata` dict enables filtered retrieval: "find reentrancy exploits from 2023 only."

### BaseFetcher Interface

```python
class BaseFetcher(ABC):
    def __init__(self, data_dir: Path): ...
    def fetch(self) -> list[Document]: ...           # all documents (full rebuild)
    def fetch_since(self, since: datetime) -> list[Document]: ...  # incremental
    def source_name(self) -> str: ...                 # human-readable name
    def health_check(self) -> bool: ...               # verify source reachable
```

Fetchers are stateless workers. State (`last_run`, `seen_hashes`) lives in the pipeline, not in individual fetchers.

## `github_fetcher.py` — DeFiHackLabsFetcher

Parses Solidity PoC files from the DeFiHackLabs repository. Each `.sol` file becomes one Document.

### Comment Formats

| Format | Marker | Files | Content |
|--------|--------|-------|---------|
| A (`@Summary`) | `// @Summary` | ~25 | Step-by-step attack narrative |
| B (`@KeyInfo`) | `// @KeyInfo` | ~473 | Loss + addresses + `@Analysis` URLs |
| C (free-form) | — | ~159 | Plain text / bare URLs (older files) |

### Extraction Pipeline per `.sol` File

```
1. _extract_date()              → "2023-03-01" (from directory name)
2. _extract_summary_block()     → attack narrative (Format A)
3. _extract_keyinfo_block()     → loss + addresses (Format B)
4. _extract_all_analysis_urls() → post-mortem links
5. _extract_first(PATTERN_TX)   → transaction URL
6. _extract_root_cause()        → root cause string
7. _extract_loss()              → $197M → 197000000
8. _infer_vuln_type()           → "reentrancy", "flash_loan", etc.
```

### Vulnerability Type Inference

`_infer_vuln_type()` pattern-matches on `root_cause` + `summary_block` + `keyinfo` (not raw file content — the first 1000 chars of any .sol file are always SPDX/pragma/imports). Returns one of: `reentrancy`, `flash_loan`, `oracle_manipulation`, `access_control`, `integer_overflow`, `front_running`, `logic_error`, `timestamp_dependence`, `delegatecall`, `denial_of_service`, `other`.

### Directory Scanning (FIX-20)

`fetch()` scans both `src/test/` (main corpus) AND `past/` (archived exploits). Old code only scanned `src/test/` — `past_path` was declared but unused.

### fetch_since (FIX-21)

Undated files are always included in `fetch_since()` results. Old code silently dropped docs with no YYYY-MM directory, systematically excluding historically significant exploits from incremental updates.

### Usage

```python
from src.rag.fetchers.github_fetcher import DeFiHackLabsFetcher

fetcher = DeFiHackLabsFetcher(
    repo_path=Path("data/defihacklabs"),
    data_dir=Path("data/exploits"),
)
docs = fetcher.fetch()           # 726 documents
recent = fetcher.fetch_since(datetime(2024, 1, 1))  # incremental
```

## `json_corpus_fetcher.py` — JsonCorpusFetcher (A.5, 2026-06-21)

Shared base for the 5 corpus-expansion fetchers. Each reads a curated JSON corpus
from `data/knowledge/<corpus_key>.json` — a list of `{title, content, vuln_type,
severity, protocol, date, url, chain, loss_usd}` records — and converts each record to
a `Document`. Design rationale: **deterministic, offline, unit-testable** — no network
flakiness in CI. Production scale-up means replacing the JSON file with a full export;
the fetcher contract (and all calling code) does not change.

```python
class Code4renaFetcher(JsonCorpusFetcher):
    corpus_key = "code4rena"        # → data/knowledge/code4rena.json
    _source_name = "code4rena"
```

| Concrete fetcher | Corpus file | Seed doc count | Focus |
|---|---|---|---|
| `Code4renaFetcher` | `code4rena.json` | 5 | Contest-graded High/Medium findings |
| `SherlockFetcher` | `sherlock.json` | 4 | Oracle manipulation, MEV, state bugs |
| `SoloditFetcher` | `solodit.json` | 5 | Cross-firm aggregated findings |
| `ImmunefiFetcher` | `immunefi.json` | 3 | Paid bounty post-mortems (incl. `loss_usd`) |
| `SWCRegistryFetcher` | `swc_registry.json` | 7 | Canonical SWC-1xx weakness definitions |

Missing corpus file → `health_check()` returns `False`, `fetch()` returns `[]`
(degrades gracefully; `build_index.py` continues with DeFiHackLabs + whichever
corpora ARE present). Malformed JSON → logged warning, `[]`, never raises.

Wired into `src/rag/build_index.py:_collect_extra_documents()` — runs after the
DeFiHackLabs fetch and adds its documents to the same chunk/embed/FAISS/BM25 pipeline.
Live rebuild (2026-06-21): 750 total documents / 776 chunks (726 DeFiHackLabs + 24 from
the 5 new corpora).
