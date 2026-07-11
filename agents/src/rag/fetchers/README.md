# Fetchers — RAG Data Sources

Strategy-pattern fetchers that supply documents to the RAG ingestion pipeline. Each
fetcher implements `BaseFetcher` — the pipeline talks to the abstract interface, never
to concrete implementations. Swap data sources without changing pipeline code.

> **WS2 (2026-06-22):** Only `DeFiHackLabsFetcher` is active in `build_index/`.
> The 5 Phase A.5 corpus fetchers below are **disabled** (`_extra_fetchers()` returns
> `[]`) — their seed corpora were synthetic hand-written placeholders, and
> `SoloditFetcher` directly caused a hallucinated verdict.
> Re-enable with real data per `02_RAG_BUILD_PLAN.md`.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `base_fetcher.py` | Abstract `BaseFetcher` + `Document` dataclass | Active |
| `github_fetcher.py` | `DeFiHackLabsFetcher` — .sol exploit PoC parser (726 docs) | Active |
| `json_corpus_fetcher.py` | **(A.5)** Shared base for curated JSON-backed corpora | Disabled |
| `code4rena_fetcher.py` | **(A.5)** Code4rena contest findings | Disabled |
| `sherlock_fetcher.py` | **(A.5)** Sherlock contest findings | Disabled |
| `solodit_fetcher.py` | **(A.5)** Solodit aggregated findings | Disabled |
| `immunefi_fetcher.py` | **(A.5)** Immunefi bug-bounty disclosures | Disabled |
| `swc_registry_fetcher.py` | **(A.5)** SWC weakness-classification registry | Disabled |

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

### BaseFetcher Interface

```python
class BaseFetcher(ABC):
    def fetch(self) -> list[Document]: ...           # all documents (full rebuild)
    def fetch_since(self, since: datetime) -> list[Document]: ...  # incremental
    def source_name(self) -> str: ...
    def health_check(self) -> bool: ...
```

Fetchers are stateless. State (`last_run`, `seen_hashes`) lives in the pipeline.

## `github_fetcher.py` — DeFiHackLabsFetcher

Parses Solidity PoC files from the DeFiHackLabs repository.

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

`_infer_vuln_type()` pattern-matches on `root_cause` + `summary_block` + `keyinfo`
(not raw file content — the first 1000 chars of any .sol file are always SPDX/pragma).

Returns one of: `reentrancy`, `flash_loan`, `oracle_manipulation`, `access_control`,
`integer_overflow`, `front_running`, `logic_error`, `timestamp_dependence`,
`delegatecall`, `denial_of_service`, `other`.

### Key Fixes

- **FIX-20:** `fetch()` now scans both `src/test/` AND `past/` — `past_path` was
  declared but unused in older code.
- **FIX-21:** Undated files are always included in `fetch_since()` — old code silently
  dropped docs with no YYYY-MM directory.
- **FIX-22b:** `_infer_vuln_type()` does not slice the raw `.sol` file (first 1000 chars
  are always SPDX/pragma/imports); matches only against extracted metadata fields.

## `json_corpus_fetcher.py` — JsonCorpusFetcher (A.5, **DISABLED**)

Shared base for the 5 corpus-expansion fetchers. Each reads a curated JSON corpus from
`data/knowledge/<corpus_key>.json`. Design is deterministic, offline, and unit-testable
— no network flakiness in CI.

| Concrete fetcher | Corpus file | Focus |
|---|---|---|
| `Code4renaFetcher` | `code4rena.json` | Contest-graded High/Medium findings |
| `SherlockFetcher` | `sherlock.json` | Oracle manipulation, MEV, state bugs |
| `SoloditFetcher` | `solodit.json` | Cross-firm aggregated findings |
| `ImmunefiFetcher` | `immunefi.json` | Paid bounty post-mortems |
| `SWCRegistryFetcher` | `swc_registry.json` | Canonical SWC-1xx weakness definitions |

Missing corpus file → `health_check()` returns `False`, `fetch()` returns `[]`
(degrades gracefully). Malformed JSON → logged warning, `[]`, never raises.

Wired into `src/rag/build_index/_pipeline.py:_collect_extra_documents()` — currently
disabled via `_extra_fetchers()` returning `[]`.
