# SENTINEL showcase

This is the shortest credible way to inspect SENTINEL without pretending that a fresh clone contains every historical model, DATA, analyzer, proving, RPC, or signing prerequisite.

The default showcase is deliberately **boundary-focused**. It runs with the Python standard library only and inspects committed source/configuration for four high-value claims:

1. the current audit orchestration is the 14-node LangGraph from `ml_assessment` to `visualizer`;
2. the live audit MCP exposes only its three version-aware read operations;
3. the retained ZKML proxy remains `128→64→32→10` with 138 public signals and the known `check_mode="UNSAFE"` limitation;
4. R4 Phase 8 is still `IN_PROGRESS`, R4-D-011 remains the accepted V10 V2.6 physical authority, confirmed negatives remain zero, and full repaired training is not authorized.

Anything that is not actually exercised is emitted as `NOT_RUN`. That is intentional: in SENTINEL, unavailable or unexecuted evidence is not equivalent to a clean result.

## Run it

From the repository root:

```bash
python3 tools/showcase_sentinel.py
```

Machine-readable form:

```bash
python3 tools/showcase_sentinel.py --json
```

No Poetry environment, GPU, model checkpoint, MCP service, RPC endpoint, analyzer binary, or proving key is required for this default path.

## Expected output

The exact formatting is stable enough for review, while paths/details may evolve with versioned project changes. A healthy run has this shape:

```text
SENTINEL fresh-clone boundary showcase
==========================================
[PASS] orchestration_topology: nodes=14 entry=ml_assessment exit=visualizer
[PASS] audit_mcp_surface: read_only=true tools=check_audit_exists,get_audit_history,get_latest_audit
[PASS_WITH_LIMITATION] zkml_proxy_boundary: proxy=128→64→32→10 public_signals=138 check_mode=UNSAFE
[PASS] r4_authority: phase8=IN_PROGRESS ... full_training_authorized=false confirmed_negatives=0

Explicitly not exercised:
[NOT_RUN] live_ml_inference: ...
[NOT_RUN] external_analyzers_and_formal_tools: ...
[NOT_RUN] live_langgraph_audit: ...
[NOT_RUN] zk_proof_generation: ...
[NOT_RUN] v3_signing_and_broadcast: ...

Overall: PASS
```

A non-zero process exit means one of the checked repository boundaries no longer matches the showcase contract and the showcase/doc layer needs review.

## What this demonstrates

The showcase is useful for a recruiter, reviewer, or engineer who wants a fast, reproducible answer to: **“Are the architecture and trust-boundary claims in this repository actually represented in current source/config?”**

It checks real committed implementation/configuration rather than replaying screenshots or hard-coded marketing output. In particular, the read-only MCP result is derived from the live handler source, the graph topology is derived from the real graph builder, and the proxy/check-mode values come from the retained ZKML source/settings.

## What this does not demonstrate

`Overall: PASS` does **not** mean:

- a Solidity contract was audited successfully;
- the historical Run12 model is present or accurate;
- Slither, Aderyn, Halmos, RAG, or other external tools ran;
- Phase-8 repaired training has occurred;
- model quality, false-positive rate, threshold quality, or calibration is established;
- an EZKL proof was generated in this run;
- the ZK circuit proves teacher/source/LangGraph execution or the final verdict;
- an on-chain V3 transaction was signed or broadcast;
- SENTINEL is production-ready.

Those boundaries are intentionally visible rather than hidden.

## Optional deeper orchestration smoke

After installing the AGENTS environment, the existing orchestration smoke runs the real LangGraph code while mocking external MCP responses:

```bash
cd agents
poetry install
AGENTS_DISABLE_LLM=1 poetry run python scripts/smoke_langgraph.py
```

That path is useful for inspecting graph execution and report assembly. Its mocked ML/RAG/audit payloads are fixtures, **not vulnerability-quality evidence**. Local analyzer/formal-verification availability may also affect degraded-status fields.

For a genuinely live audit harness, see `agents/scripts/run_real_audit.py` and the module/runtime prerequisites in [DEVELOPMENT.md](DEVELOPMENT.md). The full live path is intentionally not the fresh-clone portfolio default because large/local artifacts and external services are not guaranteed.

## Evidence philosophy shown by the demo

SENTINEL deliberately distinguishes:

```text
PASS                = the checked claim was established
PASS_WITH_LIMITATION = the checked claim is true with an explicit known limitation
NOT_RUN             = this showcase did not execute that capability
FAIL                = a checked invariant no longer matches repository truth
```

That distinction is part of the project architecture, not just presentation. See [Security and trust](docs/handbook/12_security_and_trust.md), [Runtime flows](docs/handbook/02_runtime_flows.md), and [Current status](docs/handbook/16_current_status.md) for the deeper boundaries.
