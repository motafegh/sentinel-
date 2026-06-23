# Step 2 — AuditState

## What a TypedDict is

A plain Python `dict` has no fixed shape. A `TypedDict` is a blueprint that says
"this dict should have these keys, each holding this type of value" — for humans
and tooling, not enforced at runtime (it's still a normal dict underneath).
`AuditState` uses `total=False`: no key is required, any node can return a partial
dict with just what it computed.

## How the shared state actually fills up (LangGraph mechanic)

1. Every node = a function taking the current state, returning a small dict of
   only what it added/changed.
2. LangGraph **merges** that returned dict into the master state. Default merge =
   overwrite the key with the new value.
3. Exception: `routing_decisions` is `Annotated[list[str], operator.add]` — instead
   of overwriting, LangGraph **appends** to the existing list (`operator.add` =
   Python's `+`, and `list + list` concatenates). This exists because multiple
   nodes log routing decisions across one audit and you want the full history,
   not just the last node's contribution.

## The fields, grouped

| Category | Fields | Set by |
|---|---|---|
| Input (immutable) | `contract_code`, `contract_address` | caller |
| ML evidence | `ml_result`, `ml_hotspots` | `ml_assessment` |
| Tool evidence | `quick_screen_hits`, `static_findings`, `external_call_summary` | `quick_screen`, `static_analysis` |
| RAG evidence | `rag_results` | `rag_research` |
| History | `audit_history` | `audit_check` |
| Verdicts (original) | `verdicts`, `confirmations`, `contradictions` | `cross_validator` / `synthesizer` fallback |
| Verdicts (Phase A) | `consensus_verdict`, `confidence_by_class` | `consensus_engine` |
| Output | `final_report`, `narrative` | `synthesizer` |
| Output (Phase A) | `reflection_notes`, `metric_attribution`, `hotspot_visualization` | `reflection`, `explainer`, `visualizer` |
| Reserved, unused | `symbolic_findings`, `bytecode_analysis`, `taint_flows`, `permission_graph` | nobody yet — Phase B placeholders |

## The thing worth remembering

**Two parallel verdict systems coexist**: the original `verdicts` (debate or old
rule-based fallback) and the new `consensus_verdict` (consensus_engine). Both live
in state at once. This is the exact shape of the bug fixed today — `synthesizer`
only checked the old one and silently skipped the new one on debate failure.
Two systems writing to the same shared state, only one "official," is a footgun.
Comes back in Step 4 and Step 7.

→ You now know: state is a shared, append-only-in-one-spot dict that nodes build
up incrementally — and right now it has two competing verdict systems living
side by side.
