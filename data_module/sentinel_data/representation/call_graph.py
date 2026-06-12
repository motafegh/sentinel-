"""Call-graph builder — DEFERRED to v3.1.

See AUDIT_PATCHES 2-P9 and Stage 2 plan §D-2.4 for the deferral rationale:
  - Call-graph extraction is a v3+ feature; Run 11 does not need it.
  - The Stage 2 ICFG-Lite (CALL_ENTRY/RETURN_TO edges in the graph) already
    encodes the most important cross-function topology for training.
  - A standalone call-graph artifact is useful for Stage 6 (complexity proxy,
    contract interaction analysis) but not for the Run 11 training graph.

This file is a placeholder so the directory structure is ready for v3.1.
The v3.1 work is a drop-in: implement ``build_call_graph`` here and wire it
to the CLI's ``--emit-callgraph`` flag (already reserved in ``cli.py``).

Planned public API (v3.1):
    build_call_graph(sol_path, config, sha256, source) -> CallGraphArtifact

A CallGraph captures:
  - Intra-contract calls (function → function within the same contract)
  - Cross-contract calls (high-level calls to typed interfaces)
  - Low-level calls (raw .call() / .delegatecall() / .staticcall())
  - Library calls (``using X for Y`` resolved to the library implementation)
"""


def build_call_graph(*_args, **_kwargs):
    """Build a call-graph artifact for a Solidity contract.

    Raises:
        NotImplementedError: Always — call-graph builder is deferred to v3.1.
    """
    raise NotImplementedError(
        "Call-graph builder is DEFERRED to v3.1.  "
        "See docs/proposal/Data_Module_Proposals/actionable_plans/03_stage_2_representation.md §D-2.4 "
        "and AUDIT_PATCHES.md §2-P9."
    )
