"""PDG (Program Dependence Graph) builder — DEFERRED to v3.1.

See AUDIT_PATCHES 2-P9 and Stage 2 plan §D-2.4 for the deferral rationale:
  - The lightweight PDG described in the original Stage 2 plan is a v3+ feature.
  - Time spent on PDG in v2 is better allocated to Stage 4 (label verification).
  - Schema additions for a standalone PDG would require a v2.1 schema change.
  - Run 11 does not need PDG features.

This file is a placeholder so the directory structure is ready for v3.1.
The v3.1 work is a drop-in: implement ``build_pdg`` here and wire it to the
CLI's ``--emit-pdg`` flag (already reserved in ``cli.py``).

Planned public API (v3.1):
    build_pdg(sol_path, config, sha256, source) -> PdgArtifact

A PDG combines:
  - Data-dependence edges (def → use of the same variable)
  - Control-dependence edges (control predicate → dominated statements)
both intra- and inter-function.  The inter-function case uses the ICFG-Lite
already built by ``graph_extractor.py`` as the skeleton.
"""


def build_pdg(*_args, **_kwargs):
    """Build a Program Dependence Graph for a Solidity contract.

    Raises:
        NotImplementedError: Always — PDG builder is deferred to v3.1.
    """
    raise NotImplementedError(
        "PDG builder is DEFERRED to v3.1.  "
        "See docs/proposal/Data_Module_Proposals/actionable_plans/03_stage_2_representation.md §D-2.4 "
        "and AUDIT_PATCHES.md §2-P9."
    )
