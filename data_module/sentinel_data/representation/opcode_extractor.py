"""EVM opcode extractor — DEFERRED to v3.1.

See AUDIT_PATCHES 2-P9 and Stage 2 plan §D-2.4 for the deferral rationale:
  - Opcode-level features require compiled bytecode (solc ``--bin`` output),
    adding a compilation step that is separate from Slither parsing.
  - The payoff for Run 11 is unclear: the GNN already captures structural
    vulnerability patterns at the AST/CFG level; opcode adds a second,
    potentially overlapping signal channel.
  - Opcode feature engineering (which opcodes to count, how to normalise)
    needs empirical validation — a v2.1 experiment, not a v2 assumption.

This file is a placeholder so the directory structure is ready for v3.1.
The v3.1 work is a drop-in: implement ``extract_opcodes`` here and wire it
to the CLI's ``--emit-opcode`` flag (already reserved in ``cli.py``).

Planned public API (v3.1):
    extract_opcodes(sol_path, config, sha256, source) -> OpcodeArtifact

An OpcodeArtifact captures:
  - Per-function opcode frequency histograms (CALL, SLOAD, SSTORE, etc.)
  - Dangerous opcode markers (DELEGATECALL, SELFDESTRUCT, CALLVALUE)
  - Bytecode length (proxy for contract complexity)
"""


def extract_opcodes(*_args, **_kwargs):
    raise NotImplementedError(
        "Opcode extractor is DEFERRED to v3.1.  "
        "See docs/proposal/Data_Module_Proposals/actionable_plans/03_stage_2_representation.md §D-2.4 "
        "and AUDIT_PATCHES.md §2-P9."
    )
