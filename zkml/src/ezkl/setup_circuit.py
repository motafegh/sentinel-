"""
setup_circuit.py — EZKL Pipeline Steps 1-5 (one-time setup)

RECALL — the full EZKL pipeline has two phases:
    ONE-TIME (this file):
        Step 1: gen_settings    → how to represent floats as integers
        Step 2: calibrate       → refine scale using real data
        Step 3: compile_circuit → ONNX → ZK circuit (R1CS constraints)
        Step 4: get_srs         → cryptographic foundation for keys
        Step 5: setup           → proving_key.pk + verification_key.vk

    PER AUDIT (run_proof.py):
        Step 6: gen_witness     → witness file from real input
        Step 7: prove           → proof π (~2KB)
        Step 8: verify          → true/false

RECALL — async behaviour in EZKL 23.x (learned through testing):
    gen_settings:       sync  → call directly
    calibrate_settings: sync  → call directly
    compile_circuit:    sync  → call directly
    get_srs:            async Rust future → needs asyncio.run() + await
    setup:              sync  → call directly

    get_srs wraps a Rust/tokio future. Python's asyncio.run() provides
    the event loop that tokio needs. Without it: "no running event loop".
    Without await: returns a pending Future instead of executing.
    This is NOT standard Python async — it's a PyO3 Rust binding quirk.

RECALL — what the outputs are used for:
    settings.json       → describes circuit numerics
    model.compiled      → the ZK circuit itself (R1CS constraints)
    srs.params          → cryptographic reference string (public, ~4MB)
    proving_key.pk      → how to construct proofs (PRIVATE — never commit)
    verification_key.vk → how to verify proofs (PUBLIC → Solidity)

Usage:
    cd ~/projects/sentinel
    poetry run python zkml/src/ezkl/setup_circuit.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import ezkl
from loguru import logger

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ONNX_MODEL       = "zkml/models/proxy.onnx"
CALIBRATION      = "zkml/ezkl/calibration.json"
SETTINGS         = "zkml/ezkl/settings.json"
COMPILED         = "zkml/ezkl/model.compiled"
SRS              = "zkml/ezkl/srs.params"
PROVING_KEY      = "zkml/ezkl/proving_key.pk"
VERIFICATION_KEY = "zkml/ezkl/verification_key.vk"


def check_prerequisites() -> None:
    """Verify all required input files exist before starting."""
    required = {
        "ONNX model":       ONNX_MODEL,
        "Calibration data": CALIBRATION,
    }
    for name, path in required.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Run these first:\n"
                f"  python zkml/src/distillation/export_onnx.py\n"
                f"  python zkml/src/distillation/generate_calibration.py"
            )
    logger.info("Prerequisites check passed")


async def _download_srs() -> None:
    """
    Download SRS inside an async context.

    RECALL — why this needs its own async function:
        get_srs returns a Rust/tokio future (PyO3 binding).
        It requires a running Python event loop to execute.
        asyncio.run() provides that loop.
        await unwraps the future and blocks until download completes.
        Without await: returns pending Future, download never happens.
        Without asyncio.run(): "no running event loop" RuntimeError.
    """
    await ezkl.get_srs(
        settings_path=SETTINGS,
        srs_path=SRS,
    )


def run_pipeline() -> None:
    """
    Execute EZKL Steps 1-5 sequentially.

    Sync functions called directly.
    get_srs called via asyncio.run() — the one async exception.
    """
    Path("zkml/ezkl").mkdir(parents=True, exist_ok=True)
    check_prerequisites()

    # ------------------------------------------------------------------
    # Step 1 — gen_settings
    # ------------------------------------------------------------------
    # RECALL — reads ONNX graph, produces initial scale factor.
    # Scale 13 = values multiplied by 2^13=8192 for integer arithmetic.
    # Our calibration showed range [0.0, 1.4613] — compact, efficient.
    logger.info("Step 1/5 — gen_settings")

    res = ezkl.gen_settings(
        model=ONNX_MODEL,
        output=SETTINGS,
    )
    # A-13 fix: `assert` is silently stripped by `python -O` (optimised mode).
    # Using an explicit RuntimeError ensures the failure is ALWAYS visible.
    if not res:
        raise RuntimeError(
            "gen_settings failed — check ONNX file is valid opset 11.\n"
            "Possible causes:\n"
            "  - ONNX file is missing or corrupt: check ONNX_MODEL path\n"
            "  - ONNX opset > 11: re-export with opset_version=11 in torch.onnx.export()\n"
            "  - EZKL version mismatch: check `ezkl --version`\n"
            f"  ONNX path: {ONNX_MODEL}"
        )
    logger.info(f"Settings generated: {SETTINGS}")

    # ------------------------------------------------------------------
    # Step 2 — calibrate_settings
    # ------------------------------------------------------------------
    # RECALL — runs real features through ONNX, observes actual value
    # ranges, refines scale factor. Prevents overflow in circuit.
    # target="resources" balances proof size vs proving time.
    logger.info("Step 2/5 — calibrate_settings")

    # calibrate_settings return value is not documented to be boolean in all
    # EZKL versions — check both falsy return and missing settings file.
    res = ezkl.calibrate_settings(
        data=CALIBRATION,
        model=ONNX_MODEL,
        settings=SETTINGS,
        target="resources",
    )
    if res is not None and not res:
        raise RuntimeError(
            "calibrate_settings failed — check calibration data and ONNX model.\n"
            "Possible causes:\n"
            "  - Calibration input file is empty or malformed\n"
            "  - ONNX model produces NaN/Inf for the calibration inputs\n"
            f"  Calibration file: {CALIBRATION}"
        )
    if not Path(SETTINGS).exists():
        raise RuntimeError(
            f"calibrate_settings completed but settings file is missing: {SETTINGS}\n"
            "This usually means calibration silently failed — check EZKL logs."
        )
    logger.info(f"Settings calibrated: {SETTINGS}")

    # ------------------------------------------------------------------
    # Step 3 — compile_circuit
    # ------------------------------------------------------------------
    # RECALL — converts ONNX ops → R1CS arithmetic constraints.
    # Weights separated as private witness — not in circuit structure.
    # This is why the circuit captures structure, not weights.
    # Why setup survives retraining: weights change, structure doesn't.
    logger.info("Step 3/5 — compile_circuit")

    res = ezkl.compile_circuit(
        model=ONNX_MODEL,
        compiled_circuit=COMPILED,
        settings_path=SETTINGS,
    )
    if not res:
        raise RuntimeError(
            "compile_circuit failed — check settings.json is calibrated.\n"
            "Possible causes:\n"
            "  - settings.json was not calibrated (run Step 2 first)\n"
            "  - settings.json is from a different ONNX model than ONNX_MODEL\n"
            "  - ONNX model uses an operation not supported by EZKL\n"
            f"  Settings path: {SETTINGS}"
        )

    compiled_size_kb = Path(COMPILED).stat().st_size / 1024
    logger.info(f"Circuit compiled: {COMPILED} ({compiled_size_kb:.1f} KB)")

    # ------------------------------------------------------------------
    # Step 4 — get_srs
    # ------------------------------------------------------------------
    # RECALL — downloads Structured Reference String (~4MB).
    # BN254 elliptic curve points — public cryptographic foundation.
    # Same SRS used by all EZKL users for the same circuit size.
    # Downloaded once, cached at srs_path, reused forever.
    # Needs internet: https://kzg.ezkl.xyz
    logger.info("Step 4/5 — get_srs (~4MB from kzg.ezkl.xyz)")

    asyncio.run(_download_srs())

    srs_size_mb = Path(SRS).stat().st_size / (1024 * 1024)
    logger.info(f"SRS ready: {SRS} ({srs_size_mb:.1f} MB)")

    # RECALL — BN254 SRS for our circuit size is ~4 MB.
    # A file outside 3.5–5.5 MB indicates corrupt or truncated download.
    SRS_MIN_MB, SRS_MAX_MB = 3.5, 5.5
    if not (SRS_MIN_MB <= srs_size_mb <= SRS_MAX_MB):
        raise ValueError(
            f"SRS size {srs_size_mb:.1f} MB is outside expected range "
            f"[{SRS_MIN_MB}, {SRS_MAX_MB}] MB — may be corrupt or truncated. "
            f"Delete {SRS} and rerun."
        )

    # ------------------------------------------------------------------
    # Step 5 — setup
    # ------------------------------------------------------------------
    # RECALL — derives proving + verification keys from circuit + SRS.
    #
    # proving_key.pk (PRIVATE, never commit to git):
    #   Encodes how to construct valid proofs for THIS circuit.
    #   Contains circuit structure + cryptographic trapdoor info.
    #
    # verification_key.vk (PUBLIC, embed in Solidity):
    #   Circuit fingerprint only — no weights, no trapdoor.
    #   Gets baked as constants into ZKMLVerifier.sol.
    #
    # RECALL — why keys survive proxy retraining (ADR-007):
    #   Keys derived from circuit STRUCTURE not weights.
    #   Retrain → weights change → circuit unchanged → keys valid.
    #   Resize architecture → circuit changes → must rerun setup.
    logger.info("Step 5/5 — setup")

    res = ezkl.setup(
        model=COMPILED,
        vk_path=VERIFICATION_KEY,
        pk_path=PROVING_KEY,
        srs_path=SRS,
    )
    if not res:
        raise RuntimeError(
            "setup failed — check compiled circuit and SRS are valid.\n"
            "Possible causes:\n"
            "  - SRS file is corrupt or truncated (check size: expect ~4MB)\n"
            "  - Compiled circuit was produced by a different EZKL version than the SRS\n"
            "  - Proving key or verification key path is not writable\n"
            f"  Compiled circuit: {COMPILED}\n"
            f"  SRS path:         {SRS}"
        )

    pk_size_mb = Path(PROVING_KEY).stat().st_size / (1024 * 1024)
    vk_size_kb = Path(VERIFICATION_KEY).stat().st_size / 1024
    compiled_size_kb = Path(COMPILED).stat().st_size / 1024
    srs_size_mb = Path(SRS).stat().st_size / (1024 * 1024)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EZKL setup complete — all artifacts generated")
    logger.info(f"  Settings:         {SETTINGS}")
    logger.info(f"  Compiled circuit: {COMPILED} ({compiled_size_kb:.1f} KB)")
    logger.info(f"  SRS:              {SRS} ({srs_size_mb:.1f} MB)")
    logger.info(f"  Proving key:      {PROVING_KEY} ({pk_size_mb:.1f} MB) PRIVATE")
    logger.info(f"  Verification key: {VERIFICATION_KEY} ({vk_size_kb:.1f} KB) PUBLIC")
    logger.info("Next: run_proof.py to generate and verify a proof")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()