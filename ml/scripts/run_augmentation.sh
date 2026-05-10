#!/usr/bin/env bash
# run_augmentation.sh — Phase 3 Data Augmentation Pipeline
#
# Runs the full augmentation pipeline for SENTINEL v5:
#   1. Generate safe variants from BCCC Reentrancy (CEI swap, target 500+)
#   2. Generate safe variants from BCCC MishandledException (bare call wrap)
#   3. Generate safe variants from BCCC DenialOfService (loop guard)
#   4. Extract graphs + tokens for all augmented contracts
#   5. Update train split (freeze val/test)
#   6. Validate augmented graphs
#
# Run from the project root:
#   export TRANSFORMERS_OFFLINE=1
#   bash ml/scripts/run_augmentation.sh
#
# To smoke-test the pipeline first (5 contracts, no writes):
#   DRY_RUN=1 MAX_CONTRACTS=5 bash ml/scripts/run_augmentation.sh

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────

BCCC_ROOT="BCCC-SCsVul-2024/SourceCodes"
AUG_ROOT="ml/data/augmented"
GRAPHS_DIR="ml/data/graphs"
TOKENS_DIR="ml/data/tokens"
MULTILABEL_CSV="ml/data/processed/multilabel_index.csv"
SPLITS_DIR="ml/data/splits"

DRY_RUN="${DRY_RUN:-0}"
MAX_CONTRACTS="${MAX_CONTRACTS:-}"   # empty = no limit

# Build CLI flags
DRY_FLAG=""
MAX_FLAG=""
[ "${DRY_RUN}" = "1" ] && DRY_FLAG="--dry-run"
[ -n "${MAX_CONTRACTS}" ] && MAX_FLAG="--max-contracts ${MAX_CONTRACTS}"

echo "=== SENTINEL v5 Data Augmentation Pipeline ==="
echo "BCCC root   : ${BCCC_ROOT}"
echo "Output root : ${AUG_ROOT}"
echo "Dry run     : ${DRY_RUN}"
echo "Max contracts: ${MAX_CONTRACTS:-unlimited}"
echo ""

# ── Step 1: Generate safe variants — Reentrancy → CEI ───────────────────────

echo "--- Step 1: Reentrancy → CEI-safe variants ---"
poetry run python ml/scripts/generate_safe_variants.py \
    --input-dir "${BCCC_ROOT}/Reentrancy" \
    --output-dir "${AUG_ROOT}/reentrancy_safe" \
    --strategy reentrancy-cei \
    ${MAX_FLAG} ${DRY_FLAG}

# ── Step 2: Generate safe variants — MishandledException → wrapped call ─────

echo ""
echo "--- Step 2: MishandledException → return-value-checked variants ---"
poetry run python ml/scripts/generate_safe_variants.py \
    --input-dir "${BCCC_ROOT}/MishandledException" \
    --output-dir "${AUG_ROOT}/mishandled_safe" \
    --strategy mishandled-exception \
    ${MAX_FLAG} ${DRY_FLAG}

# ── Step 3: Generate safe variants — DoS → bounded loop ─────────────────────

echo ""
echo "--- Step 3: DenialOfService → bounded-loop variants ---"
poetry run python ml/scripts/generate_safe_variants.py \
    --input-dir "${BCCC_ROOT}/DenialOfService" \
    --output-dir "${AUG_ROOT}/dos_safe" \
    --strategy dos-bounded \
    ${MAX_FLAG} ${DRY_FLAG}

# ── Step 4: Extract graphs + tokens, update multilabel_index.csv ─────────────

echo ""
echo "--- Step 4a: Extract reentrancy_safe (label: NonVulnerable) ---"
poetry run python ml/scripts/extract_augmented.py \
    --input-dir "${AUG_ROOT}/reentrancy_safe" \
    --graphs-dir "${GRAPHS_DIR}" \
    --tokens-dir "${TOKENS_DIR}" \
    --multilabel-csv "${MULTILABEL_CSV}" \
    ${MAX_FLAG} ${DRY_FLAG}

echo ""
echo "--- Step 4b: Extract mishandled_safe (label: NonVulnerable) ---"
poetry run python ml/scripts/extract_augmented.py \
    --input-dir "${AUG_ROOT}/mishandled_safe" \
    --graphs-dir "${GRAPHS_DIR}" \
    --tokens-dir "${TOKENS_DIR}" \
    --multilabel-csv "${MULTILABEL_CSV}" \
    ${MAX_FLAG} ${DRY_FLAG}

echo ""
echo "--- Step 4c: Extract dos_safe (label: NonVulnerable) ---"
poetry run python ml/scripts/extract_augmented.py \
    --input-dir "${AUG_ROOT}/dos_safe" \
    --graphs-dir "${GRAPHS_DIR}" \
    --tokens-dir "${TOKENS_DIR}" \
    --multilabel-csv "${MULTILABEL_CSV}" \
    ${MAX_FLAG} ${DRY_FLAG}

# ── Step 5: Update train split (freeze val/test) ─────────────────────────────

if [ "${DRY_RUN}" != "1" ]; then
    echo ""
    echo "--- Step 5: Update train split (--freeze-val-test) ---"
    poetry run python ml/scripts/create_splits.py \
        --multilabel-index "${MULTILABEL_CSV}" \
        --splits-dir "${SPLITS_DIR}" \
        --freeze-val-test
else
    echo ""
    echo "--- Step 5: Skipped (dry-run) ---"
fi

# ── Step 6: Validate augmented graphs ────────────────────────────────────────

if [ "${DRY_RUN}" != "1" ]; then
    echo ""
    echo "--- Step 6: Validate augmented graphs (v5 schema checks) ---"
    # --check-contains-edges and --check-control-flow only apply after v5 full
    # re-extraction. For augmented-only validation, just check dim + edge types.
    poetry run python ml/scripts/validate_graph_dataset.py \
        --graphs-dir "${GRAPHS_DIR}" \
        --check-dim 12 \
        --check-edge-types 7
else
    echo ""
    echo "--- Step 6: Skipped (dry-run) ---"
fi

echo ""
echo "=== Augmentation pipeline complete ==="
echo "Next steps:"
echo "  Phase 4: Re-extract all 68K graphs with v5 schema (ast_extractor.py --force)"
echo "  Phase 5: Smoke run then full training (train.py)"
