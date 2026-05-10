# ml/src/data_extraction/__init__.py
#
# Data-generation tools used to produce the 68,556 graph .pt files
# and 68,570 token .pt files from raw Solidity source contracts.
#
# These scripts are NOT part of the inference pipeline —
# they were run ONCE to build ml/data/graphs/ and ml/data/tokens/.
# They are kept here for reproducibility and audit purposes.
#
# Contents:
#   ast_extractor.py — Slither-based AST → PyG graph builder (was: ast_extractor_v4_production.py)
#   tokenizer.py     — CodeBERT tokenizer + .pt serializer (was: tokenizer_v1_production.py)
