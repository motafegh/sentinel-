"""
preprocess.py — Inline Preprocessing for SENTINEL Inference

Converts a Solidity contract into the two inputs SentinelModel expects:
  - graph : PyG Data object  (AST/CFG structure  → GNNEncoder)
  - tokens: dict             (CodeBERT tensors   → TransformerEncoder)

Two public entry points:
  process(sol_path)          — contract already on disk (batch use, predictor CLI)
  process_source(source_code)— raw string from HTTP API or stdin

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DON'T USE graph_builder.GraphBuilder HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
graph_builder.py uses one-hot encoding for node features:
  type_onehot(4) + visibility_onehot(4) + mutability_onehot(6) + flags(3) = 17 dims

The training data (68,555 .pt files in ml/data/graphs/) was built by
ml/data_extraction/ast_extractor.py, which uses a completely different
feature function producing 8 raw floats per node:
  [type_id, visibility, pure, view, payable, reentrant, complexity, loc]

GNNEncoder has in_channels=8 hardcoded and the checkpoint was trained on 8-dim
features. Passing 17-dim features causes:
  "mat1 and mat2 shapes cannot be multiplied (N×17 and 8×64)"

graph_builder.py was written as a separate experiment and was NEVER used to
produce the training data. It is not deleted — it may be used if the model
is retrained with richer features — but it must NOT be used here.

The _extract_graph() method below replicates ast_extractor.ASTExtractorV4
node_features() exactly. Any future change to feature engineering must update:
  1. ml/data_extraction/ast_extractor.py  (offline dataset build)
  2. _extract_graph() in this file        (online inference)
  3. GNNEncoder(in_channels=N)            (model architecture)
  4. Retrain the model from scratch

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY SLITHER REQUIRES A TEMP FILE FOR process_source()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Slither shells out to solc (the Solidity compiler). solc requires a real
file path — it cannot read from a pipe or string argument.
process_source() writes a NamedTemporaryFile, runs Slither, then deletes
the file in a finally block. The tokeniser is called directly with the
string — no redundant file read.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHAPE CONTRACT (must match training data exactly — do not change without retraining)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  graph.x              [N, 8]     float32 — N nodes, 8 features each
  graph.edge_index     [2, E]     int64   — E directed edges
  tokens["input_ids"]  [1, 512]   long    — batch dim=1 for single inference
  tokens["attention_mask"] [1, 512] long

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASHING CONVENTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  process()        → MD5 of resolved file path   (matches offline pipeline)
  process_source() → MD5 of source content       (content-addressable)
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch_geometric.data import Data
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Feature vocabulary — MUST be identical to ast_extractor_v4_production.py.
# If you change any value here, you must rebuild all .pt files and retrain.
# ─────────────────────────────────────────────────────────────────────────────

_NODE_TYPES: dict[str, int] = {
    "STATE_VAR":   0,
    "FUNCTION":    1,
    "MODIFIER":    2,
    "EVENT":       3,
    "FALLBACK":    4,
    "RECEIVE":     5,
    "CONSTRUCTOR": 6,
    "CONTRACT":    7,
}

# Visibility: ordinal encoding (not one-hot).
# public/external=0 (most open), internal=1, private=2 (most closed).
_VISIBILITY_MAP: dict[str, int] = {
    "public":   0,
    "external": 0,
    "internal": 1,
    "private":  2,
}

_EDGE_TYPES: dict[str, int] = {
    "CALLS":    0,
    "READS":    1,
    "WRITES":   2,
    "EMITS":    3,
    "INHERITS": 4,
}


class ContractPreprocessor:
    """
    Converts a Solidity contract into (graph, tokens) for SentinelModel.

    Instantiate once, reuse for many contracts.
    All expensive initialisation (tokenizer load) happens in __init__.

    Public API:
      process(sol_path)           — contract file already on disk
      process_source(source_code) — raw Solidity string (HTTP API, stdin)
    """

    TOKENIZER_NAME   = "microsoft/codebert-base"
    MAX_TOKEN_LENGTH = 512
    # Reject source_code over this size before any Slither/tokenizer work begins.
    # api.py also enforces this at the HTTP boundary; this is a defence-in-depth guard.
    MAX_SOURCE_BYTES = 1 * 1024 * 1024  # 1 MB

    def __init__(self) -> None:
        logger.info("ContractPreprocessor initialising...")

        # Load CodeBERT tokenizer once — reused for every inference call.
        # Must match tokenizer_v1_production.py used during offline preprocessing.
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)

        logger.info("ContractPreprocessor ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, sol_path: str | Path) -> tuple[Data, dict]:
        """
        Convert a .sol file on disk into (graph, tokens) for SentinelModel.

        Hashing: MD5 of resolved absolute path — matches offline pipeline
        (ast_extractor_v4_production.py → get_contract_hash).

        Args:
            sol_path: Path to the Solidity source file (.sol).

        Returns:
            (graph, tokens) — see process_source() for shape details.

        Raises:
            FileNotFoundError: if sol_path does not exist.
            ValueError:        if graph extraction fails.
        """
        sol_path = Path(sol_path)
        if not sol_path.exists():
            raise FileNotFoundError(f"Contract not found: {sol_path}")

        logger.info(f"Preprocessing file: {sol_path.name}")

        contract_hash = hashlib.md5(str(sol_path.resolve()).encode()).hexdigest()

        graph  = self._extract_graph(sol_path, contract_hash)
        tokens = self._tokenize(
            sol_path.read_text(encoding="utf-8", errors="ignore"),
            contract_hash,
        )

        self._log_result(graph, tokens)
        return graph, tokens

    def process_source(
        self,
        source_code: str,
        name: str = "contract",
    ) -> tuple[Data, dict]:
        """
        Convert a raw Solidity string into (graph, tokens) for SentinelModel.

        Slither shells out to solc which requires a real file path.
        This method writes source to a NamedTemporaryFile, runs extraction,
        then deletes it in a finally block.

        Hashing: MD5 of source content — same source always → same hash,
        enabling API-layer response caching.

        Args:
            source_code: Raw Solidity source text.
            name:        Label for log messages and temp file prefix.

        Returns:
            graph: PyG Data object.
                graph.x              [N, 8]   float32
                graph.edge_index     [2, E]   int64
                graph.contract_hash  str
            tokens: dict
                "input_ids"          [1, 512] long
                "attention_mask"     [1, 512] long
                "contract_hash"      str
                "num_tokens"         int
                "truncated"          bool

        Raises:
            ValueError: if source_code is empty or graph extraction fails.
        """
        if not source_code or not source_code.strip():
            raise ValueError("source_code is empty")

        source_bytes = len(source_code.encode("utf-8"))
        if source_bytes > self.MAX_SOURCE_BYTES:
            raise ValueError(
                f"source_code too large ({source_bytes:,} bytes > "
                f"{self.MAX_SOURCE_BYTES:,} limit). "
                "Consider splitting or summarising the contract before analysis."
            )

        logger.info(f"Preprocessing source: {name!r} ({len(source_code)} chars)")

        contract_hash = hashlib.md5(source_code.encode("utf-8")).hexdigest()

        # Sanitize name for use as a temp-file prefix — strip characters that are
        # unsafe in file names (path separators, spaces, etc.).
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name[:32])

        # delete=False: close before Slither opens (required on some platforms).
        # finally block guarantees cleanup even if extraction raises.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".sol",
            prefix=f"sentinel_{safe_name}_",
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        try:
            tmp.write(source_code)
            tmp.close()
            graph = self._extract_graph(Path(tmp.name), contract_hash)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError as _e:
                # Log so the failure is visible — temp files accumulate silently
                # if this goes unnoticed (e.g. process killed while Slither runs).
                logger.warning(f"Failed to delete temp file {tmp.name}: {_e}")

        tokens = self._tokenize(source_code, contract_hash)
        self._log_result(graph, tokens)
        return graph, tokens

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_graph(self, sol_path: Path, contract_hash: str) -> Data:
        """
        Parse a .sol file → Slither AST → PyG Data object.

        ⚠️  CRITICAL — DO NOT CHANGE NODE FEATURE LOGIC WITHOUT RETRAINING ⚠️

        Replicates node_features() from ml/data_extraction/ast_extractor.py
        (ASTExtractorV4) EXACTLY. That script built all 68,555 training .pt files.
        The checkpoint was trained on those exact 8-dim float vectors.

        Node feature vector (8 dims, float32):
          Index  Feature      Notes
          ─────  ──────────── ──────────────────────────────────────────────
            0    type_id      float from _NODE_TYPES (0–7)
            1    visibility   0=public/external, 1=internal, 2=private
            2    pure         1.0 if pure function, else 0.0
            3    view         1.0 if view function, else 0.0
            4    payable      1.0 if payable function, else 0.0
            5    reentrant    1.0 if is_reentrant, else 0.0
            6    complexity   float(len(func.nodes)) — CFG node count
            7    loc          float(len(source_mapping.lines)) — lines of code

        graph_builder.GraphBuilder is NOT used — it produces 17-dim one-hot
        features that were never used during training. See module docstring.

        Args:
            sol_path:      Path to .sol file (must exist on disk).
            contract_hash: MD5 string — attached to returned graph.

        Returns:
            PyG Data with graph.x [N, 8] and graph.edge_index [2, E].

        Raises:
            ValueError: wraps any Slither/solc error with file context.
        """
        try:
            from slither import Slither
            from slither.core.declarations import Function

            # detectors_to_run=[] disables security detectors — we only need AST.
            sl = Slither(str(sol_path), detectors_to_run=[])

            # Skip imported dependencies (OpenZeppelin etc.) — analyse user code only.
            contracts = [c for c in sl.contracts if not c.is_from_dependency()]
            if not contracts:
                raise ValueError("No non-dependency contracts found")

            contract = contracts[0]
            node_features_list: list[list[float]] = []
            node_map: dict[str, int] = {}  # canonical_name → index

            # ── Node feature extraction ────────────────────────────────────

            def _node_feat(obj: Any, type_id: int) -> list[float]:
                """8-dim vector — mirrors node_features() in production script."""
                vis_str    = str(getattr(obj, "visibility", "public"))
                visibility = float(_VISIBILITY_MAP.get(vis_str, 0))
                pure = view = payable = reentrant = complexity = loc = 0.0

                src = getattr(obj, "source_mapping", None)
                if src:
                    lines = getattr(src, "lines", None)
                    if lines:
                        loc = float(len(lines) if isinstance(lines, list) else lines)

                if isinstance(obj, Function):
                    pure      = 1.0 if obj.pure    else 0.0
                    view      = 1.0 if obj.view    else 0.0
                    payable   = 1.0 if obj.payable else 0.0
                    reentrant = 1.0 if getattr(obj, "is_reentrant", False) else 0.0
                    try:
                        complexity = float(len(obj.nodes)) if obj.nodes else 0.0
                    except Exception:
                        complexity = 0.0
                    # Special function kinds override the default FUNCTION(1) type_id.
                    if obj.is_constructor:
                        type_id = _NODE_TYPES["CONSTRUCTOR"]
                    elif obj.is_fallback:
                        type_id = _NODE_TYPES["FALLBACK"]
                    elif obj.is_receive:
                        type_id = _NODE_TYPES["RECEIVE"]

                return [float(type_id), visibility, pure, view, payable,
                        reentrant, complexity, loc]

            def _add_node(obj: Any, type_id: int) -> None:
                """Register node, skip duplicates. Key = canonical_name."""
                name = (
                    obj.name
                    if not hasattr(obj, "canonical_name")
                    else obj.canonical_name
                )
                if name not in node_map:
                    node_map[name] = len(node_features_list)
                    node_features_list.append(_node_feat(obj, type_id))

            # Node insertion order matches production script — do not reorder.
            # Order affects node indices which affects edge_index correctness.
            _add_node(contract, _NODE_TYPES["CONTRACT"])
            for var   in contract.state_variables: _add_node(var,   _NODE_TYPES["STATE_VAR"])
            for func  in contract.functions:       _add_node(func,  _NODE_TYPES["FUNCTION"])
            for mod   in contract.modifiers:       _add_node(mod,   _NODE_TYPES["MODIFIER"])
            for event in contract.events:          _add_node(event, _NODE_TYPES["EVENT"])

            if not node_features_list:
                raise ValueError("Contract produced zero graph nodes")

            # ── Edge extraction ───────────────────────────────────────────
            # edge_attr records the edge type ID matching _EDGE_TYPES.
            # GNNEncoder does not currently use edge_attr, but storing it here
            # ensures online/offline graph object parity (offline .pt files include it).

            edges:      list[list[int]] = []
            edge_types: list[int]       = []

            def _add_edge(src: str, dst: str, etype: int) -> None:
                si, di = node_map.get(src), node_map.get(dst)
                if si is not None and di is not None:
                    edges.append([si, di])
                    edge_types.append(etype)

            for func in contract.functions:
                fn = func.canonical_name
                for call in func.internal_calls:
                    if hasattr(call, "canonical_name"):
                        _add_edge(fn, call.canonical_name, _EDGE_TYPES["CALLS"])
                for var in func.state_variables_read:
                    _add_edge(fn, var.canonical_name, _EDGE_TYPES["READS"])
                for var in func.state_variables_written:
                    _add_edge(fn, var.canonical_name, _EDGE_TYPES["WRITES"])
                # events_emitted not available in all Slither versions — guarded.
                if hasattr(func, "events_emitted"):
                    try:
                        for evt in func.events_emitted:
                            _add_edge(fn, evt.canonical_name, _EDGE_TYPES["EMITS"])
                    except Exception:
                        pass

            # INHERITS edges — contract → parent contracts.
            try:
                for parent in contract.inheritance:
                    _add_edge(contract.name, parent.name, _EDGE_TYPES["INHERITS"])
            except Exception:
                pass

            # ── Assemble PyG Data object ──────────────────────────────────

            x = torch.tensor(node_features_list, dtype=torch.float)  # [N, 8]

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr  = torch.tensor(edge_types, dtype=torch.long)  # [E]
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr  = torch.zeros(0, dtype=torch.long)

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph.contract_hash = contract_hash
            graph.contract_path = str(sol_path.resolve())
            graph.contract_name = contract.name
            # Dummy label — never used in forward pass; required by PyG internals.
            graph.y = torch.tensor([0], dtype=torch.long)

            return graph

        except ImportError as exc:
            raise RuntimeError(
                "Slither is not installed. "
                "Run: pip install slither-analyzer  (or: poetry add slither-analyzer)"
            ) from exc
        except Exception as exc:
            exc_str = str(exc).lower()
            # Distinguish user-input errors (bad Solidity) from infrastructure failures.
            # Solc/compilation errors are the user's fault → ValueError (HTTP 400).
            # Missing solc, Slither internals, OS errors → RuntimeError (HTTP 500).
            if any(kw in exc_str for kw in ("solc", "compil", "syntax", "parsing", "invalid solidity")):
                raise ValueError(
                    f"Solidity compilation error in '{sol_path.name}': {exc}"
                ) from exc
            raise RuntimeError(
                f"Graph extraction failed (infrastructure error) for '{sol_path.name}': {exc}"
            ) from exc

    def _tokenize(self, source_code: str, contract_hash: str) -> dict:
        """
        Tokenize Solidity source using CodeBERT tokenizer.

        Pure function — no file I/O. Called by both process() and process_source().

        Settings match tokenizer_v1_production.py exactly:
          max_length=512, truncation=True, padding="max_length"
        Changing these requires rebuilding token files and retraining.

        Truncation detection uses a second encode() call (without truncation)
        to get the true token count. Comparing decoded text is unreliable —
        the tokenizer normalises whitespace on the round-trip.

        Args:
            source_code:   Raw Solidity source string.
            contract_hash: MD5 string — stored in returned dict.

        Returns:
            dict with "input_ids" [1,512], "attention_mask" [1,512],
            "contract_hash", "num_tokens", "truncated".
        """
        encoded = self.tokenizer(
            source_code,
            max_length=self.MAX_TOKEN_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        num_tokens: int = int(encoded["attention_mask"].sum().item())

        true_token_count = len(
            self.tokenizer.encode(source_code, add_special_tokens=True)
        )
        truncated: bool = true_token_count > self.MAX_TOKEN_LENGTH

        if truncated:
            logger.debug(
                f"Truncated: {true_token_count} tokens → {self.MAX_TOKEN_LENGTH} "
                f"(lost {true_token_count - self.MAX_TOKEN_LENGTH} tokens from tail)"
            )

        # Shape invariant — must match training data and GNNEncoder in_channels.
        # RuntimeError (not assert) so python -O cannot strip these guards silently.
        expected = torch.Size([1, self.MAX_TOKEN_LENGTH])
        if encoded["input_ids"].shape != expected:
            raise RuntimeError(
                f"input_ids shape mismatch: got {encoded['input_ids'].shape}, "
                f"expected {expected}. "
                "Check tokenizer max_length and padding settings — "
                f"TOKENIZER_NAME={self.TOKENIZER_NAME}, MAX_TOKEN_LENGTH={self.MAX_TOKEN_LENGTH}."
            )
        if encoded["attention_mask"].shape != expected:
            raise RuntimeError(
                f"attention_mask shape mismatch: got {encoded['attention_mask'].shape}, "
                f"expected {expected}."
            )

        return {
            "input_ids":      encoded["input_ids"],       # [1, 512]
            "attention_mask": encoded["attention_mask"],  # [1, 512]
            "contract_hash":  contract_hash,
            "num_tokens":     num_tokens,
            "truncated":      truncated,
        }

    def _log_result(self, graph: Data, tokens: dict) -> None:
        logger.info(
            f"Preprocessing complete — "
            f"nodes: {graph.num_nodes}, "
            f"edges: {graph.num_edges}, "
            f"real tokens: {tokens['num_tokens']}, "
            f"truncated: {tokens['truncated']}"
        )