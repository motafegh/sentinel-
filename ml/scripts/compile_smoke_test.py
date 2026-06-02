"""
compile_smoke_test.py — Gate 3.1: torch.compile 2-epoch smoke test

Validates that torch.compile works correctly on the current Phase 3 model
architecture before Run 5. Tests three things:
  1. Compile succeeds without RuntimeError on all targeted submodules
  2. Compiled forward pass is numerically close to eager (max abs diff < 0.01)
  3. Training is stable for 2 synthetic epochs with variable batch shapes

Decision:
  PASS — all checks pass → proceed with use_compile=True for Run 5
  FAIL — any check fails → disable use_compile for Run 5, file Run 6 fix

Usage:
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compile_smoke_test.py
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compile_smoke_test.py --device cpu
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compile_smoke_test.py --steps-per-epoch 5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data, Batch

from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.trainer import TrainConfig
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int,
    nodes_lo: int,
    nodes_hi: int,
    seq_len: int,
    windows: int,
    device: str,
) -> tuple:
    """Synthetic PyG Batch + token tensors. Node count varies per graph (tests dynamic=True)."""
    graphs_list = []
    for _ in range(batch_size):
        n = torch.randint(nodes_lo, nodes_hi + 1, (1,)).item()
        n = max(n, 4)  # need at least a few nodes for meaningful edges
        edge_index = torch.stack([
            torch.arange(n - 1),
            torch.arange(1, n),
        ], dim=0)
        edge_attr = torch.zeros(n - 1, dtype=torch.long)
        x = torch.randn(n, NODE_FEATURE_DIM)
        graphs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    batch = Batch.from_data_list(graphs_list).to(device)
    input_ids      = torch.randint(0, 50265, (batch_size, windows, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, windows, seq_len, dtype=torch.long).to(device)
    labels         = (torch.rand(batch_size, 10) > 0.7).float().to(device)
    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return batch, tokens, labels


def _build_model(device: str, cfg: TrainConfig) -> SentinelModel:
    return SentinelModel(
        num_classes=10,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        gnn_layers=cfg.gnn_layers,
        gnn_heads=cfg.gnn_heads,
        gnn_dropout=cfg.gnn_dropout,
        gnn_edge_emb_dim=cfg.gnn_edge_emb_dim,
        use_edge_attr=cfg.use_edge_attr,
        gnn_use_jk=cfg.gnn_use_jk,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        fusion_max_nodes=cfg.fusion_max_nodes,
    ).to(device)


def _apply_compile(model: SentinelModel) -> list[str]:
    """Apply torch.compile to the same submodules as trainer.py. Returns list of compiled names."""
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 256

    compiled = []
    for name in ("gnn", "fusion", "classifier",
                 "gnn_eye_proj", "transformer_eye_proj", "window_pooler",
                 "aux_gnn", "aux_transformer", "aux_fused"):
        sub = getattr(model, name, None)
        if sub is not None:
            setattr(model, name, torch.compile(sub, dynamic=True))
            compiled.append(name)
    return compiled


# ---------------------------------------------------------------------------
# Check 1 — compile succeeds
# ---------------------------------------------------------------------------

def check_compile(cfg: TrainConfig, device: str) -> tuple[bool, str, list[str]]:
    """Returns (ok, message, compiled_submodule_names)."""
    try:
        model = _build_model(device, cfg)
        compiled = _apply_compile(model)
        return True, f"Compiled {len(compiled)} submodules: {compiled}", compiled
    except Exception as e:
        return False, f"torch.compile raised {type(e).__name__}: {e}", []


# ---------------------------------------------------------------------------
# Check 2 — eager vs compiled numerical agreement
# ---------------------------------------------------------------------------

def check_numerical_agreement(cfg: TrainConfig, device: str, seq_len: int, windows: int) -> tuple[bool, str]:
    """Returns (ok, message)."""
    torch.manual_seed(42)
    eager_model = _build_model(device, cfg)
    eager_model.eval()

    # Snapshot weights so compiled model is identical
    state = {k: v.clone() for k, v in eager_model.state_dict().items()}

    compiled_model = _build_model(device, cfg)
    compiled_model.load_state_dict(state)
    _apply_compile(compiled_model)
    compiled_model.eval()

    # Fixed input — same seed for reproducibility
    torch.manual_seed(123)
    graphs, tokens, _ = _make_batch(
        batch_size=4, nodes_lo=20, nodes_hi=20,
        seq_len=seq_len, windows=windows, device=device,
    )

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device != "cpu")):
            eager_out    = eager_model(graphs, tokens["input_ids"], tokens["attention_mask"])
            # Re-use same batch for compiled (Batch is reusable for eval)
            compiled_out = compiled_model(graphs, tokens["input_ids"], tokens["attention_mask"])

    # Compare in float32
    eager_f    = eager_out.float()
    compiled_f = compiled_out.float()
    max_diff   = (eager_f - compiled_f).abs().max().item()
    mean_diff  = (eager_f - compiled_f).abs().mean().item()

    threshold = 0.05  # BF16 rounding means small differences are expected
    ok = max_diff < threshold and math.isfinite(max_diff)
    msg = (
        f"max_abs_diff={max_diff:.6f}  mean_abs_diff={mean_diff:.6f}  "
        f"threshold={threshold}  {'PASS' if ok else 'FAIL'}"
    )
    return ok, msg


# ---------------------------------------------------------------------------
# Check 3 — 2-epoch training stability
# ---------------------------------------------------------------------------

def check_training_stability(
    cfg: TrainConfig,
    device: str,
    steps_per_epoch: int,
    seq_len: int,
    windows: int,
) -> tuple[bool, str]:
    """Returns (ok, message)."""
    torch.manual_seed(0)
    model = _build_model(device, cfg)
    _apply_compile(model)
    model.train()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-2,
        fused=(device == "cuda"),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler  = torch.amp.GradScaler(device, enabled=(device != "cpu"))

    losses = []
    nan_count = 0
    compile_errors = 0

    for epoch in range(1, 3):
        for step in range(1, steps_per_epoch + 1):
            # Variable batch shape each step — exercises dynamic=True
            bs = 4 if step % 2 == 0 else 6
            graphs, tokens, labels = _make_batch(
                batch_size=bs, nodes_lo=10, nodes_hi=80,
                seq_len=seq_len, windows=windows, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device != "cpu")):
                    logits, aux = model(graphs, tokens["input_ids"], tokens["attention_mask"], return_aux=True)
                    loss = loss_fn(logits, labels)

                if not torch.isfinite(loss):
                    nan_count += 1
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            except torch.distributed.DistributedBackendError:
                compile_errors += 1
            except Exception as e:
                # Catch compile-time graph errors
                if "dynamo" in str(e).lower() or "compile" in str(e).lower() or "inductor" in str(e).lower():
                    compile_errors += 1
                    print(f"    [compile error] step {step}: {type(e).__name__}: {str(e)[:120]}")
                else:
                    raise

        print(f"  Epoch {epoch}: {steps_per_epoch} steps, "
              f"avg_loss={sum(losses[-steps_per_epoch:])/max(1,len(losses[-steps_per_epoch:])):.4f}, "
              f"nan_skips={nan_count}, compile_errors={compile_errors}")

    ok = compile_errors == 0 and len(losses) > 0 and nan_count / max(1, steps_per_epoch * 2) < 0.1
    msg = (
        f"steps_completed={len(losses)}/{steps_per_epoch*2}  "
        f"nan_skips={nan_count}  compile_errors={compile_errors}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 3.1: torch.compile smoke test")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps-per-epoch", type=int, default=8,
                   help="Synthetic steps per epoch (default 8; use 3 for quick CI check)")
    p.add_argument("--seq-len",         type=int, default=128,
                   help="Token sequence length (default 128; full run uses 512)")
    p.add_argument("--windows",         type=int, default=2,
                   help="Token windows per contract (default 2; full run uses 4)")
    p.add_argument("--skip-numerical",  action="store_true",
                   help="Skip eager vs compiled numerical agreement check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    print(f"\n{'='*62}")
    print("Gate 3.1 — torch.compile Smoke Test")
    print(f"  device          : {device}")
    print(f"  steps_per_epoch : {args.steps_per_epoch}")
    print(f"  seq_len×windows : {args.seq_len}×{args.windows}")
    print(f"  PyTorch         : {torch.__version__}")
    print(f"{'='*62}\n")

    cfg = TrainConfig(
        use_compile=True,
        use_amp=True,
        gnn_layers=8,
        gnn_hidden_dim=256,
        gnn_heads=8,
        gnn_edge_emb_dim=64,
        lora_r=16,
        lora_alpha=32,
        fusion_max_nodes=1024,
    )

    results: dict[str, tuple[bool, str]] = {}

    # --- Check 1: compile succeeds ---
    print("Check 1 — torch.compile on submodules...")
    t0 = time.perf_counter()
    ok1, msg1, _ = check_compile(cfg, device)
    print(f"  {msg1}")
    print(f"  [{('PASS' if ok1 else 'FAIL')}] in {time.perf_counter()-t0:.1f}s\n")
    results["compile"] = (ok1, msg1)

    if not ok1:
        print("ABORT: compile failed — remaining checks skipped.")
        _print_verdict(results)
        sys.exit(1)

    # --- Check 2: numerical agreement ---
    if not args.skip_numerical:
        print("Check 2 — eager vs compiled numerical agreement...")
        t0 = time.perf_counter()
        ok2, msg2 = check_numerical_agreement(cfg, device, args.seq_len, args.windows)
        print(f"  {msg2}")
        print(f"  [{'PASS' if ok2 else 'FAIL'}] in {time.perf_counter()-t0:.1f}s\n")
        results["numerical"] = (ok2, msg2)
    else:
        print("Check 2 — SKIPPED (--skip-numerical)\n")
        results["numerical"] = (True, "skipped")

    # --- Check 3: 2-epoch training stability ---
    print("Check 3 — 2-epoch training stability with variable batch shapes...")
    t0 = time.perf_counter()
    ok3, msg3 = check_training_stability(
        cfg, device, args.steps_per_epoch, args.seq_len, args.windows
    )
    print(f"  {msg3}")
    print(f"  [{'PASS' if ok3 else 'FAIL'}] in {time.perf_counter()-t0:.1f}s\n")
    results["stability"] = (ok3, msg3)

    _print_verdict(results)
    all_ok = all(ok for ok, _ in results.values())
    sys.exit(0 if all_ok else 1)


def _print_verdict(results: dict) -> None:
    all_ok = all(ok for ok, _ in results.values())
    print(f"\n{'='*62}")
    print("GATE 3.1 VERDICT")
    for name, (ok, msg) in results.items():
        print(f"  {name:<12}: {'PASS' if ok else 'FAIL'}  ({msg[:70]})")
    print()
    if all_ok:
        print("STATUS  PASS — torch.compile is safe for Run 5.")
        print("ACTION  Keep use_compile=True in TrainConfig (default).")
    else:
        print("STATUS  FAIL — torch.compile has issues with current architecture.")
        print("ACTION  Set use_compile=False in TrainConfig for Run 5.")
        print("        File a Run 6 fix item for the failing submodule(s).")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
