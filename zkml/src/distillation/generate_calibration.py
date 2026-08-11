"""Generate EZKL calibration inputs from an explicitly selected DATA export.

There is no implicit historical export. The calibration artifact is accompanied
by a manifest binding it to the teacher checkpoint and DATA export manifest so
circuit setup cannot silently consume a feature distribution from another run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.datasets.collate import sentinel_collate_fn
from ml.src.datasets.sentinel_dataset import SentinelDataset
from ml.src.models.sentinel_model import SentinelModel

TEACHER_CHECKPOINT = Path(
    "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
)
CALIBRATION_OUTPUT = Path("zkml/ezkl/calibration.json")
CALIBRATION_MANIFEST = Path("zkml/ezkl/calibration.manifest.json")
N_CALIBRATION_SAMPLES = 200
RANDOM_SEED = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_teacher(device: str) -> SentinelModel:
    if not TEACHER_CHECKPOINT.exists():
        raise FileNotFoundError(f"teacher checkpoint missing: {TEACHER_CHECKPOINT}")
    checkpoint = torch.load(
        TEACHER_CHECKPOINT, map_location=device, weights_only=False
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config = checkpoint.get("config", {}) or {}
    else:
        state_dict = checkpoint
        config = {}

    teacher = SentinelModel(
        num_classes=config.get("num_classes", 10),
        fusion_output_dim=config.get("fusion_output_dim", 128),
        gnn_prefix_k=config.get("gnn_prefix_k", 0),
        gnn_prefix_warmup_epochs=config.get("gnn_prefix_warmup_epochs", 15),
        use_edge_attr=config.get("use_edge_attr", True),
        gnn_hidden_dim=config.get("gnn_hidden_dim", 256),
        gnn_num_layers=config.get("gnn_layers", 8),
        gnn_heads=config.get("gnn_heads", 8),
        gnn_dropout=config.get("gnn_dropout", 0.2),
        gnn_use_jk=config.get("gnn_use_jk", True),
        gnn_jk_mode=config.get("gnn_jk_mode", "attention"),
        fusion_max_nodes=config.get("fusion_max_nodes", 1024),
    ).to(device)
    teacher.load_state_dict(state_dict)
    teacher.float().eval()
    return teacher


@torch.no_grad()
def generate(
    export_dir: Path,
    *,
    n_samples: int = N_CALIBRATION_SAMPLES,
    output: Path = CALIBRATION_OUTPUT,
    manifest_output: Path = CALIBRATION_MANIFEST,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    export_dir = export_dir.resolve()
    export_manifest = export_dir / "manifest.json"
    if not export_manifest.exists():
        raise FileNotFoundError(
            f"export manifest missing: {export_manifest}; refusing unbound calibration"
        )
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    teacher = _load_teacher(device)
    val_dataset = SentinelDataset(split="val", export_dir=str(export_dir))
    if len(val_dataset) == 0:
        raise RuntimeError("validation split is empty")
    n_requested = min(n_samples, len(val_dataset))
    loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=sentinel_collate_fn,
    )

    features: list[torch.Tensor] = []
    collected = 0
    for graphs, tokens, _y, _cids, _tiers in loader:
        _logits, aux = teacher(
            graphs.to(device),
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
            return_aux=True,
        )
        fusion = aux["fusion_embedding"].float().cpu()
        if fusion.ndim != 2 or fusion.shape[1] != 128:
            raise RuntimeError(
                f"teacher fusion shape must be [B,128], got {tuple(fusion.shape)}"
            )
        if not torch.isfinite(fusion).all():
            raise RuntimeError("teacher produced non-finite fusion feature(s)")
        features.append(fusion)
        collected += fusion.shape[0]
        if collected >= n_requested:
            break

    if not features:
        raise RuntimeError("no calibration features were produced")
    tensor = torch.cat(features)[:n_requested]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"input_data": tensor.numpy().tolist()}), encoding="utf-8"
    )

    flat = tensor.flatten()
    manifest = {
        "schema": "sentinel-zkml-calibration-v1",
        "input_dim": 128,
        "sample_count": int(tensor.shape[0]),
        "selection": "validation_split_prefix_deterministic",
        "random_seed": RANDOM_SEED,
        "teacher_checkpoint": {
            "path": TEACHER_CHECKPOINT.as_posix(),
            "sha256": _sha256(TEACHER_CHECKPOINT),
        },
        "data_export": {
            "path": export_dir.as_posix(),
            "manifest_path": export_manifest.as_posix(),
            "manifest_sha256": _sha256(export_manifest),
        },
        "calibration": {
            "path": output.as_posix(),
            "sha256": _sha256(output),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Calibration generated and bound: {}", manifest_output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=N_CALIBRATION_SAMPLES)
    parser.add_argument("--output", type=Path, default=CALIBRATION_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=CALIBRATION_MANIFEST)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate(
        args.export_dir,
        n_samples=args.samples,
        output=args.output,
        manifest_output=args.manifest,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
