"""One-epoch optimization and positive-only selection for R4 Phase 8."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from sentinel_data.vnext.policy import CLASS_NAMES
from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_losses import masked_bce_positive_loss, positive_selection_metrics
from ml.src.training.vnext_phase8_config import Phase8Settings


def _move_batch(batch, device: torch.device):
    graphs, tokens, supervision, contract_ids, roles, group_ids = batch
    graphs = graphs.to(device)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    supervision = {k: v.to(device) for k, v in supervision.items()}
    return graphs, tokens, supervision, contract_ids, roles, group_ids


def _batch_limit(loader: DataLoader, limit: int | None) -> int:
    return len(loader) if limit is None else min(len(loader), max(1, int(limit)))


def train_masked_epoch(
    *,
    model: SentinelModel,
    loader: DataLoader,
    sampler: DeterministicGroupSampler,
    optimizer,
    scheduler,
    device: torch.device,
    settings: Phase8Settings,
    epoch: int,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    sampler.set_epoch(epoch)
    model.train()
    model._current_epoch = epoch
    optimizer.zero_grad(set_to_none=True)
    total_batches = _batch_limit(loader, max_batches)
    accum = max(1, settings.gradient_accumulation_steps)
    aux_weight = settings.aux_loss_weight * min(
        1.0, epoch / max(1, settings.aux_loss_warmup_epochs)
    )
    sums = {"total": 0.0, "main": 0.0, "aux": 0.0, "phase2": 0.0}
    optimizer_steps = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= total_batches:
            break
        graphs, tokens, supervision, *_ = _move_batch(batch, device)
        targets = supervision["targets"]
        loss_mask = supervision["effective_loss_mask"]
        strengths = supervision["strength_codes"]
        window_start = (batch_idx // accum) * accum
        actual_window = min(accum, total_batches - window_start)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            logits, aux = model(
                graphs,
                tokens["input_ids"],
                tokens["attention_mask"],
                return_aux=True,
            )
            main_loss = masked_bce_positive_loss(
                logits,
                targets,
                loss_mask,
                strengths,
                weak_positive_weight=settings.weak_positive_weight,
            )
            eye_losses = [
                masked_bce_positive_loss(
                    aux[name],
                    targets,
                    loss_mask,
                    strengths,
                    weak_positive_weight=settings.weak_positive_weight,
                )
                for name in ("gnn", "transformer", "fused")
            ]
            aux_loss = sum(eye_losses)
            phase2_loss = masked_bce_positive_loss(
                aux["phase2"],
                targets,
                loss_mask,
                strengths,
                weak_positive_weight=settings.weak_positive_weight,
            )
            total_loss = (
                main_loss
                + aux_weight * aux_loss
                + settings.aux_phase2_loss_weight * phase2_loss
            )
            if settings.jk_entropy_reg_lambda > 0.0:
                entropy = aux.get("jk_entropy")
                if entropy is not None:
                    max_entropy = math.log(3.0)
                    total_loss = total_loss + settings.jk_entropy_reg_lambda * (
                        max_entropy - entropy.clamp(max=max_entropy)
                    )
            scaled_loss = total_loss / actual_window

        if not torch.isfinite(scaled_loss).item():
            raise RuntimeError(
                f"non-finite Phase-8 loss at epoch={epoch} batch={batch_idx}"
            )
        scaled_loss.backward()
        should_step = ((batch_idx + 1) % accum == 0) or (
            batch_idx + 1 == total_batches
        )
        if should_step:
            trainable = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, settings.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise RuntimeError(
                    f"non-finite Phase-8 gradient norm at epoch={epoch} batch={batch_idx}"
                )
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in trainable
            ):
                raise RuntimeError(
                    f"non-finite Phase-8 gradient at epoch={epoch} batch={batch_idx}"
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        sums["total"] += float(total_loss.detach().item())
        sums["main"] += float(main_loss.detach().item())
        sums["aux"] += float(aux_loss.detach().item())
        sums["phase2"] += float(phase2_loss.detach().item())

    if optimizer_steps <= 0:
        raise RuntimeError("Phase-8 epoch produced no optimizer steps")
    return {
        "loss": sums["total"] / total_batches,
        "main_loss": sums["main"] / total_batches,
        "aux_loss": sums["aux"] / total_batches,
        "phase2_loss": sums["phase2"] / total_batches,
        "optimizer_steps": float(optimizer_steps),
        "aux_weight_effective": float(aux_weight),
    }


@torch.no_grad()
def evaluate_positive_selection(
    *,
    model: SentinelModel,
    loader: DataLoader,
    device: torch.device,
    settings: Phase8Settings,
    epoch: int,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    model._current_epoch = epoch
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    masks_all: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    total_batches = _batch_limit(loader, max_batches)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= total_batches:
            break
        graphs, tokens, supervision, contract_ids, *_ = _move_batch(batch, device)
        logits = model(graphs, tokens["input_ids"], tokens["attention_mask"])
        targets = supervision["targets"]
        metric_mask = supervision["outcome_metric_mask"]
        logits_all.append(logits.float().cpu())
        targets_all.append(targets.float().cpu())
        masks_all.append(metric_mask.bool().cpu())
        probs = torch.sigmoid(logits.float()).cpu()
        mask_cpu = metric_mask.bool().cpu()
        for row_idx, contract_id in enumerate(contract_ids):
            for class_idx in torch.where(mask_cpu[row_idx])[0].tolist():
                records.append(
                    {
                        "contract_id": contract_id,
                        "class_index": int(class_idx),
                        "class_name": CLASS_NAMES[class_idx],
                        "probability": float(probs[row_idx, class_idx].item()),
                    }
                )

    if not logits_all:
        raise RuntimeError("Phase-8 model-selection loader produced no batches")
    metrics = positive_selection_metrics(
        torch.cat(logits_all),
        torch.cat(targets_all),
        torch.cat(masks_all),
        threshold=settings.fixed_diagnostic_threshold,
    )
    return metrics, records


__all__ = ["evaluate_positive_selection", "train_masked_epoch"]
