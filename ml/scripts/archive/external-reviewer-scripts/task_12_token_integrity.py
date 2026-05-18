"""
Task 12: Token Integrity Checks
--------------------------------
Samples 100 token .pt files, checks shape, attention mask padding,
input_id range, num_tokens count, and schema version.

Run:
    python task_12_token_integrity.py
"""
import random
import numpy as np
from common import get_dirs, load_token, random_pt_sample, print_header

N = 100
CODEBERT_VOCAB_SIZE = 50265
PAD_TOKEN_ID        = 1     # CodeBERT with use_fast=True

def main():
    print_header(12, "Token Integrity Checks")
    _, _, _, tokens_dir, _, _ = get_dirs()

    paths = random_pt_sample(tokens_dir, N)
    print(f"Sampling {len(paths)} token files from {tokens_dir}\n")

    checks = {
        "shape_ok":          0,
        "mask_shape_ok":     0,
        "pad_windows_ok":    0,
        "real_windows_ok":   0,
        "vocab_range_ok":    0,
        "num_tokens_match":  0,
        "no_nan":            0,
        "pad_ids_ok":        0,
        "schema_v4":         0,
    }
    total  = 0
    failed_files = []
    window_dist  = {}   # num_windows -> count
    num_tokens_list = []

    for p in paths:
        try:
            t = load_token(p)
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")
            failed_files.append((p.stem, str(e)))
            continue
        total += 1

        # Normalise dict vs object access
        if isinstance(t, dict):
            get = lambda k: t.get(k)
        else:
            get = lambda k: getattr(t, k, None)

        ids   = get("input_ids")
        mask  = get("attention_mask")
        nw    = get("num_windows")
        ntok  = get("num_tokens")
        fschv = get("feature_schema_version")

        if ids is None:
            failed_files.append((p.stem, "missing input_ids"))
            continue

        import torch
        ids  = ids  if torch.is_tensor(ids)  else torch.tensor(ids)
        mask = mask if torch.is_tensor(mask) else torch.tensor(mask)

        max_win = ids.shape[0]
        nw = int(nw) if nw is not None else max_win

        # Track distribution
        window_dist[nw] = window_dist.get(nw, 0) + 1

        # 1. Shape
        if ids.shape == (max_win, 512):
            checks["shape_ok"] += 1

        # 2. Mask shape
        if mask.shape == ids.shape:
            checks["mask_shape_ok"] += 1

        # 3. Padding windows have all-zero mask
        pad_ok = True
        for wi in range(nw, max_win):
            if mask[wi].sum().item() != 0:
                pad_ok = False
        checks["pad_windows_ok"] += int(pad_ok)

        # 4. Real windows have non-zero mask
        real_ok = True
        for wi in range(nw):
            if mask[wi].sum().item() == 0:
                real_ok = False
        checks["real_windows_ok"] += int(real_ok)

        # 5. Vocab range
        if ids.max().item() < CODEBERT_VOCAB_SIZE and ids.min().item() >= 0:
            checks["vocab_range_ok"] += 1

        # 6. num_tokens == mask.sum()
        if ntok is not None:
            if int(mask.sum().item()) == int(ntok):
                checks["num_tokens_match"] += 1
            num_tokens_list.append(int(ntok))
        else:
            num_tokens_list.append(int(mask.sum().item()))

        # 7. No NaN/negative in ids
        ids_np = ids.float().numpy()
        if not np.any(np.isnan(ids_np)) and ids.min().item() >= 0:
            checks["no_nan"] += 1

        # 8. Padding window ids == PAD_TOKEN_ID
        pad_ids_ok = True
        for wi in range(nw, max_win):
            if not (ids[wi] == PAD_TOKEN_ID).all():
                pad_ids_ok = False
        checks["pad_ids_ok"] += int(pad_ids_ok)

        # 9. Schema version
        if fschv == "v4":
            checks["schema_v4"] += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"{'Check':<30} {'Pass':>6} / {'Total':>6}  {'Rate':>6}")
    print("-" * 52)
    for k, v in checks.items():
        rate = f"{v/total*100:.1f}%" if total else "N/A"
        print(f"  {k:<28} {v:>6} / {total:>6}  {rate:>6}")

    # ── Window distribution ───────────────────────────────────────────────────
    print("\nnum_windows distribution:")
    for w in sorted(window_dist):
        print(f"  W={w}: {window_dist[w]:>5} ({window_dist[w]/total*100:.1f}%)")

    # ── Token count stats ─────────────────────────────────────────────────────
    if num_tokens_list:
        arr = np.array(num_tokens_list)
        print(f"\nnum_tokens stats (n={len(arr)}):")
        print(f"  mean={arr.mean():.1f}  p50={np.percentile(arr,50):.0f}  "
              f"p95={np.percentile(arr,95):.0f}  max={arr.max():.0f}")

    # ── Failures ─────────────────────────────────────────────────────────────
    if failed_files:
        print(f"\n[BUG] Failed files ({len(failed_files)}):")
        for stem, reason in failed_files[:20]:
            print(f"  {stem}: {reason}")
    else:
        print("\n[CONFIRMED] All token files loaded without error.")

    # ── Summary finding ───────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────────────")
    fail_list = [(k, checks[k]) for k in checks if checks[k] < total]
    if not fail_list:
        print("  [CONFIRMED] All checks passed for all sampled token files.")
    else:
        for k, v in fail_list:
            print(f"  [BUG] {k}: {total - v} failures out of {total}")

if __name__ == "__main__":
    main()
