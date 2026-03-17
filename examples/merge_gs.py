# type: ignore
import math
from dataclasses import dataclass
from typing import Dict

import torch
import tyro
from torch import Tensor

# Re-use the two loader functions already implemented in Propagate_fill.
from Propagate_fill import load_bg_splats, create_new_splats_with_fg


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Path to the background .pt checkpoint.
    bg_ckpt: str = ""
    # Path to the foreground .pt checkpoint.
    fg_ckpt: str = ""
    # Where to write the merged .pt checkpoint.
    out_ckpt: str = "results/merged.pt"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(
    bg: torch.nn.ParameterDict,
    fg: torch.nn.ParameterDict,
) -> Dict[str, Tensor]:
    """
    Concatenate bg and fg Gaussians along the N dimension for every parameter.

    bg keys are frozen (requires_grad=False).
    fg keys are trainable (requires_grad=True) but opacity is near-zero.
    The merged tensors inherit requires_grad from whichever slice has it.
    """
    merged: Dict[str, Tensor] = {}

    all_keys = set(bg.keys()) | set(fg.keys())
    for key in all_keys:
        if key in bg and key in fg:
            # Pad SH higher-order bands if degrees differ.
            bv = bg[key].detach()
            fv = fg[key]
            if bv.dim() == 3 and fv.dim() == 3 and bv.shape[1] != fv.shape[1]:
                max_b = max(bv.shape[1], fv.shape[1])
                if bv.shape[1] < max_b:
                    pad = torch.zeros(bv.shape[0], max_b - bv.shape[1], bv.shape[2],
                                      device=bv.device)
                    bv = torch.cat([bv, pad], dim=1)
                if fv.shape[1] < max_b:
                    pad = torch.zeros(fv.shape[0], max_b - fv.shape[1], fv.shape[2],
                                      device=fv.device)
                    fv = torch.cat([fv, pad], dim=1)
            merged[key] = torch.cat([bv, fv], dim=0)
        elif key in bg:
            merged[key] = bg[key].detach()
        else:
            merged[key] = fg[key]

    return merged

def main(cfg: Config) -> None:
    assert cfg.bg_ckpt, "bg_ckpt must be set."
    assert cfg.fg_ckpt, "fg_ckpt must be set."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load background (frozen, foreground_logits dropped).
    bg_splats = load_bg_splats(cfg.bg_ckpt, device)

    # 2. Load foreground (trainable, foreground_logits dropped, opacity ≈ 0).
    fg_splats = create_new_splats_with_fg(cfg.fg_ckpt, device)

    # 3. Merge.
    merged = merge(bg_splats, fg_splats)

    n_bg  = bg_splats["means"].shape[0]
    n_fg  = fg_splats["means"].shape[0]
    n_tot = merged["means"].shape[0]
    print(f"[MERGE] bg={n_bg}  fg={n_fg}  total={n_tot}")

    # 4. Save — detach everything so the checkpoint is plain tensors.
    out_state = {k: v.detach().cpu() for k, v in merged.items()}
    torch.save({"splats": out_state}, cfg.out_ckpt)
    print(f"[MERGE] Saved merged checkpoint → {cfg.out_ckpt}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
