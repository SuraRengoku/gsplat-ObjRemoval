# type: ignore
"""
usage:
python Propagate_fill.py
    --data_dir 
    --result_dir
    --bg_ckpt
    --fg_ckpt
    --hole_mask_dir
    --hole_mask_ext
    --max_steps
    --enable_strategy

"""

import json
import inspect
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets.colmap import Dataset, Parser
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import assert_never
from utils import knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ---- data ----------------------------------------------------------------
    data_dir: str = "data/kitchen"
    data_factor: int = 4
    result_dir: str = "results/kitchen_fill"
    test_every: int = 8
    normalize_world_space: bool = True
    global_scale: float = 1.0

    # ---- checkpoints ---------------------------------------------------------
    # Path to the *background* and "foreground" checkpoint (.pt produced by OR_trainer / simple_trainer).
    bg_ckpt: str = ""
    fg_ckpt: str = ""

    # ---- hole masks ----------------------------------------------------------
    # Directory that contains per-view hole masks named by image stem, e.g.
    #   frame_0001.npy / frame_0001.png
    # Masks can be nested in subdirectories; files are discovered recursively.
    # If empty, fallback is full-image mask for every view.
    hole_mask_dir: str = ""
    # Preferred mask format when both exist for one stem.
    hole_mask_ext: Literal["auto", "npy", "png"] = "auto"

    # ---- new Gaussians -------------------------------------------------------
    # Number of new Gaussians to place inside the hole.
    num_new_gs: int = 10_000
    # Initial opacity (pre-sigmoid value: logit(0.1) ≈ -2.2).
    init_opacity: float = 0.1
    # Initial scale multiplier relative to scene scale.
    init_scale: float = 0.01

    # ---- optimizers ----------------------------------------------------------
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    sparse_grad: bool = False
    visible_adam: bool = False

    # ---- SDS -----------------------------------------------------------------
    # Hugging-Face model id for the inpainting pipeline.
    sd_model_id: str = "runwayml/stable-diffusion-inpainting"
    use_ip_adapter: bool = False
    ip_adapter_scale: float = 0.8
    # Text prompt fed to the inpainting model.
    prompt: str = "seamless background, high quality"
    negative_prompt: str = "artifacts, blurry, low quality"
    # Timestep sampling range expressed as a *fraction* of the scheduler's
    # total timesteps (T_max).  DreamFusion uses [0.02, 0.98]
    # the supervisor found [0.2, 0.6] produces sharper results by avoiding very high-noise
    # steps that contribute only blurry gradients.
    t_min: float = 0.2
    t_max: float = 0.6
    # SDS loss weight.
    sds_weight: float = 1.0
    # VAE latent scale factor (SD default 0.18215).
    vae_scale_factor: float = 0.18215
    # Classifier-free guidance scale.
    guidance_scale: float = 5.0

    # ---- training ------------------------------------------------------------
    max_steps: int = 3000
    batch_size: int = 1
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    random_bkgd: bool = False

    # ---- densification -------------------------------------------------------
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # IMPORTANT: In this script we optimize only `new_splats` while rendering
    # with `bg + new`. Densification strategies expect params/optimizers to be
    # aligned and can become inconsistent in this mixed setup. Keep disabled
    # by default for stability.
    enable_strategy: bool = False

    # ---- regularization ------------------------------------------------------
    # L2 penalty on scales of new Gaussians to keep them compact.
    scale_reg: float = 1e-3
    # L1 penalty on opacity to discourage early opaque blobs.
    opacity_reg: float = 1e-3

    # ---- optimization stabilizers -------------------------------------------
    # Warmup steps: freeze geometry (means/scales/quats), optimize only
    # appearance + opacity so SDS first learns texture/color before moving points.
    geom_warmup_steps: int = 800
    # Cap opacity during warmup to avoid early dark/opaque collapse.
    warmup_opacity_cap: float = 0.15

    # ---- misc ----------------------------------------------------------------
    save_steps: List[int] = field(default_factory=lambda: [1000, 2000, 3000])
    tb_every: int = 100
    tb_save_image: bool = False
    seed: int = 42

    def adjust_steps(self, factor: float) -> None:
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every     = int(strategy.reset_every     * factor)
            strategy.refine_every    = int(strategy.refine_every    * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter  = int(strategy.refine_stop_iter  * factor)
            strategy.refine_every      = int(strategy.refine_every      * factor)


# ---------------------------------------------------------------------------
# Gaussian utilities
# ---------------------------------------------------------------------------

def load_bg_splats(ckpt_path: str, device: str) -> torch.nn.ParameterDict:
    """Load background Gaussians from a .pt checkpoint and freeze all parameters."""
    print(f"[BG] Loading background checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    splats_state = ckpt["splats"]

    # Build a ParameterDict, freeze every parameter, and drop foreground_logits
    # (only needed during object-removal training, not during hole filling).
    EXCLUDE = {"foreground_logits"}
    splats = torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.to(device), requires_grad=False)
        for k, v in splats_state.items()
        if isinstance(v, torch.Tensor) and k not in EXCLUDE
    }).to(device)

    print(f"[BG] Loaded {len(splats['means'])} background Gaussians (frozen).")
    return splats


def create_new_splats(
    num: int,
    scene_scale: float,
    sh_degree: int,
    init_opacity: float,
    init_scale: float,
    parser: Parser,
    train_indices: List[int],
    hole_masks: Dict[int, Optional[Tensor]],
    ply_dir: Optional[str],
    device: str,
) -> torch.nn.ParameterDict:
    """
    Initialise *new* Gaussians from a coarse visual hull built by hole masks.

    We sample candidate 3D points in normalized scene bounds and keep points
    that project into hole masks across multiple training views.
    """
    print(f"[NEW] Initialising {num} new Gaussians from hole-mask visual hull.")

    def _random_means() -> torch.Tensor:
        return (torch.rand(num, 3, device=device) * 2 - 1) * scene_scale

    # Collect valid views with masks.
    view_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for parser_idx in train_indices:
        raw_mask = hole_masks.get(parser_idx, None)
        if raw_mask is None:
            continue

        mask = raw_mask.detach().cpu()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0.5)
        if mask.sum().item() == 0:
            continue

        cam_id = parser.camera_ids[parser_idx]
        K = torch.from_numpy(parser.Ks_dict[cam_id].astype(np.float32))
        camtoworld = torch.from_numpy(parser.camtoworlds[parser_idx].astype(np.float32))
        worldtocam = torch.linalg.inv(camtoworld)[:3, :]
        view_data.append((K, worldtocam, mask))

    def _save_points_ply(points: torch.Tensor, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pts = points.detach().cpu().numpy().astype(np.float32)
        with open(out_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

    if len(view_data) == 0:
        print("[NEW] WARNING: No valid hole masks found; falling back to random init.")
        means = _random_means()
    else:
        min_support = max(2, min(len(view_data), int(0.3 * len(view_data))))
        ratio_threshold = 0.55
        sample_batch = max(50_000, min(300_000, num * 8))
        max_rounds = 40

        accepted: List[torch.Tensor] = []
        accepted_count = 0

        for _ in range(max_rounds):
            pts = (torch.rand(sample_batch, 3) * 2 - 1) * scene_scale  # CPU candidates
            visible_count = torch.zeros(sample_batch, dtype=torch.int32)
            hit_count = torch.zeros(sample_batch, dtype=torch.int32)

            for K, worldtocam, mask in view_data:
                cam = pts @ worldtocam[:, :3].T + worldtocam[:, 3]
                z = cam[:, 2]
                valid_z = z > 1e-6

                x = cam[:, 0] / torch.clamp_min(z, 1e-6)
                y = cam[:, 1] / torch.clamp_min(z, 1e-6)
                u = K[0, 0] * x + K[0, 2]
                v = K[1, 1] * y + K[1, 2]

                h, w = int(mask.shape[0]), int(mask.shape[1])
                inside = (
                    valid_z
                    & (u >= 0) & (u < w)
                    & (v >= 0) & (v < h)
                )
                if inside.any():
                    visible_count[inside] += 1
                    ui = u[inside].long()
                    vi = v[inside].long()
                    hit = mask[vi, ui]
                    hit_count[inside] += hit.to(torch.int32)

            visible_ok = visible_count >= min_support
            ratio = hit_count.float() / torch.clamp_min(visible_count.float(), 1.0)
            keep = visible_ok & (hit_count >= min_support) & (ratio >= ratio_threshold)

            if keep.any():
                keep_pts = pts[keep]
                accepted.append(keep_pts)
                accepted_count += int(keep_pts.shape[0])

            if accepted_count >= num:
                break

        if accepted_count == 0:
            print("[NEW] WARNING: Visual hull produced 0 points; falling back to random init.")
            means = _random_means()
        else:
            cat = torch.cat(accepted, dim=0)
            if cat.shape[0] >= num:
                perm = torch.randperm(cat.shape[0])[:num]
                means = cat[perm].to(device)
            else:
                repeats = math.ceil(num / cat.shape[0])
                cat = cat.repeat(repeats, 1)[:num]
                jitter = torch.randn_like(cat) * (0.01 * scene_scale)
                means = (cat + jitter).to(device)

            print(
                f"[NEW] Visual hull accepted {accepted_count} candidates; "
                f"using {means.shape[0]} points."
            )

            if ply_dir:
                ply_root = Path(ply_dir)
                _save_points_ply(cat, ply_root / "visual_hull_candidates.ply")
                _save_points_ply(means.detach().cpu(), ply_root / "visual_hull_init_means.ply")
                print(
                    f"[NEW] Saved visual hull points to "
                    f"{ply_root / 'visual_hull_candidates.ply'} and "
                    f"{ply_root / 'visual_hull_init_means.ply'}"
                )

    dist = init_scale * scene_scale
    scales    = torch.full((num, 3), math.log(dist), device=device)
    quats     = F.normalize(torch.randn(num, 4, device=device), dim=-1)
    opacities = torch.full(
        (num,), math.log(init_opacity / (1.0 - init_opacity)), device=device
    )

    # SH colours: neutral gray DC band, higher bands = 0.
    neutral_rgb = torch.full((num, 3), 0.5, device=device)
    sh0 = rgb_to_sh(neutral_rgb).unsqueeze(1)
    shN = torch.zeros(num, (sh_degree + 1) ** 2 - 1, 3, device=device)

    splats = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(means),
        "scales":    torch.nn.Parameter(scales),
        "quats":     torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0":       torch.nn.Parameter(sh0),
        "shN":       torch.nn.Parameter(shN),
    })
    return splats

def create_new_splats_with_fg(ckpt_path: str, device: str) -> torch.nn.ParameterDict:
    """
    Load foreground Gaussians from a .pt checkpoint as the initial set of new
    (trainable) Gaussians for hole filling.

    All geometric and colour parameters are copied from the checkpoint and set
    as trainable.  Opacities are reset to nearly zero so the Gaussians are
    effectively invisible at the start of training and must be "grown" by the
    SDS optimiser rather than immediately polluting the render.

    Nearly-zero opacity in logit space: logit(ε) = log(ε / (1-ε)).
    Using ε = 0.005  →  logit ≈ -5.3.
    """
    NEAR_ZERO_OPACITY = 0.005  # sigmoid(-5.3) ≈ 0.005
    SCALE_LOG_OFFSET = 0.8     # shrink initial scale in log-space
    EXCLUDE = {"foreground_logits"}

    print(f"[FG] Loading foreground checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    splats_state = ckpt["splats"]

    # Load all parameters except foreground_logits; make them trainable.
    splats = torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.to(device), requires_grad=True)
        for k, v in splats_state.items()
        if isinstance(v, torch.Tensor) and k not in EXCLUDE
    }).to(device)

    # Keep geometry positions from FG init, but make appearance conservative.
    # 1) Near-zero opacity (in logit / pre-sigmoid space).
    num_gs = splats["means"].shape[0]
    init_logit = math.log(NEAR_ZERO_OPACITY / (1.0 - NEAR_ZERO_OPACITY))
    splats["opacities"] = torch.nn.Parameter(
        torch.full((num_gs,), init_logit, device=device),
        requires_grad=True,
    )

    # 2) Appearance reset: do not inherit removed-object colors.
    #    Use neutral gray DC term and zero higher-order SH.
    neutral_rgb = torch.full((num_gs, 3), 0.5, device=device)
    splats["sh0"] = torch.nn.Parameter(
        rgb_to_sh(neutral_rgb).unsqueeze(1),
        requires_grad=True,
    )
    splats["shN"] = torch.nn.Parameter(
        torch.zeros_like(splats["shN"]),
        requires_grad=True,
    )

    # 3) Conservative geometric init: keep rotations, slightly shrink scales.
    splats["quats"] = torch.nn.Parameter(
        F.normalize(splats["quats"].detach(), dim=-1),
        requires_grad=True,
    )
    splats["scales"] = torch.nn.Parameter(
        (splats["scales"].detach() - SCALE_LOG_OFFSET).clamp(min=-10.0, max=2.0),
        requires_grad=True,
    )

    print(
        f"[FG] Loaded {num_gs} foreground Gaussians as trainable new splats "
        f"(opacity reset, appearance reset, conservative scales)."
    )
    return splats

def merge_splats_for_render(
    bg: torch.nn.ParameterDict,
    new: torch.nn.ParameterDict,
) -> Dict[str, Tensor]:
    """
    Concatenate background (frozen) and new (trainable) Gaussians.
    The new Gaussians are appended after the background ones; gradient therefore
    flows only through the new-Gaussian slice of every merged tensor.
    """
    merged: Dict[str, Tensor] = {}
    for key in ("means", "scales", "quats", "opacities"):
        merged[key] = torch.cat([bg[key].detach(), new[key]], dim=0)

    bg_sh0 = bg["sh0"].detach()
    bg_shN = bg["shN"].detach()
    new_sh0 = new["sh0"]
    new_shN = new["shN"]

    # Pad higher-order SH bands if degrees differ between bg and new.
    max_bands = max(bg_shN.shape[1], new_shN.shape[1])
    if bg_shN.shape[1] < max_bands:
        pad = torch.zeros(bg_shN.shape[0], max_bands - bg_shN.shape[1], 3,
                          device=bg_shN.device)
        bg_shN = torch.cat([bg_shN, pad], dim=1)
    if new_shN.shape[1] < max_bands:
        pad = torch.zeros(new_shN.shape[0], max_bands - new_shN.shape[1], 3,
                          device=new_shN.device)
        new_shN = torch.cat([new_shN, pad], dim=1)

    merged["sh0"] = torch.cat([bg_sh0, new_sh0], dim=0)
    merged["shN"] = torch.cat([bg_shN, new_shN], dim=0)
    return merged


def create_new_splat_optimizers(
    splats: torch.nn.ParameterDict,
    cfg: Config,
    scene_scale: float,
) -> Dict[str, torch.optim.Optimizer]:
    """Create per-parameter optimizers for the new Gaussians only."""
    BS = cfg.batch_size
    optimizer_class = (
        torch.optim.SparseAdam if cfg.sparse_grad
        else SelectiveAdam       if cfg.visible_adam
        else torch.optim.Adam
    )

    param_lrs = [
        ("means",     cfg.means_lr * scene_scale),
        ("scales",    cfg.scales_lr),
        ("quats",     cfg.quats_lr),
        ("opacities", cfg.opacities_lr),
        ("sh0",       cfg.sh0_lr),
        ("shN",       cfg.shN_lr),
    ]

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, lr in param_lrs
    }
    return optimizers


# ---------------------------------------------------------------------------
# Per-view hole-mask loading
# ---------------------------------------------------------------------------

def load_hole_masks(
    parser: Parser,
    train_indices: List[int],
    device: str,
    mask_dir: str,
    mask_ext: Literal["auto", "npy", "png"] = "auto",
) -> Dict[int, Optional[Tensor]]:
    """
    Load per-view binary hole masks.
    1 = pixel belongs to the hole / inpaint region; 0 = background context.

    TODO: Implement this function.  Options:
      - Project the 3-D bounding box of the removed object into each camera.
      - Dilate the OR_trainer foreground masks already saved as .npy files.
      - Use SAM prompts on the rendered background to detect the empty region.

    Returns:
        Dict mapping image_id → float Tensor [H, W], or None (= full image).
    """
    if not mask_dir:
        print("[MASK] hole_mask_dir is empty — using full-image masks.")
        return {idx: None for idx in train_indices}

    root = Path(mask_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"[MASK] hole_mask_dir not found: {mask_dir}")

    # Build stem -> file lookup for npy/png recursively.
    npy_map: Dict[str, Path] = {}
    png_map: Dict[str, Path] = {}
    for p in root.rglob("*.npy"):
        npy_map.setdefault(p.stem, p)
    for p in root.rglob("*.png"):
        png_map.setdefault(p.stem, p)

    if not npy_map and not png_map:
        raise RuntimeError(
            f"[MASK] No .npy/.png files found under: {mask_dir}"
        )

    def _pick_file(stem: str) -> Optional[Path]:
        if mask_ext == "npy":
            return npy_map.get(stem)
        if mask_ext == "png":
            return png_map.get(stem)
        # auto: prefer npy, then png
        return npy_map.get(stem) or png_map.get(stem)

    def _load_binary_mask(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
        else:
            arr = imageio.imread(path)

        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return (arr > 0.5).astype(np.float32)

    hole_masks: Dict[int, Optional[Tensor]] = {}
    missing = 0
    loaded = 0

    for parser_idx in train_indices:
        image_name = parser.image_names[parser_idx]
        stem = Path(image_name).stem
        mask_path = _pick_file(stem)

        if mask_path is None:
            hole_masks[parser_idx] = None
            missing += 1
            continue

        mask_np = _load_binary_mask(mask_path)
        hole_masks[parser_idx] = torch.from_numpy(mask_np).to(device)
        loaded += 1

    print(
        f"[MASK] Loaded {loaded} masks from {mask_dir} "
        f"(missing: {missing}, total train views: {len(train_indices)})."
    )
    return hole_masks


# ---------------------------------------------------------------------------
# SDS Optimiser (wraps Stable Diffusion Inpainting)
# ---------------------------------------------------------------------------

class SDSInpaintOptimizer:
    """
    Computes SDS gradients using a frozen SD-Inpainting UNet.

    SDS update direction in latent space::

        d_SDS = w(t) · (ε_θ(z_t, t, y, mask) - ε)

    where ε_θ is the noise predicted by the inpainting UNet conditioned on
    the text prompt y and the inpaint mask, and ε is the actual noise added.
    The gradient is masked to zero outside the hole region before being
    back-propagated through the differentiable rasteriser.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype = torch.float16,
        use_ip_adapter: bool = False,
        ip_adapter_scale: float = 1.0,
    ) -> None:
        from diffusers import StableDiffusionInpaintPipeline

        print(f"[SDS] Loading SD Inpainting pipeline: {model_id}")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        
        self.use_ip_adapter = use_ip_adapter
        if self.use_ip_adapter:
            print("[SDS] Loading IP-Adapter into pipeline...")
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            pipe.set_ip_adapter_scale(ip_adapter_scale)
            self.image_encoder = pipe.image_encoder
            self.feature_extractor = pipe.feature_extractor

        pipe.set_progress_bar_config(disable=True)

        self.vae          = pipe.vae
        self.unet         = pipe.unet
        self.tokenizer    = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler    = pipe.scheduler
        self.pipe         = pipe
        self.device       = device
        self.dtype        = dtype

        # Cache the αᵢ cumulative-product schedule as a device tensor.
        self.alphas_cumprod: Tensor = self.scheduler.alphas_cumprod.to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text(self, text: str) -> Tensor:
        ids = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(ids)[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_text_embeddings(
        self, prompt: str, negative_prompt: str
    ) -> Tuple[Tensor, Tensor]:
        """Pre-encode text prompts (call once; re-use every SDS step)."""
        return self._encode_text(prompt), self._encode_text(negative_prompt)

    def sds_loss(
        self,
        rendered: Tensor,
        mask: Tensor,
        text_embeds: Tensor,
        uncond_embeds: Tensor,
        vae_scale: float,
        t_min: float,
        t_max: float,
        guidance_scale: float = 7.5,
        return_pixel_map: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute the SDS loss for one rendered view.

        Args:
            rendered:       Float tensor [B, H, W, 3] ∈ [0, 1].
            mask:           Float tensor [B, H, W] ∈ {0, 1}  (1 = inpaint).
            text_embeds:    Conditioned text embeddings  [1, L, D].
            uncond_embeds:  Unconditional text embeddings [1, L, D].
            vae_scale:      VAE latent scale factor.
            t_min / t_max:  Timestep range as *fractions* of T_max, e.g. 0.2 / 0.6.
                            Converted to integer indices internally.
            guidance_scale: CFG scale.

        Returns:
            Scalar SDS loss.  Its gradient w.r.t. ``rendered`` equals
            d_SDS masked to the inpaint region.
        """
        B, H, W, _ = rendered.shape
        latent_h, latent_w = H // 8, W // 8

        # ---- 1. Permute rendered to [B, 3, H, W] ---------------------------
        img_bchw = rendered.permute(0, 3, 1, 2).contiguous()  # grad flows here

        # ---- 2. Encode rendered image to VAE latent ------------------------
        #   We encode twice: once with gradient (z) and once without (z_ref
        #   used inside the noising step, so no second-order gradient leaks).
        with torch.no_grad():
            z_ref = (
                self.vae.encode(img_bchw.to(self.dtype) * 2 - 1)
                .latent_dist.sample() * vae_scale
            )  # [B, 4, H/8, W/8]

        # Gradient-carrying encode.
        z = (
            self.vae.encode(img_bchw.to(self.dtype) * 2 - 1)
            .latent_dist.sample() * vae_scale
        )  # [B, 4, H/8, W/8]

        # ---- 3. Prepare inpainting UNet inputs -----------------------------
        #   SD-Inpainting UNet expects a 9-channel input:
        #     [noisy_latent (4) | mask_latent (1) | masked_image_latent (4)]

        # Downsample mask to latent resolution.
        mask_bchw = mask.unsqueeze(1).float()               # [B, 1, H, W]
        mask_latent = F.interpolate(
            mask_bchw, size=(latent_h, latent_w), mode="nearest"
        ).to(self.dtype)                                     # [B, 1, H/8, W/8]

        # Zero out the inpaint region in the original image, then encode.
        masked_img = img_bchw.to(self.dtype) * (1.0 - mask_bchw.to(self.dtype))
        # masked_img = img_bchw.to(self.dtype)
        with torch.no_grad():
            masked_latent = (
                self.vae.encode(masked_img * 2 - 1)
                .latent_dist.sample() * vae_scale
            )  # [B, 4, H/8, W/8]

        # ---- 4. Sample timestep and add noise ------------------------------
        # Convert fractional t_min / t_max to integer indices.
        T = self.scheduler.config.num_train_timesteps  # typically 1000
        t_lo = max(1,   int(t_min * T))
        t_hi = min(T-1, int(t_max * T))
        t = torch.randint(t_lo, t_hi + 1, (B,), device=self.device, dtype=torch.long)
        noise   = torch.randn_like(z_ref)
        alpha_t = self.alphas_cumprod[t].view(B, 1, 1, 1).to(self.dtype)
        z_t     = torch.sqrt(alpha_t) * z_ref + torch.sqrt(1.0 - alpha_t) * noise

        # ---- 5. UNet noise prediction with classifier-free guidance --------
        #   Stack uncond + cond embeddings; duplicate latent inputs.
        text_in   = torch.cat([uncond_embeds, text_embeds], dim=0).to(self.dtype)
        unet_in   = torch.cat([z_t, mask_latent, masked_latent], dim=1)  # [B, 9, ...]
        unet_in   = unet_in.repeat(2, 1, 1, 1)                           # [2B, 9, ...]

        with torch.no_grad():
            added_cond_kwargs = None
            if getattr(self, "use_ip_adapter", False):
                from torchvision.transforms.functional import to_pil_image
                # Convert masked image to PIL for CLIP feature extractor
                pil_imgs = [to_pil_image(m) for m in masked_img.cpu()]
                ip_img_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=pil_imgs,
                    ip_adapter_image_embeds=None,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                added_cond_kwargs = {"image_embeds": ip_img_embeds}

            noise_pred = self.unet(
                unet_in,
                t.repeat(2),
                encoder_hidden_states=text_in,
                added_cond_kwargs=added_cond_kwargs,
            ).sample  # [2B, 4, H/8, W/8]

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
        noise_pred_guided = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        )

        # ---- 6. Compute SDS gradient direction in latent space -------------
        #
        #   d_SDS = w(t) · (ε_θ − ε)
        #
        #   Weight w(t) = (1 − ᾱ_t) follows the original SDS formulation
        #   (Poole et al., 2022).  SNR-weighted variants can be substituted.
        w       = (1.0 - alpha_t)                      # [B, 1, 1, 1]
        sds_grad = w * (noise_pred_guided.float() - noise.float())  # [B, 4, H/8, W/8]

        # Zero gradient outside the hole region.
        sds_grad = sds_grad * mask_latent.float()

        # ---- 7. Stop-gradient pseudo-loss ----------------------------------
        #   Produces  ∇_z L = d_SDS  without computing second-order gradients
        #   through the UNet (which is frozen and in no-grad mode).
        loss = (sds_grad.detach() * z).sum()

        if return_pixel_map:
            with torch.no_grad():
                pmap = torch.linalg.norm(sds_grad.float(), dim=1, keepdim=True)  # [B,1,lH,lW]
                pmap = F.interpolate(pmap, size=(H, W), mode="bilinear", align_corners=False)
                pmap = pmap.squeeze(1)  # [B, H, W]
            return loss, pmap
        return loss


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_debug_sds(
    loss_map: Tensor,  # [H, W] float — per-pixel SDS gradient magnitude
    mask: Tensor,      # [H, W] float in [0, 1]
    path: str,
) -> None:
    """
    Save a side-by-side diagnostic PNG:
      LEFT:  SDS gradient magnitude heatmap (inferno colormap; brighter = stronger signal)
      RIGHT: binary hole mask used in this step
    """
    loss_np = loss_map.detach().cpu().float().numpy()
    mask_np = mask.detach().cpu().float().numpy()

    vmax = loss_np.max()
    loss_norm = loss_np / vmax if vmax > 0 else loss_np

    cmap     = plt.get_cmap("inferno")
    heatmap  = (cmap(loss_norm)[..., :3] * 255).astype(np.uint8)
    mask_vis = (np.stack([mask_np] * 3, axis=-1) * 255).astype(np.uint8)
    imageio.imwrite(path, np.concatenate([heatmap, mask_vis], axis=1))


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, cfg: Config, device: str = "cuda") -> None:
        set_random_seed(cfg.seed)
        self.cfg    = cfg
        self.device = device
        self._warned_missing_mask = False

        # ---- output directories ------------------------------------------
        for subdir in ("ckpts", "stats", "renders", "ply", "debug"):
            os.makedirs(os.path.join(cfg.result_dir, subdir), exist_ok=True)
        self.ckpt_dir   = os.path.join(cfg.result_dir, "ckpts")
        self.stats_dir  = os.path.join(cfg.result_dir, "stats")
        self.render_dir = os.path.join(cfg.result_dir, "renders")
        self.ply_dir    = os.path.join(cfg.result_dir, "ply")
        self.debug_dir  = os.path.join(cfg.result_dir, "debug")
        self.writer     = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb"))
        self.sds_loss_history = []

        # ---- dataset -------------------------------------------------------
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset    = Dataset(self.parser, split="train")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"Scene scale: {self.scene_scale:.4f}")

        # ---- Hole masks (per view) ------------------------------------------
        self.hole_masks: Dict[int, Optional[Tensor]] = load_hole_masks(
            self.parser,
            self.trainset.indices,
            device,
            cfg.hole_mask_dir,
            cfg.hole_mask_ext,
        )

        # ---- Gaussians -------------------------------------------------------
        # 1. Background (frozen).
        assert cfg.bg_ckpt, "bg_ckpt must point to the background .pt checkpoint!"
        self.bg_splats = load_bg_splats(cfg.bg_ckpt, device)
        self.bg_count = self.bg_splats["means"].shape[0]

        # 2. New Gaussians (trainable).
        if cfg.hole_mask_dir:
            self.new_splats = create_new_splats(
                num=cfg.num_new_gs,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                init_opacity=cfg.init_opacity,
                init_scale=cfg.init_scale,
                parser=self.parser,
                train_indices=self.trainset.indices.tolist(),
                hole_masks=self.hole_masks,
                ply_dir=self.ply_dir,
                device=device,
            )
        elif cfg.fg_ckpt:
            print("[NEW] hole_mask_dir not set. Falling back to FG-based init.")
            self.new_splats = create_new_splats_with_fg(cfg.fg_ckpt, device)
        else:
            raise ValueError(
                "Need hole_mask_dir for visual-hull init, or fg_ckpt for FG fallback init."
            )

        # 3. Optimizers — only for the new Gaussians.
        self.optimizers = create_new_splat_optimizers(
            self.new_splats, cfg, self.scene_scale
        )

        # ---- Densification strategy -----------------------------------------
        self._rebuild_combined_splats()
        self.strategy_state = None
        self._default_strategy_pre_has_packed = False
        self._default_strategy_post_has_packed = False
        self._strategy_cap_warned = False
        # Internal hard cap to avoid densification explosion and OOM.
        # Intentionally not exposed as CLI arg.
        self._max_new_gs_cap = 800_000
        if cfg.enable_strategy:
            cfg.strategy.check_sanity(self.new_splats, self.optimizers)
            if isinstance(cfg.strategy, DefaultStrategy):
                self.strategy_state = cfg.strategy.initialize_state(
                    scene_scale=self.scene_scale
                )
                self._default_strategy_pre_has_packed = (
                    "packed" in inspect.signature(cfg.strategy.step_pre_backward).parameters
                )
                self._default_strategy_post_has_packed = (
                    "packed" in inspect.signature(cfg.strategy.step_post_backward).parameters
                )
            elif isinstance(cfg.strategy, MCMCStrategy):
                self.strategy_state = cfg.strategy.initialize_state()
            else:
                assert_never(cfg.strategy)

        # ---- SDS module ----------------------------------------------------
        self.sds = SDSInpaintOptimizer(
            model_id=cfg.sd_model_id,
            device=device,
            use_ip_adapter=getattr(cfg, "use_ip_adapter", False),
            ip_adapter_scale=getattr(cfg, "ip_adapter_scale", 1.0),
        )
        # Pre-encode text — constant throughout training.
        self.text_embeds, self.uncond_embeds = self.sds.get_text_embeddings(
            cfg.prompt, cfg.negative_prompt
        )
        print(f'[SDS] Prompt: "{cfg.prompt}"')

    def _rebuild_combined_splats(self) -> None:
        """
        Merge bg + new parameters into self.combined_splats.
        Called at init and after every densification step.
        """
        merged = merge_splats_for_render(self.bg_splats, self.new_splats)
        self.combined_splats = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v, requires_grad=v.requires_grad)
            for k, v in merged.items()
        })

    def _get_mask(self, parser_index: int, height: int, width: int) -> Tensor:
        """Return float mask [1, H, W] (1 = hole). Missing mask -> zero mask."""
        raw = self.hole_masks.get(parser_index, None)
        if raw is None:
            if not self._warned_missing_mask:
                print(
                    "[MASK] WARNING: missing mask found. "
                    "Using zero mask for that view (skip update)."
                )
                self._warned_missing_mask = True
            return torch.zeros(1, height, width, device=self.device)
        mask = raw.float().unsqueeze(0)
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(
                mask.unsqueeze(0), size=(height, width), mode="nearest"
            ).squeeze(0)
        return mask.to(self.device)

    @torch.no_grad()
    def _new_gs_in_hole_current_view(
        self,
        parser_index: int,
        camtoworld: Tensor,
        K: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Return bool [N_new]: projected new Gaussian centers inside current hole mask."""
        raw = self.hole_masks.get(parser_index, None)
        n = self.new_splats["means"].shape[0]
        if raw is None:
            return torch.zeros(n, dtype=torch.bool, device=self.device)

        mask = raw.float()
        if mask.shape != (height, width):
            mask = F.interpolate(mask[None, None], size=(height, width), mode="nearest")[0, 0]
        mask = mask > 0.5

        means = self.new_splats["means"].detach()
        worldtocam = torch.linalg.inv(camtoworld)[:3, :]
        cam = means @ worldtocam[:, :3].T + worldtocam[:, 3]
        z = cam[:, 2]
        valid_z = z > 1e-6

        x = cam[:, 0] / torch.clamp_min(z, 1e-6)
        y = cam[:, 1] / torch.clamp_min(z, 1e-6)
        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]

        inside = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        keep = torch.zeros(n, dtype=torch.bool, device=self.device)
        if inside.any():
            idx = torch.nonzero(inside, as_tuple=False).squeeze(1)
            ui = u[inside].long()
            vi = v[inside].long()
            keep[idx] = mask[vi, ui]
        return keep

    def _mask_new_grads_to_hole(
        self,
        parser_index: int,
        camtoworld: Tensor,
        K: Tensor,
        height: int,
        width: int,
    ) -> None:
        """Zero grad for new GSs whose projected centers are outside current hole mask."""
        keep = self._new_gs_in_hole_current_view(
            parser_index=parser_index,
            camtoworld=camtoworld,
            K=K,
            height=height,
            width=width,
        ).float()

        for key in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            grad = self.new_splats[key].grad
            if grad is None:
                continue
            mask = keep
            while mask.ndim < grad.ndim:
                mask = mask.unsqueeze(-1)
            grad.mul_(mask)

    def _rebuild_new_splats_from_keep_mask(self, keep_mask: Tensor) -> None:
        """Keep a subset of new splats and rebuild optimizers/strategy state."""
        keep_mask = keep_mask.bool()
        assert keep_mask.ndim == 1 and keep_mask.shape[0] == len(self.new_splats["means"])

        self.new_splats = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v[keep_mask].detach().clone(), requires_grad=True)
            for k, v in self.new_splats.items()
        }).to(self.device)

        self.optimizers = create_new_splat_optimizers(
            self.new_splats, self.cfg, self.scene_scale
        )

        if self.cfg.enable_strategy:
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.strategy_state = self.cfg.strategy.initialize_state(
                    scene_scale=self.scene_scale
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.strategy_state = self.cfg.strategy.initialize_state()

    def _enforce_new_gs_cap_if_needed(self) -> None:
        """Prune low-opacity new GSs to satisfy max_new_gs cap."""
        cap = self._max_new_gs_cap
        if cap <= 0:
            return

        n_new = len(self.new_splats["means"])
        if n_new <= cap:
            return

        with torch.no_grad():
            keep_count = int(cap)
            scores = torch.sigmoid(self.new_splats["opacities"]).detach()
            topk = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
            keep_mask[topk] = True

        print(
            f"[CAP] Pruning new GSs from {n_new} -> {keep_count} "
            f"(keeping highest opacity)."
        )
        self._rebuild_new_splats_from_keep_mask(keep_mask)

    def _extract_new_strategy_info(self, rast_info: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Extract rasterization info for new Gaussians only."""
        bg_count = self.bg_count

        info: Dict[str, Tensor] = {}

        key_for_gradient = "means2d"
        if isinstance(self.cfg.strategy, DefaultStrategy):
            key_for_gradient = self.cfg.strategy.key_for_gradient

        if self.cfg.packed:
            gaussian_ids = rast_info["gaussian_ids"]
            keep = gaussian_ids >= bg_count
            info["gaussian_ids"] = gaussian_ids[keep] - bg_count
            info["radii"] = rast_info["radii"][keep]
            info[key_for_gradient] = rast_info[key_for_gradient][keep]
        else:
            info["radii"] = rast_info["radii"][:, bg_count:]
            info[key_for_gradient] = rast_info[key_for_gradient][:, bg_count:]
            info["gaussian_ids"] = torch.empty(
                0, dtype=torch.long, device=info["radii"].device
            )

        return info

    def _inject_new_gradient_for_strategy(
        self,
        strategy_info: Dict[str, Tensor],
        rast_info: Dict[str, Tensor],
    ) -> None:
        """Attach new-only 2D gradient tensor to strategy_info[key].grad."""
        bg_count = self.bg_count
        key_for_gradient = "means2d"
        if isinstance(self.cfg.strategy, DefaultStrategy):
            key_for_gradient = self.cfg.strategy.key_for_gradient

        full_grad = rast_info[key_for_gradient].grad
        assert full_grad is not None, f"Missing grad for {key_for_gradient}."

        if self.cfg.packed:
            gaussian_ids = rast_info["gaussian_ids"]
            keep = gaussian_ids >= bg_count
            grad_new = full_grad[keep]
        else:
            grad_new = full_grad[:, bg_count:]

        grad_holder = torch.zeros_like(strategy_info[key_for_gradient], requires_grad=True)
        grad_holder.grad = grad_new
        strategy_info[key_for_gradient] = grad_holder

    def rasterize(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
    ) -> Tensor:
        """Render the combined (bg + new) Gaussians.  Returns [B, H, W, 3]."""
        merged = merge_splats_for_render(self.bg_splats, self.new_splats)

        renders, _, rast_info = rasterization(
            means=merged["means"],
            quats=merged["quats"] / merged["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(merged["scales"]),
            opacities=torch.sigmoid(merged["opacities"]),
            colors=torch.cat([merged["sh0"], merged["shN"]], dim=1),
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            packed=self.cfg.packed,
        )
        return renders[..., :3], rast_info  # renders [B, H, W, 3] 
        
    def train(self) -> None:
        """
        Hole-filling for 3DGS via SDS Optimization with Stable Diffusion Inpainting.

        Pipeline:
        1. Load background checkpoint (all parameters frozen).
        2. Initialize new Gaussians inside the hole region  [TODO: implement or maybe deprecate].
		   Initialize new Gaussians by setting the original foreground Gaussians' opacity close to 0
        3. For each training step:
            a. Render RGB images from COLMAP camera poses.
            b. Retrieve per-view hole masks                
            c. Encode rendered image to latent space, add noise.
            d. Run SD Inpainting UNet (no-grad) to get noise prediction.
            e. Compute SDS loss / gradient, apply hole mask.
            f. Backpropagate into new Gaussians only.
            g. Densification + boundary regularization.
        4. Save checkpoint.
        """
        cfg        = self.cfg
        device     = self.device
        max_steps  = cfg.max_steps
        global_tic = time.time()

        sh_degree_fn = lambda step: min(cfg.sh_degree, step // cfg.sh_degree_interval)

        data_iter = iter(self.trainset)
        pbar      = tqdm.tqdm(range(max_steps), desc="SDS Fill")

        for step in pbar:
            # ---- fetch one training view -----------------------------------
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.trainset)
                data      = next(data_iter)

            image_id_raw = data["image_id"]
            image_id: int = (
                int(image_id_raw.item())
                if hasattr(image_id_raw, "item")
                else int(image_id_raw)
            )
            parser_index: int = int(self.trainset.indices[image_id])
            camtoworlds     = data["camtoworld"].unsqueeze(0).to(device)  # [1,4,4]
            Ks              = data["K"].unsqueeze(0).to(device)           # [1,3,3]
            pixels          = data["image"].unsqueeze(0).to(device)       # [1,H,W,3]
            height, width   = pixels.shape[1], pixels.shape[2]

            for opt in self.optimizers.values():
                opt.zero_grad()

            # ---- render all Gaussians (bg frozen + new trainable) ----------
            sh_degree = sh_degree_fn(step)
            colors, rast_info = self.rasterize(camtoworlds, Ks, width, height, sh_degree)
            # colors: [1, H, W, 3]  ∈ [0, 1],  grad flows through new_splats

            # ---- per-view hole mask ----------------------------------------
            mask_hw = self._get_mask(parser_index, height, width)  # [1, H, W]

            # ---- SDS loss using SD-Inpainting prior ------------------------
            #
            #  The full SDS cycle (executed inside sds_loss()):
            #
            #  1. Encode rendered image I into latent z via the VAE.
            #  2. Sample a random timestep t ∈ [t_min, t_max].
            #  3. Add noise:
            #       z_t = √ᾱ_t · z  +  √(1−ᾱ_t) · ε,   ε ~ N(0,I)
            #  4. Concatenate  [z_t | mask | masked-image-z]  →  9-ch input.
            #  5. Run the frozen UNet with text prompt (CFG):
            #       ε_θ = UNet(z_t, t, prompt, mask)
            #  6. Compute the SDS pseudo-gradient:
            #       d_SDS = w(t) · (ε_θ − ε)           w(t) = (1 − ᾱ_t)
            #  7. Zero d_SDS outside the hole mask.
            #  8. Back-propagate through the VAE encoder and rasteriser,
            #     updating only the new Gaussians.
            #
            sds_loss = cfg.sds_weight * self.sds.sds_loss(
                rendered=colors,
                mask=mask_hw,
                text_embeds=self.text_embeds,
                uncond_embeds=self.uncond_embeds,
                vae_scale=cfg.vae_scale_factor,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                guidance_scale=cfg.guidance_scale,
            )
            self.sds_loss_history.append(sds_loss.item())

            # ---- boundary regularization -----------------------------------
            #   Penalise large scales so new Gaussians stay compact and do
            #   not "leak" into the frozen background region.
            reg_loss = cfg.scale_reg * (
                torch.exp(self.new_splats["scales"]).norm(dim=-1).mean()
            )

            # Encourage sparse/translucent new splats, especially early on.
            opacity_vals = torch.sigmoid(self.new_splats["opacities"])
            opacity_loss = cfg.opacity_reg * opacity_vals.mean()

            strategy_info = None
            n_new_gs = len(self.new_splats["means"])
            strategy_allowed = (
                cfg.enable_strategy
                and (self._max_new_gs_cap <= 0 or n_new_gs < self._max_new_gs_cap)
            )
            if (
                cfg.enable_strategy
                and not strategy_allowed
                and not self._strategy_cap_warned
            ):
                print(
                    f"[STRATEGY] max_new_gs={self._max_new_gs_cap} reached "
                    f"(current={n_new_gs}). Densification disabled for remaining steps."
                )
                self._strategy_cap_warned = True

            if strategy_allowed and isinstance(cfg.strategy, DefaultStrategy):
                strategy_info = self._extract_new_strategy_info(rast_info)
                strategy_info["width"] = width
                strategy_info["height"] = height
                strategy_info["n_cameras"] = 1

                # For pre-backward, retain grad on the original rasterization tensor.
                pre_info = {cfg.strategy.key_for_gradient: rast_info[cfg.strategy.key_for_gradient]}
                pre_kwargs = dict(
                    params=self.new_splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=pre_info,
                )
                if self._default_strategy_pre_has_packed:
                    pre_kwargs["packed"] = cfg.packed
                cfg.strategy.step_pre_backward(**pre_kwargs)

            total_loss = sds_loss + reg_loss + opacity_loss
            total_loss.backward()

            self._mask_new_grads_to_hole(
                parser_index=parser_index,
                camtoworld=camtoworlds[0],
                K=Ks[0],
                height=height,
                width=width,
            )

            # Geometry warmup: do not move/split shape too early under noisy SDS.
            if step < cfg.geom_warmup_steps:
                for key in ("means", "scales", "quats"):
                    grad = self.new_splats[key].grad
                    if grad is not None:
                        grad.zero_()

            # ---- optimizer step (new Gaussians only) -----------------------
            for opt in self.optimizers.values():
                opt.step()

            # Keep opacity bounded during warmup to avoid black blob collapse.
            if step < cfg.geom_warmup_steps:
                cap = min(max(cfg.warmup_opacity_cap, 1e-4), 0.999)
                cap_logit = math.log(cap / (1.0 - cap))
                with torch.no_grad():
                    self.new_splats["opacities"].clamp_(max=cap_logit)

            if strategy_allowed and isinstance(cfg.strategy, DefaultStrategy):
                assert strategy_info is not None
                self._inject_new_gradient_for_strategy(strategy_info, rast_info)
                post_kwargs = dict(
                    params=self.new_splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=strategy_info,
                )
                if self._default_strategy_post_has_packed:
                    post_kwargs["packed"] = cfg.packed
                cfg.strategy.step_post_backward(**post_kwargs)
            elif strategy_allowed and isinstance(cfg.strategy, MCMCStrategy):
                cfg.strategy.step_post_backward(
                    params=self.new_splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info={},
                    lr=cfg.means_lr * self.scene_scale,
                )

            # Hard anti-explosion guard for memory stability.
            self._enforce_new_gs_cap_if_needed()

            # ---- refresh merged state for ckpt/export ----------------------
            self._rebuild_combined_splats()

            # ---- logging ---------------------------------------------------
            pbar.set_postfix(
                sds=f"{sds_loss.item():.4f}",
                reg=f"{reg_loss.item():.4f}",
                opa=f"{opacity_loss.item():.4f}",
                ngs=len(self.new_splats["means"]),
            )

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/sds_loss",   sds_loss.item(),   step)
                self.writer.add_scalar("train/reg_loss",   reg_loss.item(),   step)
                self.writer.add_scalar("train/opacity_loss", opacity_loss.item(), step)
                self.writer.add_scalar("train/num_new_gs", len(self.new_splats["means"]), step)
                if cfg.tb_save_image:
                    canvas = colors[0].detach().cpu().numpy()
                    self.writer.add_image("train/render", canvas.transpose(2, 0, 1), step)
                self.writer.flush()
                
                # Update SDS Loss curve intermittently
                plt.figure(figsize=(10, 5))
                plt.plot(self.sds_loss_history, label="SDS Loss", linewidth=1.0, alpha=0.4, color="tab:blue")
                if len(self.sds_loss_history) >= 50:
                    smoothed = np.convolve(self.sds_loss_history, np.ones(50)/50, mode='valid')
                    shift = 50 // 2 - 1
                    plt.plot(np.arange(shift, len(smoothed) + shift), smoothed, label="Smoothed (MA=50)", linewidth=2.0, color="tab:red")
                plt.xlabel("Step")
                plt.ylabel("SDS Loss")
                plt.title("SDS Loss Progression")
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(self.stats_dir, "sds_loss_curve.png"), bbox_inches="tight")
                plt.close()

            # ---- periodic render dump + SDS debug heatmap -----------------
            if step % 50 == 0:
                render_np = colors[0].detach().clamp(0.0, 1.0).cpu().numpy()
                render_u8 = (render_np * 255.0).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(self.render_dir, f"render_{step:05d}_img{parser_index:04d}.png"),
                    render_u8,
                )

                # SDS gradient magnitude heatmap + mask side-by-side
                with torch.no_grad():
                    _, sds_pixel_map = self.sds.sds_loss(
                        rendered=colors.detach(),
                        mask=mask_hw,
                        text_embeds=self.text_embeds,
                        uncond_embeds=self.uncond_embeds,
                        vae_scale=cfg.vae_scale_factor,
                        t_min=cfg.t_min,
                        t_max=cfg.t_max,
                        guidance_scale=cfg.guidance_scale,
                        return_pixel_map=True,
                    )
                _save_debug_sds(
                    sds_pixel_map[0],  # [H, W]
                    mask_hw[0],         # [H, W]
                    os.path.join(self.debug_dir, f"debug_{step:05d}_img{parser_index:04d}.png"),
                )

            # ---- checkpoint ------------------------------------------------
            if step + 1 in cfg.save_steps or step == max_steps - 1:
                elapsed = time.time() - global_tic
                stats   = {
                    "step": step,
                    "elapsed_s": elapsed,
                    "num_new_gs": len(self.new_splats["means"]),
                }
                with open(
                    os.path.join(self.stats_dir, f"stats_{step:05d}.json"), "w"
                ) as f:
                    json.dump(stats, f, indent=2)

                # Save a high-quality SDS Loss curve on checkpoints
                plt.figure(figsize=(10, 5))
                plt.plot(self.sds_loss_history, label="SDS Loss", linewidth=1.0, alpha=0.4, color="tab:blue")
                if len(self.sds_loss_history) >= 50:
                    smoothed = np.convolve(self.sds_loss_history, np.ones(50)/50, mode='valid')
                    shift = 50 // 2 - 1
                    plt.plot(np.arange(shift, len(smoothed) + shift), smoothed, label="Smoothed (MA=50)", linewidth=2.0, color="tab:red")
                plt.xlabel("Step")
                plt.ylabel("SDS Loss")
                plt.title("SDS Loss Progression")
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(self.stats_dir, "sds_loss_curve.png"), bbox_inches="tight")
                plt.close()

                # Save new Gaussians only.
                torch.save(
                    {"step": step, "splats": self.new_splats.state_dict()},
                    os.path.join(self.ckpt_dir, f"new_gs_{step:05d}.pt"),
                )
                # Save the combined (bg + new) checkpoint for downstream use.
                combined_state = {
                    k: self.combined_splats[k].detach().cpu()
                    for k in self.combined_splats
                }
                torch.save(
                    {"step": step, "splats": combined_state},
                    os.path.join(self.ckpt_dir, f"combined_{step:05d}.pt"),
                )
                print(
                    f"\n[ckpt] step={step}  "
                    f"new_gs={len(self.new_splats['means'])}  "
                    f"time={elapsed:.1f}s"
                )

        print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: Config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = Runner(cfg, device=device)
    runner.train()


def _normalize_cli_underscores(argv: List[str]) -> List[str]:
    """
    Convert long-option names from underscore style to hyphen style.
    Example:
      --hole_mask_dir=... -> --hole-mask-dir=...
      --hole_mask_dir ... -> --hole-mask-dir ...
    Values are left untouched.
    """
    normalized: List[str] = []
    for arg in argv:
        if not arg.startswith("--"):
            normalized.append(arg)
            continue

        if "=" in arg:
            key, value = arg.split("=", 1)
            normalized.append(key.replace("_", "-") + "=" + value)
        else:
            normalized.append(arg.replace("_", "-"))
    return normalized


if __name__ == "__main__":
    cfg = tyro.cli(Config, args=_normalize_cli_underscores(sys.argv[1:]))
    main(cfg)

