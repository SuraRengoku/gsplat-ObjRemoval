# type: ignore
"""
SDS-based 3DGS Hole Filling with Sparse COLMAP View Masks.

Usage:
    python examples/sds_hole_filling.py \
        --data_dir data/Tree \
        --mask_dir data/Tree/holeMask \
        --ckpt results/Tree/ckpts/ckpt_29999_rank0.pt \
        --result_dir results/Tree_SDS \
        --prompt "a photo of a lush green forest" \
        --max_steps 5000 \
        --batch_size 2

Mask files are expected as .npy binary arrays (H, W) with values in {0, 1},
one per COLMAP training image, named to match the image filenames.
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import gaussian_blur

# gsplat rendering
from gsplat.rendering import rasterization

# Optional: diffusers for real SD. Falls back to mock if unavailable.
try:
    from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline

    _DIFFUSERS_AVAILABLE = True
except ImportError:
    _DIFFUSERS_AVAILABLE = False


# =============================================================================
# Dataset / Loader
# =============================================================================


class ColmapMaskDataset:
    """
    Wraps a loaded gsplat Parser-style dataset and pairs each training view
    with a binary hole mask loaded from *mask_dir*.

    Directory layout assumed::

        mask_dir/
            <image_stem>.npy   # one mask per training image, shape (H, W), dtype uint8/bool

    If a mask file is missing for a view, a zero mask (= no hole) is used.
    """

    def __init__(
        self,
        data_dir: str,
        mask_dir: str,
        data_factor: int = 1,
        dilation_iters: int = 5,
        blur_kernel: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.dilation_iters = dilation_iters
        self.blur_kernel = blur_kernel  # 0 = disabled; must be odd > 0 to enable

        # ---- load COLMAP scene -----------------------------------------------
        from datasets.colmap import Dataset, Parser

        parser = Parser(
            data_dir=data_dir,
            factor=data_factor,
            normalize=True,
            test_every=1_000_000,  # keep all views as "train"
        )
        dataset = Dataset(parser, split="train", load_depths=False)

        self.camtoworlds: List[Tensor] = []  # [4,4] float32
        self.Ks: List[Tensor] = []           # [3,3] float32
        self.images: List[Tensor] = []       # [H,W,3] float32 in [0,1]
        self.masks: List[Tensor] = []        # [H,W]   float32 in [0,1], dilated
        self.image_names: List[str] = []
        self.widths: List[int] = []
        self.heights: List[int] = []

        mask_dir_path = Path(mask_dir)

        for idx in range(len(dataset)):
            entry = dataset[idx]

            c2w = entry["camtoworld"].float()          # [4,4]
            K   = entry["K"].float()                   # [3,3]
            img = entry["image"].float() / 255.0       # [H,W,3]

            H, W = img.shape[:2]
            name = entry.get("image_name", str(idx))
            stem = Path(name).stem

            # ---- load mask -----------------------------------------------
            mask_path = mask_dir_path / f"{stem}.npy"
            if mask_path.exists():
                raw = np.load(str(mask_path)).astype(np.float32)
                mask_t = torch.from_numpy(raw)           # [H,W]
            else:
                mask_t = torch.zeros(H, W, dtype=torch.float32)

            # Resize mask to match potentially downsampled image
            if mask_t.shape[0] != H or mask_t.shape[1] != W:
                mask_t = F.interpolate(
                    mask_t.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="nearest",
                ).squeeze(0).squeeze(0)

            # Dilation: expand hole region so supervision bleeds into surroundings
            mask_t = self._dilate(mask_t, iters=dilation_iters)

            # Optional soft mask via Gaussian blur
            if blur_kernel > 0:
                ksize = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
                mask_t = gaussian_blur(
                    mask_t.unsqueeze(0).unsqueeze(0), kernel_size=ksize
                ).squeeze(0).squeeze(0)
                mask_t = mask_t.clamp(0.0, 1.0)

            self.camtoworlds.append(c2w)
            self.Ks.append(K)
            self.images.append(img)
            self.masks.append(mask_t)
            self.image_names.append(name)
            self.widths.append(W)
            self.heights.append(H)

        print(
            f"[Dataset] Loaded {len(self.images)} views from {data_dir} "
            f"with masks from {mask_dir}."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _dilate(mask: Tensor, iters: int) -> Tensor:
        """Binary dilation using max-pooling (GPU-friendly)."""
        if iters <= 0:
            return mask
        x = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        for _ in range(iters):
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x.squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """
        Randomly sample *batch_size* views from the dataset.
        Returns a list of dicts, each with:
            camtoworld, K, image, mask, width, height
        """
        indices = torch.randperm(len(self))[: batch_size].tolist()
        batch = []
        for i in indices:
            batch.append(
                {
                    "camtoworld": self.camtoworlds[i].to(self.device),
                    "K":          self.Ks[i].to(self.device),
                    "image":      self.images[i].to(self.device),   # [H,W,3]
                    "mask":       self.masks[i].to(self.device),     # [H,W]
                    "width":      self.widths[i],
                    "height":     self.heights[i],
                }
            )
        return batch


# SH DC normalisation constant  C0 = 1 / (2√π)
_C0 = 0.28209479177387814


# =============================================================================
# Visual Hull
# =============================================================================


def build_visual_hull(
    camtoworlds: List[Tensor],   # list of [4,4]
    Ks: List[Tensor],            # list of [3,3]
    masks: List[Tensor],         # list of [H,W] float in [0,1]
    num_points: int = 50_000,
    scene_scale: float = 1.5,
    min_support_ratio: float = 0.3,
    hit_ratio_threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Build a coarse visual hull by rejection-sampling 3-D candidates.

    Samples random points in the cube [-scene_scale, scene_scale]^3 and
    retains those that project into a hole mask in at least
    (min_support_ratio × N_valid_views) views with at least
    hit_ratio_threshold fraction of those views being mask hits.

    Returns:
        means      [M, 3]  accepted positions (on *device*)
        hull_aabb  [2, 3]  (min_xyz, max_xyz) AABB of accepted points
    """
    print(f"[Hull] Building visual hull from {len(masks)} views "
          f"(target {num_points:,} pts, scene_scale={scene_scale}) ...")

    # Pre-compute world-to-cam matrices; keep everything on CPU for bulk ops
    view_data = []
    for c2w, K, mask in zip(camtoworlds, Ks, masks):
        if mask.sum().item() == 0:
            continue
        worldtocam = torch.linalg.inv(c2w.cpu())[:3, :]   # [3, 4]
        view_data.append((K.cpu(), worldtocam, (mask > 0.5).cpu()))

    def _fallback(n: int) -> Tensor:
        return (torch.rand(n, 3) * 2 - 1) * scene_scale * 0.4

    if len(view_data) == 0:
        print("[Hull] No valid masks — random fallback.")
        means = _fallback(num_points)
        aabb  = torch.stack([means.min(0).values, means.max(0).values])
        return means.to(device), aabb.to(device)

    min_support  = max(2, int(min_support_ratio * len(view_data)))
    sample_batch = max(200_000, num_points * 10)
    max_rounds   = 60
    accepted: List[Tensor] = []
    accepted_count = 0

    for _ in range(max_rounds):
        pts           = (torch.rand(sample_batch, 3) * 2 - 1) * scene_scale
        visible_count = torch.zeros(sample_batch, dtype=torch.int32)
        hit_count     = torch.zeros(sample_batch, dtype=torch.int32)

        for K, worldtocam, mask in view_data:
            cam   = pts @ worldtocam[:, :3].T + worldtocam[:, 3]   # [N, 3]
            z     = cam[:, 2]
            valid = z > 1e-6
            x_n   = cam[:, 0] / z.clamp_min(1e-6)
            y_n   = cam[:, 1] / z.clamp_min(1e-6)
            u     = K[0, 0] * x_n + K[0, 2]
            v     = K[1, 1] * y_n + K[1, 2]
            h, w  = mask.shape
            inside = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if inside.any():
                visible_count[inside] += 1
                ui  = u[inside].long().clamp(0, w - 1)
                vi  = v[inside].long().clamp(0, h - 1)
                hit = mask[vi, ui]
                hit_count[inside] += hit.to(torch.int32)

        ratio = hit_count.float() / visible_count.float().clamp_min(1.0)
        keep  = (
            (visible_count >= min_support)
            & (hit_count   >= min_support)
            & (ratio       >= hit_ratio_threshold)
        )
        if keep.any():
            accepted.append(pts[keep])
            accepted_count += int(keep.sum())
        if accepted_count >= num_points:
            break

    if accepted_count == 0:
        print("[Hull] Visual hull empty — random fallback.")
        means = _fallback(num_points)
    else:
        cat = torch.cat(accepted, dim=0)
        if cat.shape[0] >= num_points:
            perm  = torch.randperm(cat.shape[0])[:num_points]
            means = cat[perm]
        else:
            reps  = math.ceil(num_points / cat.shape[0])
            cat   = cat.repeat(reps, 1)[:num_points]
            means = cat + torch.randn_like(cat) * (0.005 * scene_scale)
        print(f"[Hull] Accepted {accepted_count:,} candidates → "
              f"using {means.shape[0]:,} fill Gaussians.")

    aabb = torch.stack([means.min(0).values, means.max(0).values])
    return means.to(device), aabb.to(device)


# =============================================================================
# 3DGS Scene Representation
# =============================================================================


class FillScene(nn.Module):
    """
    Two-component scene for 3DGS hole filling:

      bg_splats  — background Gaussians loaded from checkpoint, **frozen**.
                   They represent the scene after object removal (with hole).
      new_splats — small set of trainable Gaussians initialised inside the
                   hole via a coarse visual hull built from the 2-D masks.
                   Only these Gaussians receive gradient.

    hull_aabb ([2, 3]) stored as a buffer; new means are hard-clamped to this
    AABB after every optimizer step to prevent fill Gaussians from drifting
    into the already-correct background region.
    """

    def __init__(
        self,
        bg_splats:  nn.ParameterDict,
        new_splats: nn.ParameterDict,
        hull_aabb:  Tensor,            # [2, 3]  (min_xyz, max_xyz)
    ) -> None:
        super().__init__()
        self.bg  = bg_splats
        self.new = new_splats
        self.register_buffer("hull_aabb", hull_aabb)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        ckpt_path:    str,
        camtoworlds:  List[Tensor],
        Ks:           List[Tensor],
        masks:        List[Tensor],
        num_fill:     int   = 50_000,
        scene_scale:  float = 1.5,
        init_opacity: float = 0.01,
        device:       torch.device = torch.device("cpu"),
    ) -> "FillScene":
        """
        Construct the scene from:
          - a background checkpoint (all Gaussians frozen)
          - a visual hull derived from multi-view hole masks (new trainable Gaussians)
        """
        # ---- Background (frozen) ----------------------------------------
        ckpt = torch.load(ckpt_path, map_location=device)
        splats_state: Dict = ckpt.get("splats", ckpt)
        EXCLUDE = {"foreground_logits"}
        bg_splats = nn.ParameterDict({
            k: nn.Parameter(v.to(device), requires_grad=False)
            for k, v in splats_state.items()
            if isinstance(v, torch.Tensor) and k not in EXCLUDE
        })
        print(f"[Scene] Loaded {bg_splats['means'].shape[0]:,} frozen background Gaussians.")

        # Ensure bg has sh0; fall back to zeros if missing
        if "sh0" not in bg_splats:
            N_bg = bg_splats["means"].shape[0]
            bg_splats["sh0"] = nn.Parameter(
                torch.zeros(N_bg, 1, 3, device=device), requires_grad=False
            )

        # ---- Visual hull → initial positions for fill Gaussians ----------
        hull_means, hull_aabb = build_visual_hull(
            camtoworlds=camtoworlds,
            Ks=Ks,
            masks=masks,
            num_points=num_fill,
            scene_scale=scene_scale,
            device=device,
        )
        M = hull_means.shape[0]

        # Estimate initial scale from hull density
        hull_extent = (hull_aabb[1] - hull_aabb[0]).max().item()
        init_scale  = max(hull_extent / (M ** (1.0 / 3.0)) * 0.5, 1e-4)
        init_logit_o = math.log(init_opacity / (1.0 - init_opacity))

        new_splats = nn.ParameterDict({
            "means":     nn.Parameter(hull_means),
            "colors":    nn.Parameter(torch.zeros(M, 3, device=device)),
            "opacities": nn.Parameter(
                torch.full((M, 1), init_logit_o, device=device)
            ),
            "scales":    nn.Parameter(
                torch.full((M, 3), math.log(init_scale), device=device)
            ),
            "quats":     nn.Parameter(
                F.normalize(torch.randn(M, 4, device=device), dim=-1)
            ),
        })
        print(f"[Scene] Initialised {M:,} trainable fill Gaussians "
              f"(init_scale={init_scale:.4f}, init_opacity={init_opacity}).")

        return cls(bg_splats, new_splats, hull_aabb)

    # ------------------------------------------------------------------
    def get_splat_params(self) -> Dict[str, Tensor]:
        """
        Merge bg (frozen) + new (grad-carrying) into a single param dict
        compatible with gsplat.rasterization (sh_degree=None mode).

        Background colors: proper SH DC evaluation  rgb = sh0_dc * C0 + 0.5
        New Gaussian colors: logit-encoded           rgb = sigmoid(colors)
        Opacities / scales / quats: logit / log / unit-quat → activated.
        """
        # Background: proper SH DC evaluation (no sigmoid!)
        bg_sh0_dc = self.bg["sh0"].detach().squeeze(1)          # [N, 3]
        bg_colors = (bg_sh0_dc * _C0 + 0.5).clamp(0.0, 1.0)    # [N, 3]

        # New Gaussians: logit → sigmoid
        new_colors = torch.sigmoid(self.new["colors"])           # [M, 3]

        # Opacities (handle both [N] and [N,1] checkpoint formats)
        bg_op = torch.sigmoid(self.bg["opacities"].detach())
        if bg_op.dim() == 2:
            bg_op = bg_op.squeeze(-1)                            # [N]
        new_op = torch.sigmoid(self.new["opacities"])
        if new_op.dim() == 2:
            new_op = new_op.squeeze(-1)                          # [M]

        bg_scales  = torch.exp(self.bg["scales"].detach())       # [N, 3]
        new_scales = torch.exp(self.new["scales"])               # [M, 3]

        bg_quats  = F.normalize(self.bg["quats"].detach(), dim=-1)  # [N, 4]
        new_quats = F.normalize(self.new["quats"],          dim=-1)  # [M, 4]

        merged_colors = torch.cat([bg_colors, new_colors], dim=0)

        return {
            "means":     torch.cat([self.bg["means"].detach(), self.new["means"]], dim=0),
            "quats":     torch.cat([bg_quats,  new_quats],  dim=0),
            "scales":    torch.cat([bg_scales, new_scales], dim=0),
            "opacities": torch.cat([bg_op,     new_op],     dim=0),
            "sh0":       merged_colors.unsqueeze(1),                # [N+M, 1, 3]
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def clamp_means_to_hull(self) -> None:
        """
        Hard-clamp new Gaussian positions to the visual hull AABB after
        every optimizer step, preventing fill Gaussians from drifting into
        the already-correct background region.
        """
        self.new["means"].data.clamp_(
            min=self.hull_aabb[0],
            max=self.hull_aabb[1],
        )

    # ------------------------------------------------------------------
    def scale_regularization(self) -> Tensor:
        """L2 penalty on log-scales to keep fill Gaussians compact."""
        return self.new["scales"].pow(2).mean()


# =============================================================================
# Renderer
# =============================================================================


def render(
    scene: FillScene,
    camtoworld: Tensor,   # [4,4]
    K: Tensor,            # [3,3]
    width: int,
    height: int,
) -> Tensor:
    """
    Render the Gaussian scene from a single COLMAP camera.

    Returns:
        rgb  [H, W, 3]  float32 in [0, 1]
    """
    params = scene.get_splat_params()

    # rasterization expects batched inputs
    viewmats = torch.linalg.inv(camtoworld).unsqueeze(0)  # [1,4,4]
    Ks_b     = K.unsqueeze(0)                              # [1,3,3]

    render_colors, _, _ = rasterization(
        means     = params["means"],
        quats     = params["quats"],
        scales    = params["scales"],
        opacities = params["opacities"],
        colors    = params["sh0"].squeeze(1),               # [N,3] direct RGB
        viewmats  = viewmats,
        Ks        = Ks_b,
        width     = width,
        height    = height,
        sh_degree = None,          # None = post-activation RGB [N,3], not SH coefficients
        render_mode        = "RGB",
        packed             = False,
        absgrad            = False,
        sparse_grad        = False,
        rasterize_mode     = "classic",
    )
    # render_colors: [1, H, W, 3]
    return render_colors[0]  # [H, W, 3]


def render_batch(
    scene: FillScene,
    batch: List[Dict],
) -> List[Tensor]:
    """
    Render all views in *batch*.

    Returns:
        list of [H, W, 3] rgb tensors
    """
    return [
        render(scene, v["camtoworld"], v["K"], v["width"], v["height"])
        for v in batch
    ]


# =============================================================================
# Diffusion Wrapper
# =============================================================================


class DiffusionWrapper(nn.Module):
    """
    Wrapper around Stable Diffusion Inpainting UNet for mask-aware SDS.

    Uses ``runwayml/stable-diffusion-inpainting`` (or any compatible inpainting
    checkpoint).  The inpainting UNet has **9 input channels**::

        [z_t (4)] + [mask_down (1)] + [masked_image_latent (4)]

    This lets the model see exactly where the hole is and what the surrounding
    context looks like, giving much better boundary fusion than plain SD v1.5.

    Falls back to a lightweight mock (random noise prediction) when *diffusers*
    is unavailable or *model_id* is None.

    Interface::

        noise_pred = wrapper.predict_noise(x_t, t, prompt_embeds,
                                           mask_latent, masked_image_latent)
    """

    def __init__(
        self,
        model_id: Optional[str] = "runwayml/stable-diffusion-inpainting",
        device: torch.device = torch.device("cpu"),
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.use_mock = use_mock or (not _DIFFUSERS_AVAILABLE) or (model_id is None)

        if not self.use_mock:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            self.unet      = pipe.unet.eval()
            self.vae       = pipe.vae.eval()
            self.tokenizer = pipe.tokenizer
            self.text_enc  = pipe.text_encoder.eval()
            self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            # Freeze all SD weights — only 3DGS params are optimised
            for p in self.unet.parameters():
                p.requires_grad_(False)
            for p in self.vae.parameters():
                p.requires_grad_(False)
            for p in self.text_enc.parameters():
                p.requires_grad_(False)
            self.num_train_timesteps = self.scheduler.config.num_train_timesteps
            # Store the actual dtype the models were loaded with (dtype kwarg may be
            # silently ignored by some pipeline versions, so read it from the weights)
            self.model_dtype = next(self.vae.parameters()).dtype
            print(f"[Diffusion] Loaded inpainting model '{model_id}' (dtype={self.model_dtype}).")
        else:
            self.num_train_timesteps = 1000
            print("[Diffusion] Using mock noise predictor (no SD model loaded).")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_prompt(self, prompt: str, batch_size: int = 1) -> Tensor:
        """Encode a text prompt → [B, seq_len, hidden] embeddings."""
        if self.use_mock:
            return torch.zeros(batch_size, 77, 768, device=self.device)
        tokens = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_enc(tokens.input_ids.to(self.device))[0]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, rgb: Tensor) -> Tensor:
        """
        Encode an RGB image [B, 3, H, W] in [0,1] → VAE latent [B, 4, H//8, W//8].
        """
        if self.use_mock:
            B, _, H, W = rgb.shape
            return torch.randn(B, 4, H // 8, W // 8, device=self.device)
        x = rgb * 2.0 - 1.0  # → [-1, 1]
        return self.vae.encode(x.to(self.device, dtype=self.model_dtype)).latent_dist.sample() * 0.18215

    # ------------------------------------------------------------------
    def add_noise(self, latents: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample ε ~ N(0,I) and add to *latents* at timestep *t*.

        Returns: (noisy_latents, noise)
        """
        noise = torch.randn_like(latents)
        if self.use_mock:
            noisy = latents + noise * (t.float() / self.num_train_timesteps).view(-1, 1, 1, 1)
            return noisy, noise

        alphas_cumprod = self.scheduler.alphas_cumprod.to(latents.device)
        sqrt_alpha     = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        noisy = sqrt_alpha * latents + sqrt_one_minus * noise
        return noisy, noise

    # ------------------------------------------------------------------
    def predict_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        prompt_embeds: Tensor,
        mask_latent: Optional[Tensor] = None,
        masked_image_latent: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run the inpainting UNet forward pass.

        Args:
            x_t:                  [B, 4, Hl, Wl]  noisy latents
            t:                    [B] int64
            prompt_embeds:        [B, seq, hidden]
            mask_latent:          [B, 1, Hl, Wl]  downsampled hole mask (required for inpainting)
            masked_image_latent:  [B, 4, Hl, Wl]  VAE latent of image*(1-mask) (required for inpainting)

        Returns:
            eps_pred: [B, 4, Hl, Wl]  predicted noise
        """
        if self.use_mock:
            return torch.randn_like(x_t)

        dtype = self.model_dtype

        # Inpainting UNet: concatenate [z_t | mask | masked_image_latent] → 9 channels
        assert mask_latent is not None and masked_image_latent is not None, (
            "mask_latent and masked_image_latent are required for inpainting UNet"
        )
        unet_input = torch.cat(
            [x_t, mask_latent, masked_image_latent], dim=1
        )  # [B, 9, Hl, Wl]

        return self.unet(
            unet_input.to(dtype),
            t.to(self.device),
            encoder_hidden_states=prompt_embeds.to(dtype),
        ).sample


# =============================================================================
# Loss Functions
# =============================================================================


def sds_loss(
    diffusion: DiffusionWrapper,
    rgb_rendered: Tensor,     # [H, W, 3]  or [B, H, W, 3]
    mask: Tensor,             # [H, W]     or [B, H, W]   — COLMAP-view mask
    prompt_embeds: Tensor,    # [1, seq, hidden]
    t_range: Tuple[int, int] = (50, 950),
    return_pixel_map: bool = False,  # if True, also return [H,W] per-pixel loss map
) -> Tensor:
    """
    Inpainting-conditioned SDS loss:

        L_sds = || (ε_pred - ε) * mask_latent ||²

    The inpainting UNet receives three conditioning signals:
      - z_t:                 noisy latent of the full rendered image
      - mask_latent:         downsampled binary hole mask  (÷8)
      - masked_image_latent: VAE latent of image * (1 - mask), i.e. the
                             known background with the hole zeroed out

    This lets the UNet see exactly where the hole is and what surrounds it,
    producing gradients that are much more spatially aware than plain SDS.
    """
    # Ensure [B, H, W, 3] and [B, H, W]
    if rgb_rendered.dim() == 3:
        rgb_rendered = rgb_rendered.unsqueeze(0)
        mask = mask.unsqueeze(0)

    B, H, W, _ = rgb_rendered.shape

    # [B, 3, H, W] for VAE
    rgb_b = rgb_rendered.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

    with torch.no_grad():
        latents = diffusion.encode_image(rgb_b)             # [B, 4, Hl, Wl]

    Hl, Wl = latents.shape[2], latents.shape[3]

    # Downsampled mask for latent space conditioning  [B, 1, Hl, Wl]
    mask_latent = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(Hl, Wl),
        mode="bilinear",
        align_corners=False,
    )
    mask_latent = (mask_latent > 0.5).float()

    # Masked image: zero the hole region, encode to latent  [B, 4, Hl, Wl]
    # The inpainting UNet uses this to understand the surrounding context.
    with torch.no_grad():
        bg_mask_pixel = (1.0 - mask).unsqueeze(1)           # [B, 1, H, W]
        masked_rgb    = rgb_b * bg_mask_pixel                # hole pixels → 0
        masked_image_latent = diffusion.encode_image(masked_rgb)  # [B, 4, Hl, Wl]

    # Sample random timestep
    t = torch.randint(t_range[0], t_range[1], (B,), device=latents.device)

    with torch.no_grad():
        noisy_latents, noise_gt = diffusion.add_noise(latents, t)
        embeds     = prompt_embeds.expand(B, -1, -1)
        noise_pred = diffusion.predict_noise(
            noisy_latents,
            t,
            embeds,
            mask_latent=mask_latent,
            masked_image_latent=masked_image_latent,
        )

    # SDS: use (ε_pred − ε) as a fixed gradient signal, masked to hole region
    diff = noise_pred - noise_gt                            # [B, 4, Hl, Wl]
    masked_sq = (diff.detach() * mask_latent).pow(2)       # [B, 4, Hl, Wl]
    loss = masked_sq.mean()

    if return_pixel_map:
        # Average over latent channels → [B, Hl, Wl], then upsample to pixel space
        pixel_map = masked_sq.mean(dim=1, keepdim=True)    # [B, 1, Hl, Wl]
        pixel_map = F.interpolate(
            pixel_map, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)                                        # [B, H, W]
        return loss, pixel_map[0]                           # scalar, [H, W]
    return loss


def keep_loss(
    rgb_rendered: Tensor,  # [H, W, 3] or [B, H, W, 3]
    rgb_original: Tensor,  # [H, W, 3] or [B, H, W, 3]
    mask: Tensor,          # [H, W]    or [B, H, W]
) -> Tensor:
    """
    Preserve known (non-hole) regions:

        L_keep = || (I_render - I_orig) * (1 - mask) ||²
    """
    if rgb_rendered.dim() == 3:
        rgb_rendered = rgb_rendered.unsqueeze(0)
        rgb_original = rgb_original.unsqueeze(0)
        mask = mask.unsqueeze(0)

    bg = (1.0 - mask).unsqueeze(-1)          # [B, H, W, 1]
    return ((rgb_rendered - rgb_original) * bg).pow(2).mean()


def multiview_consistency_loss(rendered_views: List[Tensor]) -> Tensor:
    """
    Penalise large photometric differences between rendered views.

    Currently uses simple mean pairwise L2 at a common (min) spatial resolution.
    Designed to be extended later with reprojection-based warping.

    Args:
        rendered_views: list of [H_i, W_i, 3] tensors

    Returns:
        scalar loss
    """
    if len(rendered_views) < 2:
        return torch.tensor(0.0, device=rendered_views[0].device)

    # Resize all views to the smallest common size before comparing
    target_H = min(v.shape[0] for v in rendered_views)
    target_W = min(v.shape[1] for v in rendered_views)

    resized = [
        F.interpolate(
            v.permute(2, 0, 1).unsqueeze(0),   # [1, 3, H, W]
            size=(target_H, target_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                             # [3, H, W]
        for v in rendered_views
    ]

    loss = torch.tensor(0.0, device=rendered_views[0].device)
    count = 0
    for i in range(len(resized)):
        for j in range(i + 1, len(resized)):
            loss = loss + (resized[i] - resized[j]).pow(2).mean()
            count += 1

    return loss / max(count, 1)


# =============================================================================
# Training Loop
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDS Hole Filling for 3DGS")
    p.add_argument("--data_dir",   type=str, required=True,  help="COLMAP scene directory")
    p.add_argument("--mask_dir",   type=str, required=True,  help="Directory of .npy hole masks")
    p.add_argument("--ckpt",       type=str, required=True,  help="3DGS checkpoint .pt")
    p.add_argument("--result_dir", type=str, default="results/sds_fill")
    p.add_argument("--prompt",     type=str, default="a clean background scene, high quality")
    p.add_argument("--sd_model",   type=str, default=None,   help="HuggingFace SD model ID (None = mock). Recommended: runwayml/stable-diffusion-inpainting")
    p.add_argument("--max_steps",  type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=2,      help="Views per iteration (>=2 for MV loss)")
    p.add_argument("--data_factor",type=int, default=1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--dilation_iters", type=int, default=5, help="Mask dilation iterations")
    p.add_argument("--blur_kernel",    type=int, default=0,  help="Soft-mask blur kernel (0=disabled, odd)")
    p.add_argument("--lambda_sds",   type=float, default=1.0)
    p.add_argument("--lambda_keep",  type=float, default=1.0)
    p.add_argument("--lambda_mv",       type=float, default=0.1)
    p.add_argument("--lambda_scale_reg", type=float, default=1e-3,  help="L2 penalty on fill Gaussian log-scales")
    p.add_argument("--num_fill_splats",  type=int,   default=50_000, help="Number of fill Gaussians initialised in visual hull")
    p.add_argument("--hull_scene_scale", type=float, default=1.5,   help="Bounding cube half-extent for visual hull sampling")
    p.add_argument("--init_opacity",     type=float, default=0.01,  help="Initial opacity of fill Gaussians")
    p.add_argument("--geom_warmup_steps",type=int,   default=500,   help="Freeze geometry (means/scales/quats) for first N steps")
    p.add_argument("--t_min", type=int, default=50,  help="Min diffusion timestep for SDS")
    p.add_argument("--t_max", type=int, default=950, help="Max diffusion timestep for SDS")
    p.add_argument("--log_every",    type=int, default=100)
    p.add_argument("--render_every",  type=int, default=50,   help="Save a preview render to renders/ every N steps")
    p.add_argument("--save_every",    type=int, default=1_000)
    p.add_argument("--device",        type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_image(path: str, tensor: Tensor) -> None:
    """Save [H,W,3] float [0,1] tensor as PNG."""
    img = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    imageio.imwrite(path, img)


def _plot_losses(
    history: Dict[str, List[float]],
    steps: List[int],
    stats_dir: str,
) -> None:
    """Save a loss-curve PNG to *stats_dir*.  Called after every log step."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    configs = [
        ("total", "Total Loss",            "tab:blue"),
        ("sds",   "SDS Loss",              "tab:orange"),
        ("keep",  "Keep Loss",             "tab:green"),
        ("mv",    "Multi-View Consistency", "tab:red"),
    ]
    for ax, (key, title, color) in zip(axes.flat, configs):
        vals = history.get(key, [])
        if vals:
            ax.plot(steps[: len(vals)], vals, color=color, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(stats_dir, "losses.png"), dpi=120)
    plt.close(fig)


def _save_debug_sds(
    loss_map: Tensor,  # [H, W] float — per-pixel SDS loss
    mask: Tensor,      # [H, W] float in [0, 1]
    path: str,
) -> None:
    """
    Save a side-by-side diagnostic image:
      LEFT:  SDS loss heatmap (inferno colormap; brighter = larger gradient signal)
      RIGHT: binary hole mask used in this step
    """
    loss_np = loss_map.detach().cpu().float().numpy()  # [H, W]
    mask_np = mask.detach().cpu().float().numpy()      # [H, W]

    # Normalise loss to [0, 1] for colormap
    vmax = loss_np.max()
    loss_norm = loss_np / vmax if vmax > 0 else loss_np

    cmap = plt.get_cmap("inferno")
    heatmap = (cmap(loss_norm)[..., :3] * 255).astype(np.uint8)   # [H, W, 3]
    mask_vis = (np.stack([mask_np] * 3, axis=-1) * 255).astype(np.uint8)  # [H, W, 3]

    combined = np.concatenate([heatmap, mask_vis], axis=1)         # [H, 2W, 3]
    imageio.imwrite(path, combined)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "renders"), exist_ok=True)
    stats_dir = os.path.join(args.result_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    ckpts_dir = os.path.join(args.result_dir, "ckpts")
    os.makedirs(ckpts_dir, exist_ok=True)
    debug_dir = os.path.join(args.result_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.result_dir, "tb"))

    # Loss history for matplotlib plots
    loss_history: Dict[str, List[float]] = {"total": [], "sds": [], "keep": [], "mv": []}
    loss_steps:   List[int] = []

    # ---- Dataset -----------------------------------------------------------
    dataset = ColmapMaskDataset(
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        data_factor=args.data_factor,
        dilation_iters=args.dilation_iters,
        blur_kernel=args.blur_kernel,
        device=device,
    )
    assert len(dataset) >= 1, "No views loaded — check data_dir/mask_dir."

    batch_size = min(args.batch_size, len(dataset))
    if batch_size < 2:
        print("[Warning] batch_size < 2; multi-view consistency loss will be 0.")

    # ---- Scene: frozen background + trainable fill Gaussians from visual hull
    scene = FillScene.build(
        ckpt_path=args.ckpt,
        camtoworlds=dataset.camtoworlds,
        Ks=dataset.Ks,
        masks=dataset.masks,
        num_fill=args.num_fill_splats,
        scene_scale=args.hull_scene_scale,
        init_opacity=args.init_opacity,
        device=device,
    ).to(device)

    # Only optimise the new (fill) Gaussians; background is permanently frozen
    optimizer = Adam(
        [
            {"params": [scene.new["means"]],     "lr": args.lr * 0.1, "name": "means"},
            {"params": [scene.new["colors"]],    "lr": args.lr,        "name": "colors"},
            {"params": [scene.new["opacities"]], "lr": args.lr * 0.5,  "name": "opacities"},
            {"params": [scene.new["scales"]],    "lr": args.lr * 0.5,  "name": "scales"},
            {"params": [scene.new["quats"]],     "lr": args.lr * 0.3,  "name": "quats"},
        ]
    )

    # ---- Diffusion ---------------------------------------------------------
    use_mock = (args.sd_model is None)
    diffusion = DiffusionWrapper(
        model_id=args.sd_model,
        device=device,
        use_mock=use_mock,
    ).to(device)

    prompt_embeds = diffusion.encode_prompt(args.prompt, batch_size=1)  # [1, seq, hid]

    # ---- Training ----------------------------------------------------------
    print(f"\n[Train] Starting SDS hole-filling: {args.max_steps} steps.\n")

    for step in range(1, args.max_steps + 1):
        # 1. Sample batch of COLMAP views (with their precomputed masks)
        batch = dataset.sample_batch(batch_size)

        # 2. Render all views
        rendered: List[Tensor] = render_batch(scene, batch)  # list of [H,W,3]

        # 3. Accumulate losses over batch views
        loss_sds_total  = torch.tensor(0.0, device=device)
        loss_keep_total = torch.tensor(0.0, device=device)

        for view, rgb_r in zip(batch, rendered):
            mask     = view["mask"]       # [H, W]
            rgb_orig = view["image"]      # [H, W, 3]

            # SDS loss — mask-aware, operates in latent space
            loss_sds_total = loss_sds_total + sds_loss(
                diffusion=diffusion,
                rgb_rendered=rgb_r,
                mask=mask,
                prompt_embeds=prompt_embeds,
                t_range=(args.t_min, args.t_max),
            )

            # Keep loss — preserve non-hole regions
            loss_keep_total = loss_keep_total + keep_loss(rgb_r, rgb_orig, mask)

        loss_sds_total  = loss_sds_total  / len(batch)
        loss_keep_total = loss_keep_total / len(batch)

        # 4. Multi-view consistency loss
        loss_mv = multiview_consistency_loss(rendered)

        # 5. Scale regularisation: keep fill Gaussians compact
        loss_scale_reg = scene.scale_regularization()

        # 6. Total loss
        loss = (
            args.lambda_sds       * loss_sds_total
            + args.lambda_keep    * loss_keep_total
            + args.lambda_mv      * loss_mv
            + args.lambda_scale_reg * loss_scale_reg
        )

        # 7. Backprop
        optimizer.zero_grad()
        loss.backward()

        # Geometry warmup: freeze positions/scales/quats in early steps
        # so appearance is learned first and floaters are suppressed.
        if step <= args.geom_warmup_steps:
            for pname in ("means", "scales", "quats"):
                if scene.new[pname].grad is not None:
                    scene.new[pname].grad.zero_()

        optimizer.step()

        # Hard-clamp new means to hull AABB to prevent background contamination
        scene.clamp_means_to_hull()

        # ---- Logging (console + TensorBoard) ------------------------------
        if step % args.log_every == 0 or step == 1:
            geom_note = "  [geom frozen]" if step <= args.geom_warmup_steps else ""
            print(
                f"Step {step:>6d}/{args.max_steps} | "
                f"loss={loss.item():.4f}  "
                f"sds={loss_sds_total.item():.4f}  "
                f"keep={loss_keep_total.item():.4f}  "
                f"mv={loss_mv.item():.4f}  "
                f"scale_reg={loss_scale_reg.item():.4f}"
                + geom_note
            )
            writer.add_scalar("loss/total",     loss.item(),             step)
            writer.add_scalar("loss/sds",       loss_sds_total.item(),   step)
            writer.add_scalar("loss/keep",      loss_keep_total.item(),  step)
            writer.add_scalar("loss/mv",        loss_mv.item(),          step)
            writer.add_scalar("loss/scale_reg", loss_scale_reg.item(),   step)

        # ---- Every render_every steps: preview render + debug viz + loss plot
        if step % args.render_every == 0:
            # 1. Loss history point (every 50 steps → finer x-axis in plot)
            loss_history["total"].append(loss.item())
            loss_history["sds"].append(loss_sds_total.item())
            loss_history["keep"].append(loss_keep_total.item())
            loss_history["mv"].append(loss_mv.item())
            loss_steps.append(step)
            _plot_losses(loss_history, loss_steps, stats_dir)

            # 2. Preview render
            with torch.no_grad():
                preview_view = dataset.sample_batch(1)[0]
                preview_rgb  = render(
                    scene,
                    preview_view["camtoworld"],
                    preview_view["K"],
                    preview_view["width"],
                    preview_view["height"],
                )
            save_image(
                os.path.join(args.result_dir, "renders", f"step_{step:06d}.png"),
                preview_rgb,
            )

            # 3. SDS debug visualization — show where the gradient acts
            #    Re-run sds_loss on the first batch view to get the pixel loss map
            with torch.no_grad():
                _, sds_pixel_map = sds_loss(
                    diffusion=diffusion,
                    rgb_rendered=rendered[0].detach(),
                    mask=batch[0]["mask"],
                    prompt_embeds=prompt_embeds,
                    t_range=(args.t_min, args.t_max),
                    return_pixel_map=True,
                )
            _save_debug_sds(
                sds_pixel_map,
                batch[0]["mask"],
                os.path.join(debug_dir, f"step_{step:06d}.png"),
            )

        # ---- Checkpoint + full visualisation -------------------------------
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = os.path.join(ckpts_dir, f"ckpt_step_{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    # Fill Gaussians only — for resuming training
                    "fill_splats": {
                        "means":     scene.new["means"].detach(),
                        "quats":     F.normalize(scene.new["quats"].detach(), dim=-1),
                        "scales":    scene.new["scales"].detach(),
                        "opacities": scene.new["opacities"].detach(),
                        "sh0":       torch.sigmoid(scene.new["colors"]).unsqueeze(1).detach(),
                    },
                    # Merged bg + fill — compatible with gsplat viewer
                    "splats": {
                        k: v.detach()
                        for k, v in scene.get_splat_params().items()
                    },
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

    writer.close()
    print("\n[Train] Done.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
