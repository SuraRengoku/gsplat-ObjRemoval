# type: ignore

"""
usage: python examples/OR_trainer2.py \
        --data_dir data/Tree \
        --result_dir results/Tree_Removed \
        --data_factor 2 \
        --ckpt results/Tree/ckpts/ckpt_29999_rank0.pt \
        --remove_foreground_gaussians \
        --foreground_thresh 0.5 \
        --max_steps 5000 \
        --mask_type Sam2 \
        --save_ply
"""

import json
import math
import os
import time 
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

@dataclass
class Config:
    disable_viewer: bool = False
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"

    data_dir: str = "data/kitchen"
    data_factor: int = 4
    result_dir: str = "results/kitchen"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    port: int = 8080

    batch_size: int = 1
    steps_scalar: float = 1.0

    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2

    near_plane: float = 0.01
    far_plane: float = 1e10

    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = True 
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False

    random_bkgd: bool = False

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # foreground loss settings
    foreground_loss: bool = True # whether to use foreground loss
    foreground_lambda: float = 0.1 # foreground loss weights
    foreground_lr: float = 1e-3 # learning rate
    foreground_warmup_steps: int = 500
    freeze_scene_params: bool = False  # freeze scene (only train foreground_logits)
    
    # Foreground loss enhancement options
    use_soft_labels: bool = True  # Blur GT mask with a Gaussian kernel to create soft labels
    use_focal_loss: bool = True   # Use focal loss to emphasize hard samples
    use_dice_loss: bool = True    # Use Dice loss to improve global consistency

    foreground_thresh: float = 0.5 
    remove_during_train: bool = False # whether to remove gaussian ellipsoid during train
    post_prune_ckpt: bool = True # whether to store pruned checkpoint
    
    # Foreground masking mode: set foreground gaussians to black
    foreground_mask_to_black: bool = False  # force foreground ellipsoid to be black
    green_threshold_lower: Tuple[float, float, float] = (0.0, 0.5, 0.0)  # green lower bound RGB
    green_threshold_upper: Tuple[float, float, float] = (0.5, 1.0, 0.5)  # green upper bound RGB 

    # Foreground removal mode: remove gaussians that fall on foreground mask
    remove_foreground_gaussians: bool = False
    removal_threshold: float = 0.5
    removal_steps: int = 1000 # remove once after steps
    mask_type: str = "Sam2"  # mask type: Sam2, Flow, Filled, Anchor
 

    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6

    use_bilateral_grid: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    depth_loss: bool = False
    depth_lambda: float = 1e-2

    tb_every: int = 100
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"
    
    with_ut: bool = False
    with_eval3d: bool = False

    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimiers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    foreground_lr: float = 1e-3,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    checkpoint_splats: Optional[Dict] = None,
    freeze_original_params: bool = False, # free initial splats parameters
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    
    # Initialize the params list (used to create optimizers).
    params = []
    
    # If checkpoint is provided, use it directly instead of initializing
    if checkpoint_splats is not None:
        # if there is no foreground_logits in checkpoint, add it
        if "foreground_logits" not in checkpoint_splats:
            num_gs = checkpoint_splats["means"].shape[0]
            checkpoint_splats["foreground_logits"] = torch.nn.Parameter(
                torch.zeros((num_gs), device=device)
            )
            print(f"Initialized foreground_logits for {num_gs} Gaussians (not in checkpoint)")

        if freeze_original_params:
            # Frozen mode: only add foreground_logits to params (used to create optimizers).
            params = [
                ("foreground_logits", checkpoint_splats["foreground_logits"], foreground_lr),
            ]

            # Create a wrapper that exposes only trainable params to the strategy.
            class FrozenParameterDict(torch.nn.ParameterDict):
                """Expose only trainable parameters to the strategy checker."""
                def __init__(self, trainable_params, frozen_params):
                    # Initialize _frozen_keys before calling parent __init__.
                    object.__setattr__(self, '_frozen_keys', set(frozen_params.keys()))
                    
                    # Initialize ParameterDict with trainable_params only.
                    super().__init__(trainable_params)
                    
                    # Store frozen_params as plain attributes (not registered parameters).
                    for key, value in frozen_params.items():
                        # Use object.__setattr__ to bypass ParameterDict.__setitem__.
                        object.__setattr__(self, key, value)
                
                def __getitem__(self, key):
                    # First try to get from ParameterDict.
                    if key in self._parameters:
                        return self._parameters[key]
                    # Otherwise get from frozen attributes.
                    if key in self._frozen_keys:
                        return object.__getattribute__(self, key)
                    raise KeyError(f"Key '{key}' not found")
                
                def __setitem__(self, key, value):
                    # Allow updating parameters (e.g., during pruning).
                    if key in self._parameters:
                        # Trainable params: use parent method.
                        super().__setitem__(key, value)
                    elif key in self._frozen_keys:
                        # Frozen params: update attributes directly.
                        object.__setattr__(self, key, value)
                    else:
                        # New params: treat as trainable by default.
                        super().__setitem__(key, value)
                
                def __contains__(self, key):
                    # Check if a key exists (including frozen ones).
                    return key in self._parameters or key in self._frozen_keys
                
                def keys(self):
                    # Return all keys (including frozen ones).
                    return list(set(self._parameters.keys()) | self._frozen_keys)
                
                def parameters(self, recurse=True):
                    # Return only trainable parameters; used by the strategy checker.
                    return super().parameters(recurse)
                
                def state_dict(self, destination=None, prefix='', keep_vars=False):
                    """
                    Override state_dict to include frozen parameters.
                    Ensure checkpoints include both trainable and frozen params.
                    """
                    # Get state_dict for trainable parameters first.
                    state = super().state_dict(destination, prefix, keep_vars)
                    
                    # Add frozen parameters.
                    for key in self._frozen_keys:
                        frozen_param = object.__getattribute__(self, key)
                        if keep_vars:
                            state[prefix + key] = frozen_param
                        else:
                            state[prefix + key] = frozen_param.detach()
                    
                    return state
                
                def load_state_dict(self, state_dict, strict=True):
                    """
                    Override load_state_dict to handle frozen parameters.
                    """
                    # Split trainable and frozen parameters.
                    trainable_state = {}
                    frozen_state = {}
                    
                    for key, value in state_dict.items():
                        if key in self._parameters:
                            trainable_state[key] = value
                        elif key in self._frozen_keys:
                            frozen_state[key] = value
                        elif strict:
                            raise KeyError(f"Unexpected key in state_dict: {key}")
                    
                    # Load trainable parameters.
                    super().load_state_dict(trainable_state, strict=strict)
                    
                    # Load frozen parameters.
                    for key, value in frozen_state.items():
                        if isinstance(value, torch.Tensor):
                            object.__setattr__(self, key, value)
                        else:
                            object.__setattr__(self, key, torch.nn.Parameter(value))
            
            # Prepare trainable and frozen parameters.
            trainable_params = {
                "foreground_logits": checkpoint_splats["foreground_logits"]
            }
            
            frozen_params = {}
            for key in ["means", "scales", "quats", "opacities"]:
                param = checkpoint_splats[key]
                param.requires_grad_(False)
                frozen_params[key] = param.to(device)

            if "sh0" in checkpoint_splats:
                checkpoint_splats["sh0"].requires_grad_(False)
                checkpoint_splats["shN"].requires_grad_(False)
                frozen_params["sh0"] = checkpoint_splats["sh0"].to(device)
                frozen_params["shN"] = checkpoint_splats["shN"].to(device)
            elif "features" in checkpoint_splats:
                checkpoint_splats["features"].requires_grad_(False)
                checkpoint_splats["colors"].requires_grad_(False)
                frozen_params["features"] = checkpoint_splats["features"].to(device)
                frozen_params["colors"] = checkpoint_splats["colors"].to(device)
            
            splats = FrozenParameterDict(trainable_params, frozen_params).to(device)

            print("Freezing mode enabled:")
            print(f"  - Frozen params: means, scales, quats, opacities, sh0/shN (or features/colors)")
            print(f"  - Trainable params: foreground_logits only")
        else:
            params = [
                ("means", checkpoint_splats["means"], means_lr * scene_scale),
                ("scales", checkpoint_splats["scales"], scales_lr),
                ("quats", checkpoint_splats["quats"], quats_lr),
                ("opacities", checkpoint_splats["opacities"], opacities_lr),
                ("foreground_logits", checkpoint_splats["foreground_logits"], foreground_lr),
            ]
            
            if "sh0" in checkpoint_splats:
                params.append(("sh0", checkpoint_splats["sh0"], sh0_lr))
                params.append(("shN", checkpoint_splats["shN"], shN_lr))
            elif "features" in checkpoint_splats:
                params.append(("features", checkpoint_splats["features"], sh0_lr))
                params.append(("colors", checkpoint_splats["colors"], sh0_lr))
            
            splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    else:
        # Normal initialization
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("init_type must be 'sfm' or 'random'")
        
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

        points = points[world_rank::world_size]
        rgbs = rgbs[world_rank::world_size]
        scales = scales[world_rank::world_size]

        N = points.shape[0]
        quats = torch.rand((N, 4))
        opacities = torch.logit(torch.full((N,), init_opacity)) 

        # Add foreground parameter; logit(0.5)=0 represents uncertainty.
        foreground_logits = torch.zeros((N, ))

        params = [
            # name, value, lr
            ("means", torch.nn.Parameter(points), means_lr * scene_scale),
            ("scales", torch.nn.Parameter(scales), scales_lr),
            ("quats", torch.nn.Parameter(quats), quats_lr),
            ("opacities", torch.nn.Parameter(opacities), opacities_lr),
            ("foreground_logits", torch.nn.Parameter(foreground_logits), foreground_lr),
        ]

        if feature_dim is None:
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
        else:
            features = torch.rand(N, feature_dim)
            params.append(("features", torch.nn.Parameter(features), sh0_lr))
            colors = torch.logit(rgbs)
            params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = batch_size
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    
    optimizers = {}
    for name, param, lr in params:
        if param.requires_grad:
            optimizers[name] = optimizer_class(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
    
    print(f"Created optimizers for: {list(optimizers.keys())}")

    return splats, optimizers

class Runner:
    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        os.makedirs(cfg.result_dir, exist_ok=True)

        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.parser = Parser(
            data_dir = cfg.data_dir,
            factor = cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.train_masks = self._load_foreground_masks()

        feature_dim = 32 if cfg.app_opt else None
        
        # Check if we need to load from checkpoint
        checkpoint_splats = None
        if cfg.ckpt is not None and len(cfg.ckpt) > 0:
            print(f"Loading checkpoint from {cfg.ckpt[world_rank]}")
            checkpoint = torch.load(cfg.ckpt[world_rank], map_location=self.device)
            checkpoint_splats = checkpoint["splats"]
            
            # If foreground_logits is not in checkpoint, initialize it
            if "foreground_logits" not in checkpoint_splats:
                num_gs = checkpoint_splats["means"].shape[0]
                checkpoint_splats["foreground_logits"] = torch.nn.Parameter(
                    torch.zeros((num_gs,), device=self.device)
                )
                print(f"Initialized foreground_logits for {num_gs} Gaussians (not in checkpoint)")
        
        self.splats, self.optimizers = create_splats_with_optimiers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            foreground_lr=cfg.foreground_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            checkpoint_splats=checkpoint_splats,
            freeze_original_params=cfg.freeze_scene_params,  # Use config flag instead of auto-detect.
        )
        
        if checkpoint_splats is not None:
            print(f"Loaded checkpoint. Number of GS: {len(self.splats['means'])}")
            
            # In frozen mode, disable strategy refinement (no densification/pruning).
            if cfg.freeze_scene_params:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    print("Freezing mode: Disabling strategy refinement (no densification/pruning)")
                    # Set refine_stop_iter=0 to disable refinement.
                    self.cfg.strategy.refine_stop_iter = 0
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    print("Freezing mode: Disabling MCMC strategy refinement")
                    self.cfg.strategy.refine_stop_iter = 0
            else:
                print("Joint training mode: Scene params and foreground_logits will be optimized together")
        else:
            print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification strategy.
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
            
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses and metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer.
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def _prepare_mask_directory(self, mask_type: str) -> Path:
        """
        prepare the mask folder and create scaled images if needed

        Args:
            mask_type: mask type('Sam2', 'Flow', 'Filled', 'Anchor')
        
        Returns:
            the mask folder path that will be used
        """
        data_dir = Path(self.cfg.data_dir)
        factor = self.cfg.data_factor
        
        # new directory structure: Tree/mask/Sam2/images/
        mask_base_dir = data_dir / "mask" / mask_type
        mask_images_dir = mask_base_dir / "images"
        
        # If factor=1, use the original mask.
        if factor == 1:
            if mask_images_dir.exists():
                return mask_images_dir
            else:
                print(f"Warning: Mask directory {mask_images_dir} not found.")
                return None
        
        # If factor>1, check the scaled mask version.
        mask_scaled_dir = mask_base_dir / f"images_{factor}"
        
        # Check if it already exists.
        if mask_scaled_dir.exists():
            print(f"Using existing scaled mask directory: {mask_scaled_dir}")
            return mask_scaled_dir
        
        # If not, create it.
        if not mask_images_dir.exists():
            print(f"Warning: Source mask directory {mask_images_dir} not found.")
            return None
        
        print(f"Creating scaled mask directory: {mask_scaled_dir}")
        os.makedirs(mask_scaled_dir, exist_ok=True)
        
        # Get all files to be scaled.
        mask_files = list(mask_images_dir.glob("*"))
        
        from PIL import Image
        for mask_file in tqdm.tqdm(mask_files, desc=f"Scaling {mask_type} masks"):
            base_name = mask_file.stem
            ext = mask_file.suffix
            
            # Skip _debug files (they can be scaled later if needed).
            is_debug = '_debug' in base_name
            
            if ext == '.npy':
                # Scale .npy file (bool mask).
                try:
                    mask = np.load(mask_file)  # [H, W] bool
                    # Use PIL.
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_img = Image.fromarray(mask_uint8)
                    new_size = (mask_img.width // factor, mask_img.height // factor)
                    resized = mask_img.resize(new_size, Image.NEAREST)  # nearest to preserve bool mask
                    resized_mask = (np.array(resized) > 127)  # convert back to bool
                    
                    output_path = mask_scaled_dir / mask_file.name
                    np.save(output_path, resized_mask)
                except Exception as e:
                    print(f"Warning: Failed to scale {mask_file}: {e}")
                    
            elif ext.lower() in ['.jpg', '.jpeg', '.png']:
                # Scale image.
                try:
                    img = imageio.imread(mask_file)
                    img_pil = Image.fromarray(img)
                    new_size = (img_pil.width // factor, img_pil.height // factor)
                    resized = img_pil.resize(new_size, Image.BICUBIC)
                    
                    # Store as PNG to preserve quality.
                    output_name = base_name + '.png'
                    output_path = mask_scaled_dir / output_name
                    imageio.imwrite(output_path, np.array(resized))
                except Exception as e:
                    print(f"Warning: Failed to scale {mask_file}: {e}")
        
        print(f"Scaled mask directory created: {mask_scaled_dir}")
        return mask_scaled_dir

    def _load_foreground_masks(self) -> Dict[int, torch.Tensor]:
        """
        load foreground mask
        prefer .npy files (bool array, foreground=True, background=False)
        if no .npy file, use green screen .jpg/.png masks
        ignore *_debug files
        
        new directory structure supports
        - Tree/mask/Sam2/images/     (original mask)
        - Tree/mask/Sam2/images_2/   (scaled mask, genarated automatically)
        """
        masks = {}
        
        # Use the mask type from config.
        mask_type = self.cfg.mask_type
        mask_dir = self._prepare_mask_directory(mask_type)
        
        if mask_dir is None:
            # If the specified mask type does not exist, try alternatives.
            print(f"Specified mask type '{mask_type}' not found, trying alternatives...")
            data_dir = Path(self.cfg.data_dir)
            mask_types = ['Sam2', 'Flow', 'Filled', 'Anchor']
            
            for alt_mask_type in mask_types:
                if alt_mask_type == mask_type:
                    continue
                potential_dir = self._prepare_mask_directory(alt_mask_type)
                if potential_dir is not None:
                    mask_dir = potential_dir
                    print(f"Using {alt_mask_type} masks from: {mask_dir}")
                    break
        
        if mask_dir is None:
            data_dir = Path(self.cfg.data_dir)
            print(f"Warning: No mask directory found in {data_dir / 'mask'}")
            return masks
        
        for image_id in self.trainset.indices:
            image_name = self.parser.image_names[image_id]
            base_name = Path(image_name).stem  # remove ext name
            
            # Try .npy file first.
            npy_path = mask_dir / f"{base_name}.npy"
            if npy_path.exists():
                try:
                    mask = np.load(npy_path)  # bool array, foreground=True, background=False
                    mask = torch.from_numpy(mask).float()  # convert to float: True->1.0, False->0.0
                    masks[image_id] = mask.to(self.device)
                    continue
                except Exception as e:
                    print(f"Warning: Failed to load {npy_path}: {e}")
            
            # If .npy does not exist, try green screen .jpg/.png files.
            # Do not load *_debug files.
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                img_path = mask_dir / f"{base_name}{ext}"
                debug_path = mask_dir / f"{base_name}_debug{ext}"
                
                # Skip *_debug files.
                if img_path.exists() and img_path != debug_path:
                    try:
                        mask_img = imageio.imread(img_path)  # [H, W, 3] or [H, W]
                        
                        # Check green screen (high G, low R/B).
                        if len(mask_img.shape) == 3:
                            # Green channel check.
                            green_mask = (
                                (mask_img[..., 1] > 100) &  # G > 100
                                (mask_img[..., 0] < 100) &  # R < 100
                                (mask_img[..., 2] < 100)    # B < 100
                            )
                            mask = torch.from_numpy(green_mask).float()
                        else:
                            # Grayscale image, assume foreground is highlighted.
                            mask = torch.from_numpy(mask_img).float() / 255.0
                        
                        masks[image_id] = mask.to(self.device)
                        break
                    except Exception as e:
                        print(f"Warning: Failed to load {img_path}: {e}")
            
            if image_id not in masks:
                print(f"Warning: No mask found for image: {image_name}")

        print(f"Loaded {len(masks)} foreground masks from {mask_dir}")
        if len(masks) > 0:
            first_mask = next(iter(masks.values()))
            print(f"  Mask shape: {first_mask.shape}, dtype: {first_mask.dtype}")
            print(f"  Foreground pixels: {(first_mask > 0.5).sum().item()}")
        
        return masks
    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    
    def _identify_foreground_gaussians(self, 
                                      info: Dict, 
                                      foreground_mask: torch.Tensor, # [H, W]
                                      height: int,
                                      width: int
                                      ) -> torch.Tensor:
        """ 
            identify the gaussian primitives landing on the foreground mask 
            returns: mask: [N] bool tensor, True indicates that the gaussian ellipsoid should be removed
        """
        N = self.splats["means"].shape[0]
        removal_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Debug: Print info keys
        print(f"DEBUG: info keys = {info.keys()}")
        print(f"DEBUG: cfg.packed = {self.cfg.packed}")
        if "gaussian_ids" in info:
            print(f"DEBUG: gaussian_ids shape = {info['gaussian_ids'].shape}")
        if "isect_ids" in info:
            print(f"DEBUG: isect_ids shape = {info['isect_ids'].shape}")
        if "flatten_ids" in info:
            print(f"DEBUG: flatten_ids shape = {info['flatten_ids'].shape}")

        # Packed Mode: acquired from rendering information
        if self.cfg.packed and "gaussian_ids" in info:
            
            gaussian_ids = info["gaussian_ids"]  # [nnz]
            
            # In packed mode, use 2D projected positions to check foreground.
            # Use means2d to get the projected position of each Gaussian.
            if "means2d" in info:
                means2d = info["means2d"]  # [C, N, 2] or [nnz, 2]
                
                # If means2d is [C, N, 2], take the first camera.
                if len(means2d.shape) == 3:
                    means2d = means2d[0]  # [N, 2]
                    # Only check Gaussians in gaussian_ids.
                    relevant_means2d = means2d[gaussian_ids]  # [nnz, 2]
                else:
                    relevant_means2d = means2d  # already [nnz, 2]
                
                # Convert 2D coordinates to pixel coordinates.
                u = relevant_means2d[:, 0].long().clamp(0, width - 1)
                v = relevant_means2d[:, 1].long().clamp(0, height - 1)
                
                # Check which projected Gaussians land on the foreground mask.
                is_on_foreground = foreground_mask[v, u] > self.cfg.removal_threshold
                
                # Get foreground Gaussian IDs (unique).
                foreground_gaussian_ids = gaussian_ids[is_on_foreground].unique()
                
                # Mark Gaussians to remove.
                removal_mask[foreground_gaussian_ids] = True
                
                print(f"DEBUG: Packed mode - found {foreground_gaussian_ids.numel()} unique foreground gaussians to remove")
            else:
                print("Warning: means2d not found in packed mode, cannot identify foreground gaussians")

        # Non-packed Mode: use radii info
        else:
            # Non-packed mode: check rendering info.
            # Without gaussian_ids, only a simplified method is available.
            print("Warning: Non-packed mode removal is not fully supported. Use --packed for better results.")
            
            if "radii" in info and "means2d" in info:
                radii = info["radii"]  # [C, N]
                means2d = info["means2d"]  # [C, N, 2]
                
                # First camera.
                if len(radii.shape) == 2:
                    radii_cam0 = radii[0]  # [N]
                    means2d_cam0 = means2d[0] if len(means2d.shape) == 3 else means2d  # [N, 2]
                else:
                    radii_cam0 = radii  # [N]
                    means2d_cam0 = means2d  # [N, 2]
                
                # Check dimension match.
                if radii_cam0.shape[0] != N:
                    print(f"Warning: radii shape {radii_cam0.shape} doesn't match N={N}, skipping removal")
                else:
                    # Only check visible Gaussians.
                    is_visible = radii_cam0 > 0  # [N]
                    visible_indices = torch.where(is_visible)[0]
                    
                    for idx in visible_indices:
                        i = idx.item()
                        point_2d = means2d_cam0[i]  # [2]
                        
                        if point_2d.numel() != 2:
                            continue
                        
                        u = point_2d[0].item()
                        v = point_2d[1].item()
                        x_int = int(u)
                        y_int = int(v)

                        if 0 <= x_int < width and 0 <= y_int < height:
                            if foreground_mask[y_int, x_int] > self.cfg.removal_threshold:
                                removal_mask[i] = True
            else:
                print("Warning: radii or means2d not available, skipping non-packed removal")
        
        return removal_mask

    def _update_optimizers_after_removal(self, keep_mask: torch.Tensor):
        """
        update optimizer state and remove optimizers for deleted Gaussians
        
        Args:
            keep_mask: [N] bool tensor, True to keep the gaussian ellipsoid
        """
        # Recreate optimizer because shapes changed.
        scene_scale = self.scene_scale
        
        # Recreate optimizers.
        new_optimizers = {}
        
        # Recreate optimizer for each param group.
        param_groups = [
            ("means", self.cfg.means_lr * scene_scale),
            ("scales", self.cfg.scales_lr),
            ("quats", self.cfg.quats_lr),
            ("opacities", self.cfg.opacities_lr),
            ("foreground_logits", self.cfg.foreground_lr),
        ]
        
        if self.cfg.app_opt:
            param_groups.extend([
                ("features", self.cfg.app_opt_lr),
                ("colors", self.cfg.app_opt_lr),
            ])
        else:
            param_groups.extend([
                ("sh0", self.cfg.sh0_lr),
                ("shN", self.cfg.shN_lr),
            ])
        
        for name, lr in param_groups:
            if name in self.splats:
                if self.cfg.visible_adam:
                    from gsplat.optimizers import SelectiveAdam
                    new_optimizers[name] = SelectiveAdam(
                        [self.splats[name]], lr=lr, eps=1e-15
                    )
                else:
                    new_optimizers[name] = torch.optim.Adam(
                        [self.splats[name]], lr=lr, eps=1e-15
                    )
        
        # Replace old optimizers.
        self.optimizers = new_optimizers
        
        print(f"  -> Optimizers updated for {len(self.splats['means'])} gaussians")

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump config.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
        
        # Print critical config for debugging.
        print(f"Training Config:")
        print(f"  - Packed mode: {cfg.packed}")
        print(f"  - Remove foreground: {cfg.remove_foreground_gaussians}")
        print(f"  - Removal steps: {cfg.removal_steps}")
        print(f"  - Mask type: {cfg.mask_type}")
        print(f"  - Loaded masks: {len(self.train_masks)}")

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = []
        # Means has a learning rate schedule ending at 0.01 of the initial value.
        # Create scheduler only when means optimizer exists (disabled when frozen).
        if "means" in self.optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.pose_opt:
            # Pose optimization has a learning rate schedule.
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # Bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

            # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # SH schedule.
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Forward.
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # Foreground mask to black: set Gaussians to black where the mask is foreground.
            if cfg.foreground_mask_to_black:
                image_id = image_ids[0].item()
                if image_id in self.train_masks:
                    foreground_mask = self.train_masks[image_id]  # [H_orig, W_orig]
                    # Resize mask to match rendered image size if needed
                    if foreground_mask.shape[0] != height or foreground_mask.shape[1] != width:
                        foreground_mask = torch.nn.functional.interpolate(
                            foreground_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H_orig, W_orig]
                            size=(height, width),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # [height, width]
                    # set foreground color to black
                    colors = colors * (1.0 - foreground_mask.unsqueeze(0).unsqueeze(-1))
                    
                    # Update Gaussian color parameters as well.
                    if cfg.packed and "gaussian_ids" in info and "pixel_ids" in info:
                        with torch.no_grad():
                            gaussian_ids = info["gaussian_ids"]
                            pixel_ids = info["pixel_ids"]
                            
                            # Convert pixel_ids into coordinates.
                            y = pixel_ids // width
                            x = pixel_ids % width
                            
                            # Check which Gaussians land on the foreground.
                            is_foreground = foreground_mask[y, x] > 0.5
                            foreground_gaussians = gaussian_ids[is_foreground].unique()
                            
                            # Set Gaussians to black.
                            if self.cfg.app_opt:
                                # If using appearance module, adjust colors.
                                self.splats["colors"].data[foreground_gaussians] = torch.logit(torch.tensor(0.01, device=device))
                            else:
                                # Otherwise adjust sh0 (0th SH term is base color).
                                black_sh0 = rgb_to_sh(torch.zeros(1, 1, 3, device=device))
                                self.splats["sh0"].data[foreground_gaussians] = black_sh0

            # In frozen mode, skip pre_backward in strategy.
            # retain_grad() will fail because grads are not needed in frozen mode.
            if len(self.optimizers) > 1 or "foreground_logits" not in self.optimizers:
                # Normal mode: multiple optimizers, not only foreground_logits.
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )
            # else: frozen mode, skip step_pre_backward

            # Loss.
            # Check if we are in frozen mode (only train foreground_logits).
            is_frozen_mode = (len(self.optimizers) == 1 and "foreground_logits" in self.optimizers)
            
            if is_frozen_mode:
                # Frozen mode: only use foreground loss.
                # Render loss depends on frozen params and has no gradients.
                fg_loss = torch.tensor(0.0, device=device)
                
                # In frozen mode, start from step 0 (ignore warmup_steps).
                if cfg.foreground_loss:
                    image_id = image_ids[0].item()
                    if image_id in self.train_masks:
                        foreground_mask = self.train_masks[image_id]
                        # Ensure mask size matches.
                        if foreground_mask.shape[0] != height or foreground_mask.shape[1] != width:
                            foreground_mask = torch.nn.functional.interpolate(
                                foreground_mask.unsqueeze(0).unsqueeze(0),
                                size=(height, width),
                                mode="nearest",
                            ).squeeze(0).squeeze(0)
                        
                        fg_loss = self._compute_foreground_loss(
                            camtoworlds=camtoworlds,
                            Ks=Ks,
                            mask_gt=foreground_mask,
                            height=height,
                            width=width,
                        )
                        
                        if step == 0:
                            print(f"[Frozen Mode] Starting foreground_logits training from step 0")
                            print(f"  foreground_loss: {fg_loss.item():.6f}")
                            print(f"  mask_gt shape: {foreground_mask.shape}, sum: {foreground_mask.sum().item()}")
                
-                if fg_loss.item() > 0:
                    loss = cfg.foreground_lambda * fg_loss
                else:
                    # No effective foreground loss: create a dummy loss but do not skip.
                    # Skipping could leave the viewer lock unreleased.
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                    if step == 0:
                        print(f"Warning: No valid foreground loss in frozen mode at step {step}")
                        print(f"  image_id: {image_ids[0].item()}")
                        print(f"  Has mask: {image_ids[0].item() in self.train_masks}")
            else:
                # Normal mode: render loss + foreground loss.
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
                
                if cfg.depth_loss:
                    # Query depths from depth map.
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    depths = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths = depths.squeeze(3).squeeze(1)  # [1, M]
                    # Compute loss in disparity space.
                    disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda
                if cfg.use_bilateral_grid:
                    tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                    loss += tvloss

                # Regularizations.
                if cfg.opacity_reg > 0.0:
                    loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
                if cfg.scale_reg > 0.0:
                    loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

                fg_loss = torch.tensor(0.0, device=device)
                if(
                    cfg.foreground_loss
                    and step >= cfg.foreground_warmup_steps
                ):
                    image_id = image_ids[0].item()
                    if image_id in self.train_masks:
                        foreground_mask = self.train_masks[image_id]
                        # Ensure mask size matches.
                        if foreground_mask.shape[0] != height or foreground_mask.shape[1] != width:
                            foreground_mask = torch.nn.functional.interpolate(
                                foreground_mask.unsqueeze(0).unsqueeze(0),
                                size=(height, width),
                                mode="nearest",
                            ).squeeze(0).squeeze(0)
                        
                        fg_loss = self._compute_foreground_loss(
                            camtoworlds=camtoworlds,
                            Ks=Ks,
                            mask_gt=foreground_mask,
                            height=height,
                            width=width,
                        )
                        
                        # Only add to loss if computation succeeded
                        if fg_loss.item() > 0:
                            loss += cfg.foreground_lambda * fg_loss
                        
                        # Debug at the first warmup step.
                        if step == cfg.foreground_warmup_steps:
                            print(f"[Step {step}] Foreground loss: {fg_loss.item():.6f}")
                            print(f"  mask_gt shape: {foreground_mask.shape}, sum: {foreground_mask.sum().item()}")
                        elif fg_loss.item() == 0 and step == cfg.foreground_warmup_steps:
                            print(f"Warning: foreground loss is zero at step {step}, check implementation")

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if is_frozen_mode:
                # Frozen mode: only show foreground loss.
                desc += f"fg loss={fg_loss.item():.4f}| "
            else:
                # Normal mode: show all losses.
                if cfg.depth_loss:
                    desc += f"depth loss={depthloss.item():.6f}| "
                if cfg.foreground_loss and step >= cfg.foreground_warmup_steps:
                    desc += f"fg loss={fg_loss.item():.4f}| "
                if cfg.pose_opt and cfg.pose_noise:
                    # Monitor pose error if we inject noise.
                    pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                    desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # Write images (gt and render).
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                
                if is_frozen_mode:
                    # Frozen mode: log only foreground loss.
                    self.writer.add_scalar("train/fg_loss", fg_loss.item(), step)
                else:
                    # Normal mode: log all losses.
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    if cfg.depth_loss:
                        self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                    if cfg.use_bilateral_grid:
                        self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                    if cfg.tb_save_image:
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                
                self.writer.flush()

            # Save foreground renders every 500 steps.
            if world_rank == 0 and step % 500 == 0 and cfg.foreground_loss:
                render_dir = Path(self.cfg.result_dir) / "renders"
                render_dir.mkdir(exist_ok=True)
                
                # Get current training image info.
                image_id = image_ids[0].item()
                if image_id in self.train_masks:
                    foreground_mask = self.train_masks[image_id]
                    
                    # Ensure mask size matches.
                    if foreground_mask.shape[0] != height or foreground_mask.shape[1] != width:
                        foreground_mask = torch.nn.functional.interpolate(
                            foreground_mask.unsqueeze(0).unsqueeze(0),
                            size=(height, width),
                            mode="nearest",
                        ).squeeze(0).squeeze(0)
                    
                    # Render foreground probability map.
                    with torch.no_grad():
                        fg_probs = torch.sigmoid(self.splats["foreground_logits"])
                        viewmats = torch.linalg.inv(camtoworlds)
                        
                        from gsplat import rasterization
                        renders, _, _ = rasterization(
                            means=self.splats["means"],
                            quats=self.splats["quats"] / self.splats["quats"].norm(dim=-1, keepdim=True),
                            scales=torch.exp(self.splats["scales"]),
                            opacities=torch.sigmoid(self.splats["opacities"]),
                            colors=fg_probs.unsqueeze(-1).expand(-1, 3),
                            viewmats=viewmats,
                            Ks=Ks,
                            width=width,
                            height=height,
                            packed=self.cfg.packed,
                            render_mode="RGB",
                        )
                        
                        predicted_mask = renders[..., 0]
                        if predicted_mask.dim() == 3:
                            predicted_mask = predicted_mask[0]
                        
                        # Save images.
                        import imageio
                        
                        # Save predicted foreground map (grayscale).
                        pred_img = (predicted_mask.cpu().numpy() * 255).astype(np.uint8)
                        imageio.imwrite(render_dir / f"fg_pred_step{step:05d}_img{image_id:03d}.png", pred_img)
                        
                        # Save GT mask (only at step 0).
                        if step == 0:
                            gt_img = (foreground_mask.cpu().numpy() * 255).astype(np.uint8)
                            imageio.imwrite(render_dir / f"fg_gt_img{image_id:03d}.png", gt_img)
                        
                        # Save comparison (GT on left, prediction on right).
                        gt_np = foreground_mask.cpu().numpy()
                        pred_np = predicted_mask.cpu().numpy()
                        comparison = np.hstack([gt_np, pred_np])
                        comp_img = (comparison * 255).astype(np.uint8)
                        
                        # Get actual image filename for naming.
                        actual_image_name = self.parser.image_names[image_id]
                        actual_base_name = Path(actual_image_name).stem
                        
                        imageio.imwrite(render_dir / f"fg_compare_step{step:05d}_img{image_id:03d}_{actual_base_name}.png", comp_img)
                        
                        # Print stats.
                        if step % 500 == 0:
                            print(f"\n[Render Debug - Step {step}]")
                            print(f"  Image ID: {image_id} -> image file: {actual_image_name}")
                            print(f"  Mask file: {actual_base_name}.npy")
                            print(f"  GT mask - sum: {foreground_mask.sum().item():.0f}, mean: {foreground_mask.mean():.4f}")
                            print(f"  Predicted - min: {predicted_mask.min():.4f}, max: {predicted_mask.max():.4f}, mean: {predicted_mask.mean():.4f}")
                            print(f"  Foreground logits - min: {self.splats['foreground_logits'].min():.4f}, max: {self.splats['foreground_logits'].max():.4f}")
                            fg_prob_stats = torch.sigmoid(self.splats["foreground_logits"])
                            print(f"  Foreground prob - >0.9: {(fg_prob_stats > 0.9).sum().item()}, <0.1: {(fg_prob_stats < 0.1).sum().item()}")
                            print(f"  Saved to: {render_dir / f'fg_compare_step{step:05d}_img{image_id:03d}_{actual_base_name}.png'}")

            # Save checkpoint before updating the model.
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # Eval at origin to bake appearance into colors.
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn gradients into sparse tensors before optimizer step.
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # Optimize.
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer.
            # Skip in frozen mode (only train foreground_logits; no densification/pruning).
            if len(self.optimizers) > 1 or "foreground_logits" not in self.optimizers:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                else:
                    assert_never(self.cfg.strategy)

            if(
                cfg.remove_foreground_gaussians
                and cfg.remove_during_train
                and step > 0
                and step % cfg.removal_steps == 0
            ):
                image_id = image_ids[0].item()
                if image_id in self.train_masks:
                    foreground_mask = self.train_masks[image_id]
                    if foreground_mask.shape[0] != height or foreground_mask.shape[1] != width:
                        foreground_mask = torch.nn.functional.interpolate(
                            foreground_mask.unsqueeze(0).unsqueeze(0),
                            size=(height, width),
                            mode="nearest",
                        ).squeeze(0).squeeze(0)
                    gaussians_to_remove = self._identify_foreground_gaussians(
                        info, foreground_mask, height, width
                    )
                    if gaussians_to_remove.any():
                        keep_mask = ~gaussians_to_remove
                        for key in self.splats.keys():
                            self.splats[key] = torch.nn.Parameter(
                                self.splats[key][keep_mask].detach().clone()
                            )
                            self.splats[key].requires_grad_(True)
                        self._update_optimizer_after_removal(keep_mask)
                        for k, v in self.strategy_state.items():
                            if isinstance(v, torch.Tensor) and v.shape[0] == len(keep_mask):
                                self.strategy_state[k] = v[keep_mask]
                        print(
                            f"Step {step}: Removed {gaussians_to_remove.sum().item()} gaussians "
                            f"(remaining {len(self.splats['means'])})"
                        )

            # Eval the full set.
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # Run compression.
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Evaluation method - simplified version for OR_trainer."""
        print(f"Evaluation at step {step} - skipped (not implemented in OR_trainer)")
        # Note: Full evaluation implementation can be added here if needed
        pass

    @torch.no_grad()
    def render_traj(self, step: int):
        """Trajectory rendering - simplified version for OR_trainer."""
        if self.cfg.disable_video:
            return
        print(f"Trajectory rendering at step {step} - skipped (not implemented in OR_trainer)")
        pass

    @torch.no_grad()
    def run_compression(self, step: int):
        """Compression - simplified version for OR_trainer."""
        print(f"Compression at step {step} - skipped (not implemented in OR_trainer)")
        pass
    
    def _compute_foreground_loss_BCE_Dice(
        self,
        info: Dict,
        mask_gt: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Combine BCE Loss and Dice Loss
        Combined Loss = alpha * BCE + (1 - alpha) * Dice
        """
        bce_weight = 0.5  # Can be configured in Config.
        
        if self.cfg.packed:
            gaussian_ids = info.get("gaussian_ids", None)
            
            if gaussian_ids is None or gaussian_ids.shape[0] == 0:
                return torch.tensor(0.0, device=self.device)
            
            unique_gaussian_ids = torch.unique(gaussian_ids)
            max_gaussian_id = len(self.splats["means"]) - 1
            valid_mask = unique_gaussian_ids <= max_gaussian_id
            unique_gaussian_ids = unique_gaussian_ids[valid_mask]
            
            if unique_gaussian_ids.shape[0] == 0:
                return torch.tensor(0.0, device=self.device)
            
            if "means2d" in info:
                means2d = info["means2d"]
                if len(means2d.shape) == 3:
                    means2d = means2d[0]
                
                if unique_gaussian_ids.max() >= means2d.shape[0]:
                    valid_mask = unique_gaussian_ids < means2d.shape[0]
                    unique_gaussian_ids = unique_gaussian_ids[valid_mask]
                
                if unique_gaussian_ids.shape[0] == 0:
                    return torch.tensor(0.0, device=self.device)
                
                visible_means2d = means2d[unique_gaussian_ids]
                u = visible_means2d[:, 0].long().clamp(0, width - 1)
                v = visible_means2d[:, 1].long().clamp(0, height - 1)
                
                fg_gt_per_gaussian = mask_gt[v, u]
                fg_pred_logits = self.splats["foreground_logits"][unique_gaussian_ids]
                fg_pred_probs = torch.sigmoid(fg_pred_logits)
                
                # BCE Loss
                bce_loss = F.binary_cross_entropy_with_logits(
                    fg_pred_logits,
                    fg_gt_per_gaussian,
                    reduction='mean'
                )
                
                # Dice Loss
                smooth = 1e-5
                intersection = (fg_pred_probs * fg_gt_per_gaussian).sum()
                union = fg_pred_probs.sum() + fg_gt_per_gaussian.sum()
                dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_loss = 1.0 - dice
                
                # Combined loss.
                loss = bce_weight * bce_loss + (1 - bce_weight) * dice_loss
                
                return loss
            else:
                avg_mask = mask_gt.mean()
                fg_pred_logits = self.splats["foreground_logits"][unique_gaussian_ids]
                loss = F.binary_cross_entropy_with_logits(
                    fg_pred_logits,
                    avg_mask.expand_as(fg_pred_logits),
                    reduction='mean'
                )
                return loss
        else:
            if "radii" in info and "means2d" in info:
                radii = info["radii"]
                means2d = info["means2d"]
                
                if len(radii.shape) == 2:
                    radii_cam0 = radii[0]
                    means2d_cam0 = means2d[0]
                else:
                    radii_cam0 = radii
                    means2d_cam0 = means2d
                
                is_visible = radii_cam0 > 0
                
                if not is_visible.any():
                    return torch.tensor(0.0, device=self.device)
                
                visible_means2d = means2d_cam0[is_visible]
                u = visible_means2d[:, 0].long().clamp(0, width - 1)
                v = visible_means2d[:, 1].long().clamp(0, height - 1)
                
                fg_gt_per_gaussian = mask_gt[v, u]
                fg_pred_logits = self.splats["foreground_logits"][is_visible]
                fg_pred_probs = torch.sigmoid(fg_pred_logits)
                
                # BCE + Dice.
                bce_loss = F.binary_cross_entropy_with_logits(
                    fg_pred_logits,
                    fg_gt_per_gaussian,
                    reduction='mean'
                )
                
                smooth = 1e-5
                intersection = (fg_pred_probs * fg_gt_per_gaussian).sum()
                union = fg_pred_probs.sum() + fg_gt_per_gaussian.sum()
                dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_loss = 1.0 - dice
                
                loss = bce_weight * bce_loss + (1 - bce_weight) * dice_loss
                
                return loss
            else:
                return torch.tensor(0.0, device=self.device)

    def _compute_foreground_loss(
        self,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        mask_gt: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Render foreground probability map and compare with GT mask.
        Strategy: 
        1. Treat foreground_logits as per-Gaussian foreground probability
        2. Render these probabilities using Gaussian Splatting
        3. Compare rendered map with ground truth mask
        """

        # Convert logits into probability
        fg_probs = torch.sigmoid(self.splats["foreground_logits"])  # [N]

        # Compute viewmats from camtoworlds
        viewmats = torch.linalg.inv(camtoworlds)  # [C, 4, 4]

        # Use gsplat to render foreground probability map
        # Method: render fg_probs as "color" (single-channel repeated to RGB)
        renders, _, _ = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"] / self.splats["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=fg_probs.unsqueeze(-1).expand(-1, 3),  # [N, 3] - use fg_prob as RGB
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            render_mode="RGB",
        )

        # Extract predicted mask (take any channel since all 3 are identical)
        predicted_mask = renders[..., 0]  # [C, H, W] -> [H, W]
        if predicted_mask.dim() == 3:
            predicted_mask = predicted_mask[0]  # Take first camera
        
        # Ensure predicted values are in [0, 1] to avoid numerical issues.
        predicted_mask = torch.clamp(predicted_mask, 0.0, 1.0)

        # === Improvement 1: soften GT mask for richer gradients ===
        # Apply Gaussian blur to the binary mask to create soft labels.
        if self.cfg.use_soft_labels:
            # Lightly blur the GT mask to create soft edges.
            kernel_size = 5
            sigma = 1.0
            from torch.nn.functional import conv2d
            
            # Create Gaussian kernel.
            x = torch.arange(kernel_size, dtype=torch.float32, device=mask_gt.device) - kernel_size // 2
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
            
            # Apply Gaussian blur.
            mask_gt_soft = conv2d(
                mask_gt.unsqueeze(0).unsqueeze(0).float(),
                gauss_2d,
                padding=kernel_size // 2
            ).squeeze()
            
            # Critical: clamp to [0, 1].
            mask_gt_final = torch.clamp(mask_gt_soft, 0.0, 1.0)
        else:
            mask_gt_final = mask_gt.float()

        # === Improvement 2: combine multiple loss terms ===
        # BCE loss (base term).
        bce_loss = F.binary_cross_entropy(
            predicted_mask,
            mask_gt_final,
            reduction='mean'
        )
        
        # Focal loss to emphasize hard samples (near 0.5).
        if self.cfg.use_focal_loss:
            alpha = 0.25
            gamma = 2.0
            bce_per_pixel = F.binary_cross_entropy(
                predicted_mask,
                mask_gt_final,
                reduction='none'
            )
            pt = torch.exp(-bce_per_pixel)
            focal_loss = alpha * (1 - pt) ** gamma * bce_per_pixel
            focal_loss = focal_loss.mean()
        else:
            focal_loss = torch.tensor(0.0, device=predicted_mask.device)
        
        # Dice loss for global consistency.
        if self.cfg.use_dice_loss:
            intersection = (predicted_mask * mask_gt_final).sum()
            union = predicted_mask.sum() + mask_gt_final.sum()
            dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        else:
            dice_loss = torch.tensor(0.0, device=predicted_mask.device)
        
        # Combined loss.
        loss = bce_loss + 0.5 * focal_loss + 0.3 * dice_loss

        return loss

    def _render_foreground_map(
        self, 
        info: Dict,
        fg_probs: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Render a foreground probability map using Gaussian probabilities.

        Args:
            info: render info
            fg_probs: per-Gaussian foreground probability [N]
            height, width: image size

        Returns:
            fg_map: rendered foreground map [H, W]
        """
        # Render foreground prob similar to color rendering (single channel).
        # Simplified version: build from packed info.
        if self.cfg.packed and "gaussian_ids" in info and "pixel_ids" in info:
            gaussian_ids = info["gaussian_ids"]
            pixel_ids = info["pixel_ids"]
            weights = info.get("weights", None)  # Optional per-splat weights.
            
            fg_map = torch.zeros(height * width, device=self.device)
            fg_values = fg_probs[gaussian_ids]
            
            if weights is not None:
                # Weighted average using per-splat weights.
                fg_map.scatter_add_(0, pixel_ids, fg_values * weights)
            else:
                # Simple average.
                counts = torch.zeros(height * width, device=self.device)
                fg_map.scatter_add_(0, pixel_ids, fg_values)
                counts.scatter_add_(0, pixel_ids, torch.ones_like(fg_values))
                fg_map = fg_map / (counts + 1e-8)
            
            fg_map = fg_map.reshape(height, width)
        else:
            # If packed info is missing, return a zero map.
            fg_map = torch.zeros(height, width, device=self.device)
        
        return fg_map
    
    @torch.no_grad()
    def prune_by_foreground_prob(self):
        """Prune after training using learned foreground probabilities."""
        if not self.cfg.remove_foreground_gaussians:
            print("Post-prune: remove_foreground_gaussians is False, skipping.")
            return
        
        # Compute foreground probability.
        fg_prob = torch.sigmoid(self.splats["foreground_logits"])
        
        # Debug statistics
        print(f"Foreground probability statistics:")
        print(f"  Min: {fg_prob.min().item():.4f}")
        print(f"  Max: {fg_prob.max().item():.4f}")
        print(f"  Mean: {fg_prob.mean().item():.4f}")
        print(f"  Threshold: {self.cfg.foreground_thresh}")
        
        # Remove foreground Gaussians (probability above threshold).
        # High fg_prob => foreground, so remove fg_prob > thresh.
        keep_mask = fg_prob < self.cfg.foreground_thresh
        num_remove = (~keep_mask).sum().item()
        
        if num_remove == 0:
            print("Post-prune: nothing to remove (all probs <= threshold).")
            print("  Hint: Check if foreground_loss was computed during training.")
            return
        
        print(
            f"Post-prune: removing {num_remove} gaussians by prob>{self.cfg.foreground_thresh} "
            f"(before: {len(self.splats['means'])})"
        )
        
        # Remove foreground Gaussians.
        for key in list(self.splats.keys()):
            self.splats[key] = torch.nn.Parameter(
                self.splats[key][keep_mask].detach().clone()
            )
            self.splats[key].requires_grad_(True)
        
        # Update optimizers
        self._update_optimizers_after_removal(keep_mask)
        
        # Update strategy_state
        for k, v in self.strategy_state.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == len(keep_mask):
                self.strategy_state[k] = v[keep_mask]
        
        print(f"Post-prune: remaining {len(self.splats['means'])}")
        
        # Save pruned checkpoint.
        if self.cfg.post_prune_ckpt:
            data = {"step": "post_prune", "splats": self.splats.state_dict()}
            save_path = f"{self.ckpt_dir}/ckpt_pruned_rank{self.world_rank}.pt"
            torch.save(data, save_path)
            print(f"Saved pruned checkpoint to {save_path}")

        if self.cfg.save_ply:
            self.export_to_ply(step="pruned")

    @torch.no_grad()
    def export_to_ply(self, step: Union[int, str] = "pruned"):
        print(f"Exporting to PLY format (step={step})...")
        # Prepare colors from sh or app module
        if self.cfg.app_opt:
            # Evaluate at origin to bake appearance into colors
            rgb = self.app_module(
                features=self.splats["features"],
                embed_ids=None,
                dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                sh_degree=self.cfg.sh_degree,
            )
            rgb = rgb + self.splats["colors"]
            rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
            sh0 = rgb_to_sh(rgb)
            shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
        else:
            sh0 = self.splats["sh0"]
            shN = self.splats["shN"]
        
        # Export.
        export_splats(
            means=self.splats["means"],
            scales=self.splats["scales"],
            quats=self.splats["quats"],
            opacities=self.splats["opacities"],
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
        )
        
        print(f"  -> Saved to {self.ply_dir}/point_cloud_{step}.ply")
        print(f"  -> Number of gaussians: {len(self.splats['means'])}")

    def _viewer_render_fn(
        self, camera_state, render_tab_state
    ):
        """Render function for the viewer - simplified version for OR_trainer."""
        from simple_viewer import CameraState, GsplatRenderTabState
        
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.detach().cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            from simple_viewer import apply_float_colormap
            
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .detach()
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            from simple_viewer import apply_float_colormap
            
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).detach().cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    # Decide whether training is needed.
    should_train = (
        cfg.foreground_mask_to_black or 
        cfg.remove_foreground_gaussians or 
        cfg.freeze_scene_params or  # Frozen mode still needs training.
        cfg.foreground_loss  # Foreground loss enabled still needs training.
    )

    if cfg.ckpt is not None and not should_train:
        # Run eval only (no training-related flags are enabled).
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=False)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            # Skip keys that don't exist in checkpoint (e.g., foreground_logits from old checkpoints)
            if all(k in ckpt["splats"] for ckpt in ckpts):
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            else:
                print(f"Warning: Key '{k}' not found in checkpoint, keeping initialized value")
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        # Training mode (including foreground masking/removal mode).
        runner.train()
        # Post-train pruning.
        runner.prune_by_foreground_prob()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    python examples/OR_trainer.py --data_dir data/kitchen

    # With checkpoint for foreground masking
    python examples/OR_trainer.py \
        --data_dir data/Tree_Filled \
        --ckpt results/Tree/ckpts/ckpt_29999_rank0.pt \
        --foreground_mask_to_black True

    # Distributed training on 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/OR_trainer.py --data_dir data/kitchen
    """

    # Config objects
    configs = {
        "default": (
            "Gaussian splatting training with foreground masking support.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using MCMC strategy.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    
    # Parse config
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scalar)
    
    # Launch
    cli(main, cfg, verbose=True)