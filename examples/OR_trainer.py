# type: ignore

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
    packed: bool = False
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
    
    # Foreground masking mode: set foreground gaussians to black
    foreground_mask_to_black: bool = False  # 强制前景高斯为黑色
    green_threshold_lower: Tuple[float, float, float] = (0.0, 0.5, 0.0)  # 绿色下界 RGB
    green_threshold_upper: Tuple[float, float, float] = (0.5, 1.0, 0.5)  # 绿色上界 RGB 

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
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    
    # If checkpoint is provided, use it directly instead of initializing
    if checkpoint_splats is not None:
        # Create params list from checkpoint for optimizer creation
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

        # add foreground parameter, initialized as logit(0.5) = 0, which represents uncertainty
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
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps = 1e-15 / math.sqrt(BS),
            # TODO
            betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    
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
        )
        
        if checkpoint_splats is not None:
            print(f"Loaded checkpoint. Number of GS: {len(self.splats['means'])}")
        else:
            print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
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

        # Losses & Metrics.
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

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def _load_foreground_masks(self) -> Dict[int, torch.Tensor]:
        masks = {}
        
        if self.cfg.foreground_mask_to_black:
            # 从带绿色标记的图片中提取前景mask
            print("Detecting green foreground regions from images...")
            for image_id in self.trainset.indices:
                image_name = self.parser.image_names[image_id]
                image_path = Path(self.cfg.data_dir) / "images" / image_name
                
                if not image_path.exists():
                    print(f"Warning: Image {image_path} not found")
                    continue
                
                # 加载图片
                img = imageio.imread(image_path)
                if len(img.shape) == 2:  # 灰度图
                    print(f"Warning: Image {image_name} is grayscale, skipping")
                    continue
                    
                img = torch.from_numpy(img).float() / 255.0  # [H, W, 3], 归一化到 [0, 1]
                
                # 检测绿色区域
                lower = torch.tensor(self.cfg.green_threshold_lower, device=img.device)
                upper = torch.tensor(self.cfg.green_threshold_upper, device=img.device)
                
                # 绿色mask: 所有通道都在阈值范围内
                green_mask = (img >= lower) & (img <= upper)
                green_mask = green_mask.all(dim=-1).float()  # [H, W]
                
                masks[image_id] = green_mask.to(self.device)
            
            print(f"Detected green foreground in {len(masks)} images")
        else:
            # 原来的方式：从mask目录加载
            mask_dir = Path(self.cfg.data_dir) / "masks"
            
            if not mask_dir.exists():
                print(f"Warning: Mask directory {mask_dir} not found. Foreground loss will be disabled.")
                return masks
            
            for image_id in self.trainset.indices:
                image_name = self.parser.image_names[image_id]
                mask_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                    potential_path = mask_dir / f"{Path(image_name).stem}{ext}"
                    if potential_path.exists():
                        mask_path = potential_path
                        break
                
                if mask_path is not None:
                    mask = imageio.imread(mask_path)
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]
                    mask = torch.from_numpy(mask).float() / 255.0
                    masks[image_id] = mask.to(self.device)
                else:
                    print(f"Warning: Mask not found for image {image_name}")
            
            print(f"Loaded {len(masks)} foreground masks")
        
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
    
    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
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

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
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

            # 前景遮罩模式：将落在前景的高斯球颜色设为黑色
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
                    # 将前景区域的颜色设为黑色
                    colors = colors * (1.0 - foreground_mask.unsqueeze(0).unsqueeze(-1))
                    
                    # 同时更新高斯球的颜色参数
                    if cfg.packed and "gaussian_ids" in info and "pixel_ids" in info:
                        with torch.no_grad():
                            gaussian_ids = info["gaussian_ids"]
                            pixel_ids = info["pixel_ids"]
                            
                            # 将pixel_ids转换为坐标
                            y = pixel_ids // width
                            x = pixel_ids % width
                            
                            # 检查哪些高斯球落在前景
                            is_foreground = foreground_mask[y, x] > 0.5
                            foreground_gaussians = gaussian_ids[is_foreground].unique()
                            
                            # 将这些高斯球的颜色设为黑色
                            if self.cfg.app_opt:
                                # 如果使用appearance模块，调整colors参数
                                self.splats["colors"].data[foreground_gaussians] = torch.logit(torch.tensor(0.01, device=device))
                            else:
                                # 否则调整sh0 (球谐函数的第0项对应基础颜色)
                                black_sh0 = rgb_to_sh(torch.zeros(1, 1, 3, device=device))
                                self.splats["sh0"].data[foreground_gaussians] = black_sh0

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
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
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
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
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
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
                    # eval at origin to bake the appeareance into the colors
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

            # Turn Gradients into Sparse Tensor before running optimizer
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

            # optimize
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

            # Run post-backward steps after backward and optimizer
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

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
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

    def _compute_foreground_loss(
        self,
        info: Dict,
        mask_gt: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        if self.cfg.packed:
            # Packed 模式：使用 gaussian_ids 和 pixel_ids
            gaussian_ids = info["gaussian_ids"]  # [nnz]
            pixel_ids = info["pixel_ids"]  # [nnz]
            
            # 将 pixel_ids 转换为 (y, x) 坐标
            y = pixel_ids // width
            x = pixel_ids % width
            
            # 获取每个高斯球对应像素的 ground truth 前景值
            fg_gt_per_gaussian = mask_gt[y, x]  # [nnz]
            
            # 获取每个高斯球的前景预测概率
            fg_pred_logits = self.splats["foreground_logits"][gaussian_ids]  # [nnz]
            
            # 二元交叉熵损失
            loss = F.binary_cross_entropy_with_logits(
                fg_pred_logits, 
                fg_gt_per_gaussian,
                reduction='mean'
            )
            
        else:
            # 非 Packed 模式：使用 radii
            radii = info["radii"]  # [C, N] C是相机数量，N是高斯球数量
            
            # 对于每个高斯球，检查它在图像中的可见性
            is_visible = (radii > 0).any(0)  # [N]
            
            if not is_visible.any():
                return torch.tensor(0.0, device=self.device)
            
            # 获取可见高斯球的前景预测
            fg_pred_logits = self.splats["foreground_logits"][is_visible]  # [N_visible]
            
            # 渲染前景概率图
            # 这里需要使用高斯球的前景概率重新渲染一个前景图
            fg_probs = torch.sigmoid(self.splats["foreground_logits"])  # [N]
            fg_rendered = self._render_foreground_map(info, fg_probs, height, width)
            
            # 与 ground truth mask 比较
            loss = F.binary_cross_entropy(fg_rendered, mask_gt, reduction='mean')
        
        return loss
    
    def _render_foreground_map(
        self, 
        info: Dict,
        fg_probs: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        使用高斯球的前景概率渲染前景图
        
        Args:
            info: 渲染信息
            fg_probs: 每个高斯球的前景概率 [N]
            height, width: 图像尺寸
        
        Returns:
            fg_map: 渲染的前景图 [H, W]
        """
        # 使用类似于颜色渲染的方式，用前景概率替代颜色
        # 这需要调用底层的光栅化函数，但只渲染单通道
        
        # 简化版本：直接从 packed 信息构建
        if self.cfg.packed and "gaussian_ids" in info and "pixel_ids" in info:
            gaussian_ids = info["gaussian_ids"]
            pixel_ids = info["pixel_ids"]
            weights = info.get("weights", None)  # 如果有权重信息
            
            fg_map = torch.zeros(height * width, device=self.device)
            fg_values = fg_probs[gaussian_ids]
            
            if weights is not None:
                # 使用权重进行加权平均
                fg_map.scatter_add_(0, pixel_ids, fg_values * weights)
            else:
                # 简单平均
                counts = torch.zeros(height * width, device=self.device)
                fg_map.scatter_add_(0, pixel_ids, fg_values)
                counts.scatter_add_(0, pixel_ids, torch.ones_like(fg_values))
                fg_map = fg_map / (counts + 1e-8)
            
            fg_map = fg_map.reshape(height, width)
        else:
            # 如果没有 packed 信息，返回零图
            fg_map = torch.zeros(height, width, device=self.device)
        
        return fg_map

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

    if cfg.ckpt is not None and not cfg.foreground_mask_to_black:
        # run eval only (unless in foreground masking mode)
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=False)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        # Training mode (including foreground masking mode)
        runner.train()

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