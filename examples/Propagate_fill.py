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

def get_bg_gaussians():
	return 

def create_new_gaussians():
	return 

def combine_gaussians(bg_gs: torch.Tensor, rd_gs: torch.Tensor):
	return

old_gaussians = get_bg_gaussians() # background gaussians from the initial reconstruction
new_gaussians = create_new_gaussians()  # create some new gaussians near the hole that we want to fill
all_gaussians = combine_gaussians(old_gaussians, new_gaussians)

optim = Adam(new_gaussians.parameters())  # only the new gaussians will be optimized

for step in range(steps):
	optim.zero_grad()
	poses = get_poses()  # k camera poses
	foreground_masks = get_foreground_masks(poses)  # k foreground masks
	rendered_images = render_images(all_gaussians, poses)  # k images
	ts = sample_timesteps()  # k timesteps
	alpha_ts = get_alphas(ts)  # k correponding alpha values from the noise schedule
	
	noise = randn_like(rendered_images)  # k gaussian noise images
	noisy_images = sqrt(alpha_ts) * rendered_images + sqrt(1 - alpha_ts) * noise  # k noisy rendered images

	with torch.no_grad():
		pred_noise = diffusion_model(noisy_images, ts)  # k noise predictions
	
	sds_loss = mean_squared_error(noise, pred_noise, reduction=None)  # k error images
	sds_loss = torch.where(foreground_masks, sds_loss, 0)  # set loss to 0 for background pixels
	sds_loss = sds_loss.mean()
	
	sds_loss.backward()
	optim.zero_grad()
	
