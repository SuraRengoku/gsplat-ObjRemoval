import argparse
import os
import struct
from pathlib import Path
from typing import Dict, Literal, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from datasets.colmap import Parser
from gsplat.rendering import rasterization
import matplotlib.pyplot as plt


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
	img_min = float(np.min(image))
	img_max = float(np.max(image))
	if img_max - img_min < 1e-8:
		return np.zeros_like(image, dtype=np.uint8)
	norm = (image - img_min) / (img_max - img_min)
	return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def infer_data_dir(colmap_dir: Path) -> Path:
	if (colmap_dir / "images").exists() and (colmap_dir / "sparse").exists():
		return colmap_dir
	if colmap_dir.name == "0" and colmap_dir.parent.name == "sparse":
		return colmap_dir.parent.parent
	if colmap_dir.name == "sparse":
		return colmap_dir.parent
	return colmap_dir


def load_splats(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
	checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
	if "splats" in checkpoint:
		checkpoint = checkpoint["splats"]

	required_keys = ["means", "quats", "scales", "opacities"]
	for key in required_keys:
		if key not in checkpoint:
			raise KeyError(f"Missing key '{key}' in checkpoint: {ckpt_path}")

	return {
		"means": checkpoint["means"].to(device),
		"quats": checkpoint["quats"].to(device),
		"scales": torch.exp(checkpoint["scales"].to(device)),
		"opacities": torch.sigmoid(checkpoint["opacities"].to(device)),
	}


@torch.no_grad()
def render_all_views(
	ckpt_path: Path,
	colmap_dir: Path,
	output_dir: Path,
	device: torch.device,
	near_plane: float,
	far_plane: float,
	camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"],
	packed: bool,
):
	data_dir = infer_data_dir(colmap_dir)
	parser = Parser(
		data_dir=str(data_dir),
		factor=1,
		normalize=True,
		test_every=8,
	)
	splats = load_splats(ckpt_path, device=device)

	alpha_dir = output_dir / "alpha"
	depth_dir = output_dir / "depth"
	alpha_dir.mkdir(parents=True, exist_ok=True)
	depth_dir.mkdir(parents=True, exist_ok=True)

	num_gaussians = splats["means"].shape[0]
	dummy_colors = torch.ones((num_gaussians, 3), dtype=torch.float32, device=device)

	print(f"[Info] data dir: {data_dir}")
	print(f"[Info] views: {len(parser.image_names)}")
	print(f"[Info] gaussians: {num_gaussians}")
	print(f"[Info] output: {output_dir}")

	for i, image_name in enumerate(tqdm(parser.image_names, desc="Rendering")):
		camera_id = parser.camera_ids[i]
		k = parser.Ks_dict[camera_id].astype(np.float32)
		width, height = parser.imsize_dict[camera_id]
		camtoworld = torch.from_numpy(parser.camtoworlds[i]).to(device=device, dtype=torch.float32).unsqueeze(0)
		viewmats = torch.linalg.inv(camtoworld)

		ks = torch.from_numpy(k).to(device=device, dtype=torch.float32).unsqueeze(0)

		render_depth, render_alpha, _ = rasterization(
			means=splats["means"],
			quats=splats["quats"],
			scales=splats["scales"],
			opacities=splats["opacities"],
			colors=dummy_colors,
			viewmats=viewmats,
			Ks=ks,
			width=width,
			height=height,
			near_plane=near_plane,
			far_plane=far_plane,
			render_mode="ED",
			packed=packed,
			camera_model=camera_model,
		)

		stem = Path(image_name).stem

		alpha = render_alpha[0, ..., 0].clamp(0.0, 1.0).cpu().numpy()
		alpha_uint8 = normalize_to_uint8(alpha)
		plt.imsave(alpha_dir / f"{stem}.png", alpha_uint8)

		depth = render_depth[0, ..., 0].cpu().numpy()
		np.save(depth_dir / f"{stem}.npy", depth)

		depth_uint16 = (depth * 1000).astype(np.uint16)
		
		valid_mask = depth > 0
		if valid_mask.any():
			d_min, d_max = depth[valid_mask].min(), depth[valid_mask].max()
			depth_vis = (depth - d_min) / (d_max - d_min + 1e-8)
			depth_vis[~valid_mask] = 0
		else:
			depth_vis = depth

		plt.imsave(depth_dir / f"{stem}.png", depth_vis, cmap='turbo')
		
	print("[Done] Render finished.")
	print(f"[Done] alpha maps: {alpha_dir}")
	print(f"[Done] depth maps: {depth_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Render alpha/depth maps for all COLMAP views from a 3DGS checkpoint."
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		required=True,
		help="Path to 3DGS checkpoint (.pt)",
	)
	parser.add_argument(
		"--colmap_dir",
		type=str,
		required=True,
		help="Path to COLMAP folder containing cameras.bin/images.bin or scene root with sparse/.",
	)
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		help="Output directory (will create alpha/ and depth/ subfolders)",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda",
		help="Torch device, e.g. cuda or cpu",
	)
	parser.add_argument(
		"--near_plane",
		type=float,
		default=0.01,
		help="Near clipping plane",
	)
	parser.add_argument(
		"--far_plane",
		type=float,
		default=1e10,
		help="Far clipping plane",
	)
	parser.add_argument(
		"--camera_model",
		type=str,
		default="pinhole",
		choices=["pinhole", "ortho", "fisheye", "ftheta"],
		help="Camera model for rasterization",
	)
	parser.add_argument(
		"--packed",
		action="store_true",
		help="Enable packed rasterization mode (more memory efficient)",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	device = torch.device(args.device)

	ckpt_path = Path(args.ckpt)
	colmap_dir = Path(args.colmap_dir)
	output_dir = Path(args.output)

	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	if not colmap_dir.exists():
		raise FileNotFoundError(f"COLMAP path not found: {colmap_dir}")

	os.makedirs(output_dir, exist_ok=True)

	render_all_views(
		ckpt_path=ckpt_path,
		colmap_dir=colmap_dir,
		output_dir=output_dir,
		device=device,
		near_plane=args.near_plane,
		far_plane=args.far_plane,
		camera_model=args.camera_model,
		packed=args.packed,
	)


if __name__ == "__main__":
	main()
