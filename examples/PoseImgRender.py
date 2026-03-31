"""
Usage:
python PoseImgRender.py 
    --ckpt 
    --colmap_dir [no need to direct to sparse/0]
    --output_dir
    --factor
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm

from gsplat.rendering import rasterization
from datasets.colmap import Parser

def main():
    parser = argparse.ArgumentParser(description="Render images from a gsplat checkpoint using COLMAP poses.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the gsplat .pt checkpoint file")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Path to the COLMAP folder containing sparse/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for rendered images")
    parser.add_argument("--factor", type=int, default=1, help="Downsample factor for COLMAP cameras")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt_data = torch.load(args.ckpt, map_location=device)
    if "splats" in ckpt_data:
        ckpt = ckpt_data["splats"]
    else:
        ckpt = ckpt_data

    means = ckpt["means"] # [N, 3]
    # quats = F.normalize(ckpt["quats"], p=2, dim=-1) 
    # rasterization does normalization internally 
    quats = ckpt["quats"] # [N, 4]
    scales = torch.exp(ckpt["scales"]) # [N, 3]
    opacities = torch.sigmoid(ckpt["opacities"]) # [N,]
    
    # Handle colors/sh
    if "sh0" in ckpt and "shN" in ckpt:
        sh0 = ckpt["sh0"]
        shN = ckpt["shN"]
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    else:
        colors = ckpt["colors"] if "colors" in ckpt else ckpt["features_dc"]
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1) if colors.ndim > 2 else 0
        if colors.ndim == 2:
            colors = colors.unsqueeze(1) # Add SH dimension if purely RGB

    print("Number of Gaussians:", len(means))

    # Load cameras
    print(f"Loading COLMAP poses from {args.colmap_dir}")
    colmap_parser = Parser(data_dir=args.colmap_dir, factor=args.factor, normalize=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Rendering {len(colmap_parser.image_names)} images...")
    
    for i in tqdm(range(len(colmap_parser.image_names))):
        cam_id = colmap_parser.camera_ids[i]
        image_name = colmap_parser.image_names[i]
        
        K_np = colmap_parser.Ks_dict[cam_id]
        width, height = colmap_parser.imsize_dict[cam_id]
        camtoworld = torch.from_numpy(colmap_parser.camtoworlds[i]).to(device=device, dtype=torch.float32).unsqueeze(0)
        viewmats = torch.linalg.inv(camtoworld)

        K = torch.from_numpy(K_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        
        backgrounds = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        
        with torch.no_grad():
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=K,
                width=width,
                height=height,
                sh_degree=sh_degree,
                render_mode="RGB",
                packed=False,
                backgrounds=backgrounds,
            )
            
            # shape of render_colors: [1, H, W, 3]
            img_rgb = render_colors[0, ..., 0:3].clamp(0, 1).cpu().numpy()
            img_rgb_8bit = (img_rgb * 255.0).astype(np.uint8)
            
            # Use original filename without extension + png, or just keep original name if saving png
            out_name = os.path.splitext(os.path.basename(image_name))[0] + ".png"
            out_path = os.path.join(args.output_dir, out_name)
            
            imageio.imwrite(out_path, img_rgb_8bit)

    print("Done!")

if __name__ == "__main__":
    main()
