# type: ignore
import os
import sys
import torch
from pathlib import Path
from dataclasses import dataclass
import tyro
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import StableDiffusionInpaintPipeline

@dataclass
class Config:
    # Directory containing the original RGB images
    image_dir: str = "data/kitchen/images"
    # Directory containing the binary masks (white = hole to fill)
    mask_dir: str = "data/kitchen/holeMask"
    # Where to save the 2D repainted images
    result_dir: str = "results/kitchen_2D_inpaints"
    # Prompt for Stable Diffusion
    prompt: str = "background behind the object, perfectly matching the surrounding environment"
    negative_prompt: str = "artifacts, blurry, low quality"
    # Checkpoint
    sd_model_id: str = "runwayml/stable-diffusion-inpainting"
    # Inference steps (often SDS uses fewer, but for 2D generation we want a good number, e.g., 50)
    num_inference_steps: int = 50
    # CFG scale
    guidance_scale: float = 7.5

def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading pipeline {cfg.sd_model_id} on {device}...")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        cfg.sd_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    # pipeline.enable_model_cpu_offload() # Uncomment if running out of VRAM
    
    img_dir = Path(cfg.image_dir)
    mask_dir = Path(cfg.mask_dir)
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists():
        print(f"Image directory {img_dir} does not exist!")
        return
        
    if not mask_dir.exists():
        print(f"Mask directory {mask_dir} does not exist!")
        return

    # Gather images
    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    
    print(f"Found {len(image_paths)} images. Commencing 2D inpainting...")
    
    for img_path in tqdm(image_paths):
        stem = img_path.stem
        
        # Try to find corresponding mask
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.jpg"
            
        if not mask_path.exists():
            print(f"\nSkipping {stem}, no mask found.")
            continue
            
        # Load and resize images to be safe (SD typically requires multiples of 8)
        # We process them at their original size, but handle dimension constraints if needed.
        init_image = Image.open(img_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
        
        # Ensure image dimensions are multiples of 8
        w, h = init_image.size
        w = w - w % 8
        h = h - h % 8
        init_image = init_image.resize((w, h))
        mask_image = mask_image.resize((w, h), Image.Resampling.NEAREST)
        
        # Run inference
        result = pipeline(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        ).images[0]
        
        # Save
        result.save(out_dir / f"{stem}.png")
        
    print(f"All done! Results saved to {out_dir}")

def _normalize_cli_underscores(argv):
    normalized = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
            key = key.replace("_", "-")
            normalized.append(f"{key}={val}")
        elif arg.startswith("--"):
            normalized.append(arg.replace("_", "-"))
        else:
            normalized.append(arg)
    return normalized

if __name__ == "__main__":
    cfg = tyro.cli(Config, args=_normalize_cli_underscores(sys.argv[1:]))
    main(cfg)
