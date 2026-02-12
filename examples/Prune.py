# type: ignore
"""
usage: python Prune.py --ckpt <checkpoint_path> --lthreshold <value> --rthreshold <value> [--output <output_dir>] --save_ply
"""

import argparse
import torch
from pathlib import Path
from typing import Dict
import sys

from gsplat import export_splats

def load_checkpoint(ckpt_path: str) -> Dict:
    """load checkpoint file"""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "splats" not in ckpt:
        raise ValueError("Invalid checkpoint: missing 'splats' key")
    
    return ckpt

def prune_by_threshold(splats: Dict[str, torch.Tensor], lthreshold: float, rthreshold: float) -> Dict[str, torch.Tensor]:
    """
    prune gaussian ellipsoids based on foreground_logits
    
    Args:
        splats: gaussian ellipsoids param dict
        rule: fg_prob >= threshold 
    
    Returns:
        gaussian ellipsoids param dict after pruning
    """
    if "foreground_logits" not in splats:
        raise ValueError("Checkpoint does not contain 'foreground_logits'. Please train with foreground loss first.")
    
    # load foreground_logits
    fg_logits = splats["foreground_logits"]
    fg_prob = torch.sigmoid(fg_logits)
    
    # statistics
    print(f"\n{'='*80}")
    print("Foreground probability statistics (BEFORE pruning):")
    print(f"{'='*80}")
    print(f"  Total gaussians: {len(fg_prob):,}")
    print(f"  Min probability: {fg_prob.min():.4f}")
    print(f"  Max probability: {fg_prob.max():.4f}")
    print(f"  Mean probability: {fg_prob.mean():.4f}")
    print(f"  Median probability: {fg_prob.median():.4f}")
    print(f"  Std deviation: {fg_prob.std():.4f}")
    
    # distribution
    print(f"\nProbability distribution:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins)-1):
        mask = (fg_prob >= bins[i]) & (fg_prob < bins[i+1])
        count = mask.sum().item()
        percentage = count / len(fg_prob) * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:8,} ({percentage:5.2f}%)")
    
    # create keep mask(low foreground_logits)
    keep_mask = (lthreshold <= fg_prob) & (fg_prob <= rthreshold)
    num_remove = (~keep_mask).sum().item()
    num_keep = keep_mask.sum().item()
    
    print(f"\n{'='*80}")
    print(f"Pruning with lthreshold = {lthreshold:.4f} and rthreshold = {rthreshold:.4f}")
    print(f"{'='*80}")
    print(f"  Gaussians to REMOVE (fg_prob < {lthreshold:.4f} && fg_prob > {rthreshold:.4f}): {num_remove:,} ({num_remove/len(fg_prob)*100:.2f}%)")
    print(f"  Gaussians to KEEP ({lthreshold:.4f} <= fg_prob <= {rthreshold:.4f}): {num_keep:,} ({num_keep/len(fg_prob)*100:.2f}%)")
    
    if num_remove == 0:
        print("\nWarning: No gaussians will be removed with this threshold!")
        print("  Consider lowering the threshold value.")
        return splats
    

    print(f"\nPruning parameters...")
    pruned_splats = {}
    for key, value in splats.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == len(keep_mask):
            pruned_splats[key] = value[keep_mask].clone()
            print(f"  {key:20s}: {str(value.shape):25s} -> {str(pruned_splats[key].shape)}")
        else:
            pruned_splats[key] = value
            if isinstance(value, torch.Tensor):
                print(f"  {key:20s}: {str(value.shape):25s} (unchanged)")
    
    # statistics after pruning
    fg_prob_after = torch.sigmoid(pruned_splats["foreground_logits"])
    print(f"\n{'='*80}")
    print("Foreground probability statistics (AFTER pruning):")
    print(f"{'='*80}")
    print(f"  Total gaussians: {len(fg_prob_after):,}")
    print(f"  Min probability: {fg_prob_after.min():.4f}")
    print(f"  Max probability: {fg_prob_after.max():.4f}")
    print(f"  Mean probability: {fg_prob_after.mean():.4f}")
    print(f"  Median probability: {fg_prob_after.median():.4f}")
    
    return pruned_splats

def save_checkpoint(ckpt: Dict, output_path: str):
    """store checkpoint after pruning"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(ckpt, output_path)
    print(f"\n{'='*80}")
    print(f"Saved pruned checkpoint to: {output_path}")
    print(f"{'='*80}")
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"  File size: {size_mb:.1f} MB")

def export_to_ply(splats: Dict[str, torch.Tensor], ply_path: str):
    """
    Export splats to PLY format
    
    Args:
        splats: gaussian ellipsoids param dict
        ply_path: output PLY file path
    """
    print(f"\n{'='*80}")
    print("Exporting to PLY format...")
    print(f"{'='*80}")
    
    ply_path = Path(ply_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract required parameters
    means = splats["means"]
    scales = splats["scales"]
    quats = splats["quats"]
    opacities = splats["opacities"]
    
    # Handle spherical harmonics
    if "sh0" in splats and "shN" in splats:
        sh0 = splats["sh0"]
        shN = splats["shN"]
    elif "colors" in splats:
        # If using colors instead of SH, convert to sh0
        colors = splats["colors"]
        if colors.dim() == 2:
            colors = colors.unsqueeze(1)  # [N, 3] -> [N, 1, 3]
        from utils import rgb_to_sh
        sh0 = rgb_to_sh(torch.sigmoid(colors))
        shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
    else:
        raise ValueError("Checkpoint must contain either (sh0, shN) or colors")
    
    # Export to PLY
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(ply_path),
    )
    
    size_mb = ply_path.stat().st_size / (1024**2)
    print(f"  Saved PLY to: {ply_path}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Number of gaussians: {len(means):,}")

def main():
    parser = argparse.ArgumentParser(description="Prune Gaussian Splatting checkpoint by foreground probability threshold")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to input checkpoint (e.g., results/Tree/ckpts/ckpt_4999_rank0.pt)"
    )
    parser.add_argument(
        "--lthreshold",
        type=float,
        required=True,
        help="Foreground probability left threshold. Gaussians with fg_prob < lthreshold will be REMOVED."
    )
    parser.add_argument(
        "--rthreshold",
        type=float,
        required=True,
        help="Foreground probability right threshold. Gaussians with fg_prob > threshold will be REMOVED."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for pruned checkpoint. Default: same directory as input with '_pruned' suffix"
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Also export pruned gaussians to PLY format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show statistics without saving pruned checkpoint"
    )
    
    args = parser.parse_args()
    
    # check threshold range
    if not 0.0 <= args.lthreshold <= 1.0 and 0.0 <= args.rthreshold <= 1.0 and args.lthreshold <= args.rthreshold:
        print(f"Error: Threshold must be in range [0.0, 1.0], got {args.threshold}")
        sys.exit(1)
    
    # load checkpoint
    ckpt = load_checkpoint(args.ckpt)
    
    # pruning
    pruned_splats = prune_by_threshold(ckpt["splats"], args.lthreshold, args.rthreshold)
    
    # store
    if not args.dry_run:
        if args.output is None:
            # default output filename：+ _pruned
            input_path = Path(args.ckpt)
            output_path = input_path.parent / f"{input_path.stem}_pruned_lt{args.lthreshold:.2f}_rt{args.rthreshold:.2f}.pt"
        else:
            output_path = Path(args.output)
            if output_path.is_dir():
                # if it is directory, automatically generate filename
                input_name = Path(args.ckpt).stem
                output_path = output_path / f"{input_name}_pruned_lt{args.lthreshold:.2f}_rt{args.rthreshold:.2f}.pt"
        
        # create new checkpoint
        pruned_ckpt = {
            "step": f"pruned_lt{args.lthreshold:.2f}_rt{args.rthreshold:.2f}",
            "splats": pruned_splats
        }
        
        # keep other fields (if exist)
        for key in ckpt.keys():
            if key not in ["splats", "step"]:
                pruned_ckpt[key] = ckpt[key]
        
        save_checkpoint(pruned_ckpt, str(output_path))

        # Export to PLY if requested
        if args.save_ply:
            # Determine PLY output path
            # Structure: results/Tree/ckpts/xxx.pt -> results/Tree/ply/xxx.ply
            pt_path = Path(output_path)
            result_dir = pt_path.parent.parent  # Go up from ckpts/ to results/Tree/
            ply_dir = result_dir / "ply"
            ply_filename = f"{pt_path.stem}.ply"  # e.g., ckpt_4999_rank0_pruned_t0.30.ply
            ply_path = ply_dir / ply_filename
            
            export_to_ply(pruned_splats, str(ply_path))

    else:
        print(f"\n{'='*80}")
        print("DRY RUN: No files were saved.")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()