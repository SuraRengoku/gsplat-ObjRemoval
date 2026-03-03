# type: ignore
"""
usage: python Prune.py 
    --ckpt <checkpoint_path> 
    --lthreshold <value> 
    --rthreshold <value> 
    [--output <output_dir>] 
    [--save_ply] 
    [--dbscan_clean]
    [--shell_recover]
    [--dbscan_eps 0.08]
    [--dbscan_min_samples 20]
    [--dbscan_max_points 80000]
    [--dbscan_chunk_size 100000]
    [--mesh_recover]
    [--mesh_method alpha|convex]
    [--mesh_alpha 0.03]
    [--mesh_scale 1.02]
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Tuple
import sys
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

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

def mask_splats(splats: Dict[str, torch.Tensor], keep_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Slice splat tensors (shape[0]==N) by keep_mask and keep other fields unchanged."""
    pruned_splats = {}
    n_total = int(keep_mask.shape[0])
    for key, value in splats.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n_total:
            pruned_splats[key] = value[keep_mask].clone()
        else:
            pruned_splats[key] = value
    return pruned_splats


def split_by_threshold(
    splats: Dict[str, torch.Tensor], lthreshold: float, rthreshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    prune gaussian ellipsoids based on foreground_logits
    
    Args:
        splats: gaussian ellipsoids param dict
        rule: fg_prob >= threshold 
    
    Returns:
        fg_mask, bg_mask (global masks on original splats)
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
    
    # threshold split
    fg_mask = (lthreshold <= fg_prob) & (fg_prob <= rthreshold)
    bg_mask = ~fg_mask
    num_bg = bg_mask.sum().item()
    num_fg = fg_mask.sum().item()
    
    print(f"\n{'='*80}")
    print(f"Pruning with lthreshold = {lthreshold:.4f} and rthreshold = {rthreshold:.4f}")
    print(f"{'='*80}")
    print(f"  Background ({num_bg:,}): fg_prob outside [{lthreshold:.4f}, {rthreshold:.4f}] ({num_bg/len(fg_prob)*100:.2f}%)")
    print(f"  Foreground ({num_fg:,}): {lthreshold:.4f} <= fg_prob <= {rthreshold:.4f} ({num_fg/len(fg_prob)*100:.2f}%)")
    
    # statistics on threshold foreground
    fg_prob_after = fg_prob[fg_mask]
    print(f"\n{'='*80}")
    print("Foreground probability statistics (AFTER threshold split):")
    print(f"{'='*80}")
    print(f"  Total gaussians: {int(fg_mask.sum().item()):,}")
    if fg_prob_after.numel() > 0:
        print(f"  Min probability: {fg_prob_after.min():.4f}")
        print(f"  Max probability: {fg_prob_after.max():.4f}")
        print(f"  Mean probability: {fg_prob_after.mean():.4f}")
        print(f"  Median probability: {fg_prob_after.median():.4f}")

    return fg_mask, bg_mask


def keep_largest_dbscan_cluster_mask(
    xyz: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10,
    max_points: int = 120000,
    chunk_size: int = 200000,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Keep only the largest spatial cluster of gaussians based on DBSCAN.

    Args:
        xyz: [N,3] point cloud
        eps: DBSCAN eps in world coordinate units
        min_samples: DBSCAN min_samples
        max_points: max points used for DBSCAN fitting (exact mode if N <= max_points)
        chunk_size: chunk size for full-set propagation
        random_seed: seed for reproducible sampling

    Returns:
        bool keep mask (length N), True for largest cluster points
    """
    num_before = xyz.shape[0]
    if num_before == 0:
        return np.zeros((0,), dtype=bool)

    run_exact = num_before <= max_points

    if run_exact:
        labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree").fit_predict(xyz)
        valid_mask = labels >= 0
        num_noise = int((~valid_mask).sum())
        if not valid_mask.any():
            print("Warning: DBSCAN found only noise. Keep all threshold-foreground gaussians unchanged.")
            return np.ones(num_before, dtype=bool)

        unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
        largest_label = int(unique_labels[np.argmax(counts)])
        keep_mask_np = labels == largest_label
    else:
        rng = np.random.default_rng(random_seed)
        sample_size = min(max_points, num_before)
        sample_idx = rng.choice(num_before, size=sample_size, replace=False)
        sample_xyz = xyz[sample_idx]

        sample_labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree").fit_predict(sample_xyz)
        sample_valid = sample_labels >= 0
        num_noise = int((~sample_valid).sum())
        if not sample_valid.any():
            print("Warning: DBSCAN(sampled) found only noise. Keep all threshold-foreground gaussians unchanged.")
            return np.ones(num_before, dtype=bool)

        unique_labels, counts = np.unique(sample_labels[sample_valid], return_counts=True)
        largest_label = int(unique_labels[np.argmax(counts)])
        largest_cluster_points = sample_xyz[sample_labels == largest_label]
        if largest_cluster_points.shape[0] == 0:
            print("Warning: Largest sampled cluster is empty. Keep all threshold-foreground gaussians unchanged.")
            return np.ones(num_before, dtype=bool)

        tree = KDTree(largest_cluster_points)
        keep_mask_np = np.zeros(num_before, dtype=bool)
        for start in range(0, num_before, chunk_size):
            end = min(start + chunk_size, num_before)
            counts_in_radius = tree.query_radius(xyz[start:end], r=eps, count_only=True)
            keep_mask_np[start:end] = counts_in_radius > 0

    num_keep = int(keep_mask_np.sum())
    num_remove = num_before - num_keep

    print(f"\n{'='*80}")
    print("DBSCAN clean (keep largest cluster):")
    print(f"{'='*80}")
    print(f"  Mode: {'exact' if run_exact else 'sampled'}")
    print(f"  eps: {eps}")
    print(f"  min_samples: {min_samples}")
    print(f"  max_points: {max_points}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  Input gaussians: {num_before:,}")
    print(f"  Noise points (DBSCAN set): {num_noise:,}")
    print(f"  Kept largest cluster label: {largest_label}")
    print(f"  Kept gaussians: {num_keep:,}")
    print(f"  Removed gaussians: {num_remove:,} ({num_remove/num_before*100:.2f}%)")

    return keep_mask_np


def recover_shell_points_to_foreground(
    means: torch.Tensor,
    fg_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    near_radius: float,
    shell_radius: float,
    shell_min_neighbors: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Find background points near foreground and move low-density shell points into foreground.

    Returns:
        global bool mask to move from background -> foreground
    """
    n_total = means.shape[0]
    move_to_fg = torch.zeros(n_total, dtype=torch.bool)

    fg_xyz = means[fg_mask].detach().cpu().numpy().astype(np.float32, copy=False)
    bg_idx = torch.where(bg_mask)[0]
    if fg_xyz.shape[0] == 0 or bg_idx.numel() == 0:
        return move_to_fg

    bg_xyz = means[bg_idx].detach().cpu().numpy().astype(np.float32, copy=False)

    # Step 1: suspicious background = near foreground
    fg_tree = KDTree(fg_xyz)
    suspicious = np.zeros(bg_xyz.shape[0], dtype=bool)
    for start in range(0, bg_xyz.shape[0], chunk_size):
        end = min(start + chunk_size, bg_xyz.shape[0])
        near_counts = fg_tree.query_radius(bg_xyz[start:end], r=near_radius, count_only=True)
        suspicious[start:end] = near_counts > 0

    suspicious_idx = np.where(suspicious)[0]
    if suspicious_idx.size == 0:
        print("No near-foreground background points found; skip shell recovery.")
        return move_to_fg

    suspicious_xyz = bg_xyz[suspicious_idx]

    # Step 2: strict radius outlier removal on suspicious subset
    sus_tree = KDTree(suspicious_xyz)
    neighbor_counts = sus_tree.query_radius(suspicious_xyz, r=shell_radius, count_only=True)
    ghost_local = neighbor_counts < shell_min_neighbors

    ghost_idx_bg_local = suspicious_idx[ghost_local]
    if ghost_idx_bg_local.size == 0:
        print("No shell-like ghosts detected; skip foreground recovery.")
        return move_to_fg

    ghost_idx_global = bg_idx[torch.from_numpy(ghost_idx_bg_local)]
    move_to_fg[ghost_idx_global] = True

    print(f"\n{'='*80}")
    print("Near-foreground shell recovery:")
    print(f"{'='*80}")
    print(f"  near_radius: {near_radius}")
    print(f"  shell_radius: {shell_radius}")
    print(f"  shell_min_neighbors: {shell_min_neighbors}")
    print(f"  Background candidates: {bg_idx.numel():,}")
    print(f"  Near-foreground suspicious: {suspicious_idx.size:,}")
    print(f"  Recovered to foreground: {ghost_idx_global.numel():,}")

    return move_to_fg


def _build_foreground_mesh(
    fg_xyz: np.ndarray,
    method: str,
    alpha: float,
) -> Tuple[o3d.geometry.TriangleMesh, str]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fg_xyz.astype(np.float64, copy=False))

    used_method = method
    if method == "alpha":
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map
        )
    elif method == "convex":
        mesh, _ = pcd.compute_convex_hull()
    else:
        raise ValueError(f"Unknown mesh method: {method}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if method == "alpha" and (len(mesh.vertices) == 0 or len(mesh.triangles) == 0 or not mesh.is_watertight()):
        mesh, _ = pcd.compute_convex_hull()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        used_method = "convex(fallback)"

    return mesh, used_method


def recover_points_inside_mesh_to_foreground(
    means: torch.Tensor,
    fg_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    method: str,
    alpha: float,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """
    Build foreground mesh, inflate from center, and move inside-background points to foreground.
    """
    n_total = means.shape[0]
    move_to_fg = torch.zeros(n_total, dtype=torch.bool)

    fg_idx = torch.where(fg_mask)[0]
    bg_idx = torch.where(bg_mask)[0]
    if fg_idx.numel() < 4 or bg_idx.numel() == 0:
        print("Mesh recovery skipped: foreground<4 points or background empty.")
        return move_to_fg

    fg_xyz = means[fg_idx].detach().cpu().numpy().astype(np.float32, copy=False)
    bg_xyz = means[bg_idx].detach().cpu().numpy().astype(np.float32, copy=False)

    try:
        mesh, used_method = _build_foreground_mesh(fg_xyz, method=method, alpha=alpha)
    except Exception as e:
        print(f"Mesh recovery skipped: mesh construction failed with error: {e}")
        return move_to_fg

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("Mesh recovery skipped: empty mesh.")
        return move_to_fg

    if scale != 1.0:
        vertices = np.asarray(mesh.vertices)
        center = vertices.mean(axis=0, keepdims=True)
        vertices_scaled = (vertices - center) * scale + center
        mesh.vertices = o3d.utility.Vector3dVector(vertices_scaled)

    if not mesh.is_watertight():
        print("Mesh recovery skipped: mesh is not watertight after construction/scale.")
        return move_to_fg

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    inside_bg = np.zeros(bg_xyz.shape[0], dtype=bool)
    for start in range(0, bg_xyz.shape[0], chunk_size):
        end = min(start + chunk_size, bg_xyz.shape[0])
        query = o3d.core.Tensor(bg_xyz[start:end], dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query).numpy()
        inside_bg[start:end] = occupancy > 0.5

    inside_local_idx = np.where(inside_bg)[0]
    if inside_local_idx.size == 0:
        print("Mesh recovery found no inside background points.")
        return move_to_fg

    inside_global_idx = bg_idx[torch.from_numpy(inside_local_idx)]
    move_to_fg[inside_global_idx] = True

    print(f"\n{'='*80}")
    print("Mesh-based foreground recovery:")
    print(f"{'='*80}")
    print(f"  method: {method} -> {used_method}")
    print(f"  alpha: {alpha}")
    print(f"  scale: {scale}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  Mesh vertices: {len(mesh.vertices):,}")
    print(f"  Mesh triangles: {len(mesh.triangles):,}")
    print(f"  Watertight: {mesh.is_watertight()}")
    print(f"  Background candidates: {bg_idx.numel():,}")
    print(f"  Recovered to foreground: {inside_global_idx.numel():,}")

    return move_to_fg

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


def resolve_output_paths(input_ckpt: str, output: str, lthreshold: float, rthreshold: float) -> Tuple[Path, Path]:
    input_path = Path(input_ckpt)
    default_tag = f"{input_path.stem}_lt{lthreshold:.2f}_rt{rthreshold:.2f}"

    if output is None:
        base_dir = input_path.parent
        base_name = default_tag
    else:
        out_path = Path(output)
        if out_path.suffix == ".pt":
            base_dir = out_path.parent
            base_name = out_path.stem
        else:
            base_dir = out_path
            base_name = default_tag

    fg_path = base_dir / f"{base_name}_fg.pt"
    bg_path = base_dir / f"{base_name}_bg.pt"
    return fg_path, bg_path

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
        help="Output directory or .pt prefix path. Produces both *_fg.pt and *_bg.pt"
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
    parser.add_argument(
        "--dbscan_clean",
        action="store_true",
        help="Use DBSCAN to remove floating gaussians"
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.05,
        help="DBSCAN eps in world units. Used only when --dbscan_clean is enabled."
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=10,
        help="DBSCAN min_samples. Used only when --dbscan_clean is enabled."
    )
    parser.add_argument(
        "--dbscan_max_points",
        type=int,
        default=120000,
        help="Max points for DBSCAN fitting. If gaussians exceed this, sampled mode is used."
    )
    parser.add_argument(
        "--dbscan_chunk_size",
        type=int,
        default=200000,
        help="Chunk size for propagating sampled DBSCAN largest cluster to all gaussians."
    )
    parser.add_argument(
        "--shell_recover",
        action="store_true",
        help="Enable near-foreground shell recovery (move sparse shell-like background points back to foreground)."
    )
    parser.add_argument(
        "--near_fg_radius",
        type=float,
        default=0.02,
        help="(Used with --shell_recover) background points within this radius to foreground are suspicious shell candidates."
    )
    parser.add_argument(
        "--shell_radius",
        type=float,
        default=0.02,
        help="(Used with --shell_recover) radius for strict local density check on suspicious background points."
    )
    parser.add_argument(
        "--shell_min_neighbors",
        type=int,
        default=8,
        help="(Used with --shell_recover) if neighbors within --shell_radius are fewer than this value, move point into foreground."
    )
    parser.add_argument(
        "--shell_chunk_size",
        type=int,
        default=200000,
        help="(Used with --shell_recover) chunk size used in near-foreground suspicious point search."
    )
    parser.add_argument(
        "--mesh_recover",
        action="store_true",
        help="Enable mesh-based recovery: build foreground mesh, inflate, and move inside background points to foreground."
    )
    parser.add_argument(
        "--mesh_method",
        type=str,
        choices=["alpha", "convex"],
        default="alpha",
        help="(Used with --mesh_recover) mesh construction method from foreground points."
    )
    parser.add_argument(
        "--mesh_alpha",
        type=float,
        default=0.03,
        help="(Used with --mesh_recover and alpha method) alpha-shape parameter."
    )
    parser.add_argument(
        "--mesh_scale",
        type=float,
        default=1.02,
        help="(Used with --mesh_recover) isotropic scale around mesh center for inflation."
    )
    parser.add_argument(
        "--mesh_chunk_size",
        type=int,
        default=200000,
        help="(Used with --mesh_recover) chunk size for background occupancy queries."
    )

    args = parser.parse_args()
    
    # check threshold range
    if not (0.0 <= args.lthreshold <= 1.0 and 0.0 <= args.rthreshold <= 1.0 and args.lthreshold <= args.rthreshold):
        print(
            "Error: Thresholds must satisfy 0.0 <= lthreshold <= rthreshold <= 1.0, "
            f"got lthreshold={args.lthreshold}, rthreshold={args.rthreshold}"
        )
        sys.exit(1)

    if args.dbscan_clean and args.dbscan_eps <= 0:
        print(f"Error: --dbscan_eps must be > 0, got {args.dbscan_eps}")
        sys.exit(1)

    if args.dbscan_clean and args.dbscan_min_samples < 1:
        print(f"Error: --dbscan_min_samples must be >= 1, got {args.dbscan_min_samples}")
        sys.exit(1)

    if args.dbscan_clean and args.dbscan_max_points < 1000:
        print(f"Error: --dbscan_max_points must be >= 1000, got {args.dbscan_max_points}")
        sys.exit(1)

    if args.dbscan_clean and args.dbscan_chunk_size < 10000:
        print(f"Error: --dbscan_chunk_size must be >= 10000, got {args.dbscan_chunk_size}")
        sys.exit(1)

    if args.shell_recover and (args.near_fg_radius <= 0 or args.shell_radius <= 0):
        print("Error: --near_fg_radius and --shell_radius must be > 0")
        sys.exit(1)

    if args.shell_recover and args.shell_min_neighbors < 1:
        print(f"Error: --shell_min_neighbors must be >= 1, got {args.shell_min_neighbors}")
        sys.exit(1)

    if args.shell_recover and args.shell_chunk_size < 10000:
        print(f"Error: --shell_chunk_size must be >= 10000, got {args.shell_chunk_size}")
        sys.exit(1)

    if args.mesh_recover and args.mesh_alpha <= 0:
        print(f"Error: --mesh_alpha must be > 0, got {args.mesh_alpha}")
        sys.exit(1)

    if args.mesh_recover and args.mesh_scale <= 0:
        print(f"Error: --mesh_scale must be > 0, got {args.mesh_scale}")
        sys.exit(1)

    if args.mesh_recover and args.mesh_chunk_size < 10000:
        print(f"Error: --mesh_chunk_size must be >= 10000, got {args.mesh_chunk_size}")
        sys.exit(1)
    
    # load checkpoint
    ckpt = load_checkpoint(args.ckpt)
    
    splats = ckpt["splats"]

    # threshold split -> initial foreground/background masks
    fg_mask, bg_mask = split_by_threshold(splats, args.lthreshold, args.rthreshold)

    # optional DBSCAN clean on foreground and flow removed points back to background
    if args.dbscan_clean:
        fg_xyz = splats["means"][fg_mask].detach().cpu().numpy().astype(np.float32, copy=False)
        fg_keep_local = keep_largest_dbscan_cluster_mask(
            fg_xyz,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            max_points=args.dbscan_max_points,
            chunk_size=args.dbscan_chunk_size,
        )
        fg_idx_global = torch.where(fg_mask)[0]
        fg_mask_after_dbscan = torch.zeros_like(fg_mask)
        fg_mask_after_dbscan[fg_idx_global[torch.from_numpy(fg_keep_local)]] = True
        dbscan_removed = int((fg_mask & ~fg_mask_after_dbscan).sum().item())
        print(f"  DBSCAN removed from foreground -> background: {dbscan_removed:,}")
        fg_mask = fg_mask_after_dbscan
        bg_mask = ~fg_mask

    # optional shell-like near-foreground background recovery to foreground
    if args.shell_recover:
        move_to_fg = recover_shell_points_to_foreground(
            means=splats["means"],
            fg_mask=fg_mask,
            bg_mask=bg_mask,
            near_radius=args.near_fg_radius,
            shell_radius=args.shell_radius,
            shell_min_neighbors=args.shell_min_neighbors,
            chunk_size=args.shell_chunk_size,
        )
    else:
        print("Shell recovery disabled. Use --shell_recover to enable.")
        move_to_fg = torch.zeros_like(fg_mask)
    fg_mask = fg_mask | move_to_fg
    bg_mask = ~fg_mask

    # optional mesh-based recovery to foreground
    if args.mesh_recover:
        move_to_fg_mesh = recover_points_inside_mesh_to_foreground(
            means=splats["means"],
            fg_mask=fg_mask,
            bg_mask=bg_mask,
            method=args.mesh_method,
            alpha=args.mesh_alpha,
            scale=args.mesh_scale,
            chunk_size=args.mesh_chunk_size,
        )
    else:
        print("Mesh recovery disabled. Use --mesh_recover to enable.")
        move_to_fg_mesh = torch.zeros_like(fg_mask)

    fg_mask = fg_mask | move_to_fg_mesh
    bg_mask = ~fg_mask

    fg_splats = mask_splats(splats, fg_mask)
    bg_splats = mask_splats(splats, bg_mask)

    print(f"\n{'='*80}")
    print("Final split summary:")
    print(f"{'='*80}")
    print(f"  Final foreground: {int(fg_mask.sum().item()):,}")
    print(f"  Final background: {int(bg_mask.sum().item()):,}")
    
    # store
    if not args.dry_run:
        fg_output_path, bg_output_path = resolve_output_paths(
            args.ckpt, args.output, args.lthreshold, args.rthreshold
        )

        # create new checkpoints
        fg_ckpt = {
            "step": f"foreground_lt{args.lthreshold:.2f}_rt{args.rthreshold:.2f}",
            "splats": fg_splats,
        }
        bg_ckpt = {
            "step": f"background_lt{args.lthreshold:.2f}_rt{args.rthreshold:.2f}",
            "splats": bg_splats,
        }
        
        # keep other fields (if exist)
        for key in ckpt.keys():
            if key not in ["splats", "step"]:
                fg_ckpt[key] = ckpt[key]
                bg_ckpt[key] = ckpt[key]
        
        save_checkpoint(fg_ckpt, str(fg_output_path))
        save_checkpoint(bg_ckpt, str(bg_output_path))

        # Export to PLY if requested
        if args.save_ply:
            fg_pt_path = Path(fg_output_path)
            bg_pt_path = Path(bg_output_path)

            fg_result_dir = fg_pt_path.parent.parent
            bg_result_dir = bg_pt_path.parent.parent

            fg_ply_path = (fg_result_dir / "ply") / f"{fg_pt_path.stem}.ply"
            bg_ply_path = (bg_result_dir / "ply") / f"{bg_pt_path.stem}.ply"

            export_to_ply(fg_splats, str(fg_ply_path))
            export_to_ply(bg_splats, str(bg_ply_path))

    else:
        print(f"\n{'='*80}")
        print("DRY RUN: No files were saved.")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()