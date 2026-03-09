import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import torch
import struct
from scipy.spatial import cKDTree # type: ignore
from typing import Tuple

def load_gaussian_means(ckpt_path, device="cpu"):
    print(f"loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "splats" in ckpt:
        ckpt = ckpt["splats"]

    if "means" in ckpt:
        means = ckpt["means"].to(device)
    else:
        raise KeyError(f"can not find params in pt file, currently existing keys: {list(ckpt.keys())}")

    return means

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_colmap_cameras(cam_path):
    cameras = {}
    CAMERA_MODELS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12} 

    with open(cam_path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            num_params = CAMERA_MODELS.get(model_id, 4)
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            cameras[cam_id] = {
                "width": width,
                "height": height,
                "params": params,
                "model_id": model_id,
            }
    
    return cameras

def read_colmap_images(img_path):
    images = {}

    with open(img_path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = struct.unpack("<c", f.read(1))[0]
                if char == b"\x00": break
                name += char.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24) # skip 2D point data
            images[img_id] = {"qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name}
    
    return images

def load_cameras_from_colmap(colmap_dir, device="cuda"):
    colmap_dir = Path(colmap_dir)

    cameras_data = read_colmap_cameras(colmap_dir / f"cameras.bin")
    images_data = read_colmap_images(colmap_dir / f"images.bin")

    cam_dict = {}
    for img_id, img in images_data.items():
        cam = cameras_data[img["cam_id"]]

        # 1. build viewmat(w2c matrix)
        R = qvec2rotmat(img["qvec"])
        t = img["tvec"]
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)

        # 2. build intrinsic matrix K
        params = cam["params"]
        if len(params) == 3:
            fx, fy = params[0], params[0]
            cx, cy = params[1], params[2]
        else:
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

        cam_dict[img["name"]] = {
            "viewmat": viewmat,
            "K": K,
            "width": cam["width"],
            "height": cam["height"]
        }

    return cam_dict

def build_deleted_and_kept_points_from_fg_bg(fg_means, bg_means):
    P_del = fg_means.detach().cpu().numpy()
    P_keep = bg_means.detach().cpu().numpy()
    if P_keep.shape[0] == 0:
        raise ValueError("background checkpoint has no gaussians")
    return P_del, P_keep


def select_boundary_gaussians(P_del, P_keep, k=8, tau=0.08, max_boundary_points=0):
    if P_del.shape[0] == 0 or P_keep.shape[0] == 0:
        return np.empty((0, 3), dtype=P_keep.dtype)

    tree = cKDTree(P_keep)
    knn_k = min(k, P_keep.shape[0])
    dists, idxs = tree.query(P_del, k=knn_k)
    idxs = np.atleast_2d(idxs)
    dists = np.atleast_2d(dists)

    valid = dists < tau
    if not np.any(valid):
        return np.empty((0, 3), dtype=P_keep.dtype)

    bnd_idx = np.unique(idxs[valid])
    B = P_keep[bnd_idx]

    if max_boundary_points > 0 and B.shape[0] > max_boundary_points:
        sample_idx = np.random.choice(B.shape[0], size=max_boundary_points, replace=False)
        B = B[sample_idx]

    return B


def load_fg_mask(fg_mask_path, H, W):
    fg_mask_bool = np.load(str(fg_mask_path))
    fg_mask_bin = (fg_mask_bool.astype(np.uint8)) * 255
    if fg_mask_bin.shape != (H, W):
        fg_mask_bin = cv2.resize(fg_mask_bin, (W, H), interpolation=cv2.INTER_NEAREST)
    return fg_mask_bin


def _cam_intrinsics_from_colmap_camera(cam: dict) -> Tuple[np.ndarray, int, int]:
    """
    Build K from COLMAP binary camera fields.
    Supported by model_id-based fallback:
      SIMPLE_PINHOLE(0), PINHOLE(1), SIMPLE_RADIAL(2), RADIAL(3), OPENCV(4), OPENCV_FISHEYE(5), FULL_OPENCV(6)
    """
    model_id = int(cam.get("model_id", -1))
    params = cam["params"]
    width = int(cam["width"])
    height = int(cam["height"])

    if model_id in [0, 2, 3]:
        fx = float(params[0])
        fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    else:
        fx = float(params[0])
        fy = float(params[1]) if len(params) > 1 else float(params[0])
        cx = float(params[2]) if len(params) > 2 else width * 0.5
        cy = float(params[3]) if len(params) > 3 else height * 0.5

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K, height, width


def _build_rays_for_rows(K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int, row_start: int, row_end: int) -> np.ndarray:
    """
    Build world-space rays [N, 6] for image rows [row_start, row_end).
    COLMAP uses x_cam = R @ x_world + t.
    """
    ys, xs = np.meshgrid(
        np.arange(row_start, row_end, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    ones = np.ones_like(xs, dtype=np.float32)
    pix = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3)

    Kinv = np.linalg.inv(K).astype(np.float32)
    dirs_cam = (pix @ Kinv.T).astype(np.float32)
    dirs_cam /= (np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-9)

    Rt = R.T.astype(np.float32)
    dirs_world = (dirs_cam @ Rt.T).astype(np.float32)
    dirs_world /= (np.linalg.norm(dirs_world, axis=1, keepdims=True) + 1e-9)

    cam_center = (-Rt @ t.reshape(3, 1)).reshape(1, 3).astype(np.float32)
    origins_world = np.repeat(cam_center, dirs_world.shape[0], axis=0)

    return np.concatenate([origins_world, dirs_world], axis=1).astype(np.float32)


def generate_split_surface_masks_from_colmap(
    split_mesh_path,
    colmap_dir,
    output_mask_dir,
    mask_suffix="_splitmask.png",
    dilate_size=0,
    ray_rows_chunk=256,
):
    """
    Reproject split surface mesh to each COLMAP view and output binary masks.
    """
    try:
        import open3d as o3d # type: ignore
    except Exception as exc:
        raise ImportError("open3d is required for split surface raycasting masks") from exc

    split_mesh_path = Path(split_mesh_path)
    colmap_dir = Path(colmap_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    cameras = read_colmap_cameras(colmap_dir / "cameras.bin")
    images = read_colmap_images(colmap_dir / "images.bin")

    mesh_legacy = o3d.io.read_triangle_mesh(str(split_mesh_path))
    if mesh_legacy.is_empty():
        raise RuntimeError(f"Empty mesh: {split_mesh_path}")

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    print(f"[split-mask] mesh: {split_mesh_path}")
    print(f"[split-mask] mesh triangles: {len(mesh_legacy.triangles):,}")
    print(f"[split-mask] images: {len(images):,}")

    for _, img in images.items():
        cam = cameras[img["cam_id"]]
        K, H, W = _cam_intrinsics_from_colmap_camera(cam)
        R = qvec2rotmat(img["qvec"]).astype(np.float32)
        t = np.asarray(img["tvec"], dtype=np.float32)

        mask = np.zeros((H, W), dtype=np.uint8)
        for row_start in range(0, H, ray_rows_chunk):
            row_end = min(row_start + ray_rows_chunk, H)
            rays_np = _build_rays_for_rows(K, R, t, H, W, row_start, row_end)
            rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
            hit = scene.cast_rays(rays)
            t_hit = hit["t_hit"].numpy().reshape(row_end - row_start, W)
            mask[row_start:row_end] = (np.isfinite(t_hit).astype(np.uint8) * 255)

        if dilate_size and dilate_size > 1:
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        out_name = f"{Path(img['name']).stem}{mask_suffix}"
        out_path = output_mask_dir / out_name
        cv2.imwrite(str(out_path), mask)

    print(f"[split-mask] saved to: {output_mask_dir}")


def build_hole_mask_from_knn(
    B,                # (N_b, 3) boundary gaussians
    K, R, t,          # camera intrinsics/extrinsics
    H, W,
    draw_r=4,
    fill_holes=False,
): 
    if B.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # 1. project B to image
    Xc = (R @ B.T + t.reshape(3, 1)).T
    z = Xc[:, 2]
    front = z > 1e-6
    Xc = Xc[front]
    z = z[front]

    uv = (K @ Xc.T).T
    uv = uv[:, :2] / z[:, None]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inside], v[inside]
    if u.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # 2. rasterize sparse points -> dense mask (vectorized)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v, u] = 255

    if draw_r > 0:
        kernel_size = draw_r * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if fill_holes:
        ff = mask.copy()
        h, w = ff.shape
        flood = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, flood, (0, 0), (255,))
        inv = cv2.bitwise_not(ff)
        mask = cv2.bitwise_or(mask, inv)

    return (mask > 0).astype(np.uint8)


def process_dataset(
    fg_ckpt_path,
    bg_ckpt_path,
    colmap_dir,
    output_dir,
    fg_mask_dir=None,
    k=8,
    tau=0.08,
    draw_r=4,
    max_boundary_points=0,
    fill_holes=False,
    close_ksize=7,
    close_iter=2,
    dilate_size=3,
):
    fg_means = load_gaussian_means(fg_ckpt_path, device="cpu")
    bg_means = load_gaussian_means(bg_ckpt_path, device="cpu")
    P_del, P_keep = build_deleted_and_kept_points_from_fg_bg(fg_means, bg_means)
    print(f"deleted gaussians: {len(P_del)}")
    print(f"kept gaussians: {len(P_keep)}")

    if len(P_del) == 0:
        print("no deleted gaussians found, output masks will be empty")

    B = select_boundary_gaussians(
        P_del=P_del,
        P_keep=P_keep,
        k=k,
        tau=tau,
        max_boundary_points=max_boundary_points,
    )
    print(f"boundary gaussians after KNN+tau: {len(B)}")

    cameras = load_cameras_from_colmap(colmap_dir, device="cpu")

    fg_mask_dir = None if fg_mask_dir is None else Path(fg_mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, cam_data in cameras.items():
        viewmat = cam_data["viewmat"].detach().cpu().numpy()
        K = cam_data["K"].detach().cpu().numpy()
        W = int(cam_data["width"])
        H = int(cam_data["height"])
        R = viewmat[:3, :3]
        t = viewmat[:3, 3]

        mask = build_hole_mask_from_knn(
            B=B,
            K=K,
            R=R,
            t=t,
            H=H,
            W=W,
            draw_r=draw_r,
            fill_holes=fill_holes,
        )

        final_mask = (mask * 255).astype(np.uint8)

        if close_ksize > 1 and close_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        if dilate_size > 0:
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        if fg_mask_dir is not None:
            fg_mask_path = fg_mask_dir / f"{Path(filename).stem}.npy"
            if fg_mask_path.exists():
                fg_mask = load_fg_mask(fg_mask_path, H, W)
                final_mask = cv2.bitwise_and(final_mask, fg_mask)

        output_path = output_dir / f"{Path(filename).stem}.png"
        os.makedirs(output_path.parent, exist_ok=True)
        cv2.imwrite(str(output_path), final_mask)
        
    print(f"saved masks to: {output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KNN-based inpainting masks from Gaussian checkpoints")
    parser.add_argument("--fg_ckpt", type=str, required=True, help="foreground-only checkpoint (treated as deleted gaussians)")
    parser.add_argument("--bg_ckpt", type=str, required=True, help="background-only checkpoint (treated as kept gaussians)")
    parser.add_argument("--colmap", type=str, required=True)
    parser.add_argument("--fg_dir", type=str, default=None, help="optional foreground mask directory (.npy per view)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--draw_r", type=int, default=4)
    parser.add_argument("--max_boundary_points", type=int, default=0, help="0 means no cap; set e.g. 200000 to speed up")
    parser.add_argument("--fill_holes", action="store_true", help="enable flood-fill hole completion (slower)")
    parser.add_argument("--close_ksize", type=int, default=7)
    parser.add_argument("--close_iter", type=int, default=2)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--split_mesh", type=str, default=None, help="optional split surface mesh path (.ply/.obj) for raycast reprojection masks")
    parser.add_argument("--split_mask_out", type=str, default=None, help="optional output directory for split surface masks")
    parser.add_argument("--split_mask_suffix", type=str, default="_splitmask.png", help="filename suffix for split-surface masks")
    parser.add_argument("--split_mask_dilate", type=int, default=0, help="optional dilation size for split-surface masks")
    parser.add_argument("--split_ray_rows", type=int, default=256, help="raycasting rows per chunk for split-surface reprojection")

    args = parser.parse_args()

    if args.close_ksize % 2 == 0:
        args.close_ksize += 1

    process_dataset(
        fg_ckpt_path=args.fg_ckpt,
        bg_ckpt_path=args.bg_ckpt,
        colmap_dir=args.colmap,
        output_dir=args.out_dir,
        fg_mask_dir=args.fg_dir,
        k=args.k,
        tau=args.tau,
        draw_r=args.draw_r,
        max_boundary_points=args.max_boundary_points,
        fill_holes=args.fill_holes,
        close_ksize=args.close_ksize,
        close_iter=args.close_iter,
        dilate_size=args.dilate,
    )

    if (args.split_mesh is None) ^ (args.split_mask_out is None):
        raise ValueError("--split_mesh and --split_mask_out must be provided together")

    if args.split_mesh is not None and args.split_mask_out is not None:
        if args.split_ray_rows < 1:
            raise ValueError("--split_ray_rows must be >= 1")
        generate_split_surface_masks_from_colmap(
            split_mesh_path=args.split_mesh,
            colmap_dir=args.colmap,
            output_mask_dir=args.split_mask_out,
            mask_suffix=args.split_mask_suffix,
            dilate_size=args.split_mask_dilate,
            ray_rows_chunk=args.split_ray_rows,
        )

    print("Done!")
    