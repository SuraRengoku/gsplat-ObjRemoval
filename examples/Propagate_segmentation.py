# type: ignore
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from PIL import Image
import cv2
from tqdm import tqdm

# pipeline
# 1.(Single pixel segmentation) user click -> get pixel coordinate(u1, v1) -> SAM -> Mask1
# 2.(Unprojection) pixel coordinate(u1, v1) + depth(d1) + camera paramaters(K, R1, t1) -> Hit point coordinates(P_world)
# 3.(Reprojection) P_world + camera parameters(K, R2, t2) -> pixel coordinate(u2, v2)
# 4.(Propogation segmentation) pixel coordinate(u2, v2) -> SAM -> Mask2

# TODO
# use depth value to dynamically decide whether selected uv points are valid in a single propagation iteration

from BinaryReader import read_model, get_camera_center_and_rotation, qvec2rotmat, read_colmap_depth

from segment_anything import sam_model_registry, SamPredictor

from sam2.build_sam import build_sam2_video_predictor 

try:
    from interactive_point_selector import get_start_point_interactive
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print("Warning: interactive_point_selector not available. Using manual point selection.")

def get_intrinsic_matrix(camera):
    # generate intrinsic matrix K from COLMAP camera model
    # SIMPLE_PINHOLE: f, cx, cy
    if camera.model == 0 or camera.model_name == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = fy = f
    # PINHOLE: fx, fy, cx, cy
    elif camera.model == 1 or camera.model_name == "PINHOLE":
        fx, fy, cx, cy = camera.params
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def unproject_pixel(u, v, depth, K, R_w2c, t_w2c):
    # project pixel coordinate (u, v) into 3D point in world coordinates

    # 1. pixel coordinate -> normalized camera coordinates (x_c, y_c, 1) * depth
    # P_cam = K_inv * [u, v, 1]^T * depth
    uv_homo = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    P_cam = np.dot(K_inv, uv_homo) * depth

    # 2. camera coordinates -> world coordinates
    # P_cam = R * P_world + t => P_world = R^T * (P_cam - t)
    P_world = np.dot(R_w2c.T, (P_cam - t_w2c))

    return P_world # hit point

def reproject_point(P_world, K, R_w2c, t_w2c):
    # project hit point from world coordinates to pixel coordinates

    # 1. world coordinates -> camera coordinates
    # P_cam = R * P_world + t
    P_cam = np.dot(R_w2c, P_world) + t_w2c

    if P_cam[2] <= 0:
        return None
    
    # 2. camera coordinates -> pixel coordinates
    # uv_homo = K * P_cam
    uv_homo = np.dot(K, P_cam)
    u = uv_homo[0] / uv_homo[2]
    v = uv_homo[1] / uv_homo[2]

    return np.array([u, v])

def run_propagation(
    colmap_path,
    result_path,
    images_dir,
    depths_dir,
    start_img_name,
    start_point, 
    sam_checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth",
    use_colmap_depth=True
):
    """ update the 3D world coordinate after each propagation """
    if not os.path.exists(result_path):
        print(f"Creating result directory: {result_path}")
        os.makedirs(result_path, exist_ok=True)

    # 1. load COLMAP data
    print("Loading COLMAP model...")
    cameras, images, _ = read_model(colmap_path) 
    # sort image based on filenames
    sorted_img_ids = sorted(images.keys(), key = lambda k: images[k].name) 

    # find the initial index of images
    start_idx = -1
    for i, img_id in enumerate(sorted_img_ids):
        if images[img_id].name == start_img_name:
            start_idx = i
            break
    
    if start_idx == -1:
        raise ValueError(f"Start image {start_img_name} not found in COLMAP model")
    
    # 2. initialize SAM
    print("Loading SAM model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 3. preprocess initial frame
    curr_img_id = sorted_img_ids[start_idx]
    curr_img_data = images[curr_img_id]
    curr_cam = cameras[curr_img_data.camera_id]

    # read images and depth maps
    img_path = os.path.join(images_dir, curr_img_data.name)
    
    # Read depth map (支持COLMAP和gsplat两种格式)
    if use_colmap_depth:
        depth_filename = curr_img_data.name + ".geometric.bin"
        depth_path = os.path.join(depths_dir, depth_filename)
        if not os.path.exists(depth_path):
            depth_filename = curr_img_data.name + ".photometric.bin"
            depth_path = os.path.join(depths_dir, depth_filename)
        depth_map = read_colmap_depth(depth_path)
        print(f"Loaded COLMAP depth: {depth_path}")
    else:
        depth_path = os.path.join(depths_dir, os.path.splitext(curr_img_data.name)[0] + ".npy")
        depth_map = np.load(depth_path)
        print(f"Loaded gsplat depth: {depth_path}")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get depth of picked pixel
    u, v = start_point
    # NOTE: the depth map might be scaled, we suppose that the size of depth map keeps same with the size of original image
    # if not, rescale
    d_scale_y = depth_map.shape[0] / curr_cam.height
    d_scale_x = depth_map.shape[1] / curr_cam.width

    depth_val = depth_map[int(v * d_scale_y), int(u * d_scale_x)]

    print(f"Start Frame: {curr_img_data.name}, Point: {start_point}, Depth: {depth_val:.4f}")

    # utilize SAM to segment the first frame
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=np.array([start_point]),
        point_labels=np.array([1]), # 1 denotes foreground
        multimask_output=True
    )
    current_mask = masks[2]

    # save the result of first frame 
    base_name = os.path.splitext(curr_img_data.name)[0]
    ext = os.path.splitext(curr_img_data.name)[1]
    
    # 1. original file - only mask overlay
    h, w = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    masked_overlay = np.ma.masked_where(~current_mask, current_mask.astype(float))
    ax.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    clean_path = os.path.join(result_path, curr_img_data.name)
    plt.savefig(clean_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    
    # 2. debug file with notations
    fig_debug = plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    plt.scatter([u], [v], color='lime', marker='o', s=150, 
                label='Start Point', edgecolors='white', linewidths=2.5)
    plt.title(f"Frame {start_idx}: {curr_img_data.name} (Single Point)", fontsize=14)
    plt.xlabel(f"Resolution: {w}x{h}", fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    debug_path = os.path.join(result_path, f"{base_name}_debug{ext}")
    plt.savefig(debug_path, bbox_inches='tight', dpi=150)
    plt.close(fig_debug)
    
    # 3. npy mask file
    mask_path = os.path.join(result_path, f"{base_name}.npy")
    np.save(mask_path, current_mask)

    # 4. propagate to following frames
    prev_img_data = curr_img_data
    prev_point = start_point
    prev_depth = depth_val

    K = get_intrinsic_matrix(curr_cam)

    for i in range(start_idx + 1, len(sorted_img_ids)):
        next_img_id = sorted_img_ids[i]
        next_img_data = images[next_img_id]

        print(f"Propagating to {next_img_data.name}...")

        # core: 3D projection

        # A. unprojection: restore 3D point from last frame
        R_prev = qvec2rotmat(prev_img_data.qvec)
        t_prev = prev_img_data.tvec
        P_world = unproject_pixel(prev_point[0], prev_point[1], prev_depth, K, R_prev, t_prev)

        # B. reprojection: project 3D point into current frame
        R_next = qvec2rotmat(next_img_data.qvec)
        t_next = next_img_data.tvec
        projected_uv = reproject_point(P_world, K, R_next, t_next)

        if projected_uv is None:
            print("Point projected behind camera, stopping tracking.")
            break

        # check if out of range
        if not (0 <= projected_uv[0] < curr_cam.width and 0 <= projected_uv[1] < curr_cam.height):
            print("Point projected outside image, stopping tracking.")
            break

        print(f" Projected point: {projected_uv}")

        # SAM

        # read current frame
        next_img_path = os.path.join(images_dir, next_img_data.name)
        next_image = cv2.imread(next_img_path)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

        # Read depth map (支持COLMAP和gsplat两种格式)
        if use_colmap_depth:
            depth_filename = next_img_data.name + ".geometric.bin"
            next_depth_path = os.path.join(depths_dir, depth_filename)
            if not os.path.exists(next_depth_path):
                depth_filename = next_img_data.name + ".photometric.bin"
                next_depth_path = os.path.join(depths_dir, depth_filename)
            next_depth_map = read_colmap_depth(next_depth_path)
        else:
            next_depth_path = os.path.join(depths_dir, os.path.splitext(next_img_data.name)[0] + ".npy")
            next_depth_map = np.load(next_depth_path)

        u_new, v_new = projected_uv
        d_val_new = next_depth_map[int(v_new * d_scale_y), int(u_new * d_scale_x)]

        # run SAM
        predictor.set_image(next_image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([projected_uv]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        current_mask = masks[2]
        base_name = os.path.splitext(next_img_data.name)[0]
        ext = os.path.splitext(next_img_data.name)[1]
        
        # 1. original file - green screen replacement (chroma key green)
        green_screen_img = next_image.copy()
        # Use chroma key green color (0, 177, 64) in RGB
        green_screen_img[current_mask] = [0, 177, 64]
        cv2.imwrite(os.path.join(result_path, next_img_data.name), cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))
        
        # 2. debug file with notations
        fig_debug = plt.figure(figsize=(12, 8))
        plt.imshow(next_image)
        plt.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        plt.scatter([u_new], [v_new], color='lime', marker='o', s=150,
                    label='Projected Point', edgecolors='white', linewidths=2.5)
        plt.title(f"Frame {i}: {next_img_data.name} (Single Point)", fontsize=14)
        plt.xlabel(f"Resolution: {w}x{h}", fontsize=10)
        plt.legend(loc='upper right', fontsize=10)
        debug_path = os.path.join(result_path, f"{base_name}_debug{ext}")
        plt.savefig(debug_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)
        
        # 3. npy mask file
        mask_path = os.path.join(result_path, f"{base_name}.npy")
        np.save(mask_path, current_mask)

        # update state for follow propagation
        prev_img_data = next_img_data
        prev_point = projected_uv
        prev_depth = d_val_new


"""
Args:
    start_point: Single point (x, y) for backward compatibility
    positive_points: List of (x, y) foreground points (overrides start_point if provided)
    negative_points: List of (x, y) background/exclusion points
    use_centroid: If True, use mask centroid as anchor instead of clicked point
    update_anchor: If True, update 3D anchor each frame based on new mask centroid
"""
def run_propagation_anchor(
    colmap_path,
    result_path,
    images_dir,
    depths_dir,
    start_img_name,
    start_point,
    sam_checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth",
    sam_config="vit_h",
    use_centroid=True,
    update_anchor=True,
    positive_points=None,
    negative_points=None,
    use_colmap_depth=True
):
    """
    Propagate segmentation using anchored 3D point.
    """
    if not os.path.exists(result_path):
        print(f"Creating result directory: {result_path}")
        os.makedirs(result_path, exist_ok=True)

    # 1. load COLMAP data
    print("Loading COLMAP model...")
    cameras, images, _ = read_model(colmap_path) 
    # sort image based on filenames
    sorted_img_ids = sorted(images.keys(), key = lambda k: images[k].name) 

    # find the initial index of images
    start_idx = -1
    for i, img_id in enumerate(sorted_img_ids):
        if images[img_id].name == start_img_name:
            start_idx = i
            break
    
    if start_idx == -1:
        raise ValueError(f"Start image {start_img_name} not found in COLMAP model")
    
    # 2. initialize SAM
    print("Loading SAM model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    sam = sam_model_registry[sam_config](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 3. preprocess initial frame
    curr_img_id = sorted_img_ids[start_idx]
    curr_img_data = images[curr_img_id]
    curr_cam = cameras[curr_img_data.camera_id]

    # read images and depth maps
    img_path = os.path.join(images_dir, curr_img_data.name)

    if use_colmap_depth:
        depth_filename = curr_img_data.name + ".geometric.bin"
        depth_path = os.path.join(depths_dir, depth_filename)

        if not os.path.exists(depth_path):
            depth_filename = curr_img_data.name + ".photometric.bin"
            depth_path = os.path.join(depths_dir, depth_filename)

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"COLMAP depth not found: {depth_path}")
        
        depth_map = read_colmap_depth(depth_path)
        print(f"Loaded COLMAP depth map: {depth_path} (shape: {depth_map.shape})")
    else:
        depth_path = os.path.join(depths_dir, os.path.splitext(curr_img_data.name)[0] + ".npy")
        depth_map = np.load(depth_path)
        print(f"Loaded gsplat depth map: {depth_path} (shape: {depth_map.shape})")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get depth of picked pixel
    u, v = start_point
    # NOTE: the depth map might be scaled, we suppose that the size of depth map keeps same with the size of original image
    # if not, rescale
    # (curr_cam.width, curr_cam.height) is the original resolution, while image.shape is the current image's resolution
    scale_factor_x = curr_cam.width / image.shape[1]
    scale_factor_y = curr_cam.height / image.shape[0]

    d_scale_y = depth_map.shape[0] / curr_cam.height
    d_scale_x = depth_map.shape[1] / curr_cam.width

    depth_val = depth_map[int(v * d_scale_y), int(u * d_scale_x)]

    print(f"Start Frame: {curr_img_data.name}, Point: {start_point}, Depth: {depth_val:.4f}")

    # Prepare points for SAM
    if positive_points is None:
        # Use single start_point for backward compatibility
        point_coords = np.array([start_point])
        point_labels = np.array([1])  # 1 = foreground
    else:
        # Use multiple points
        all_points = positive_points + (negative_points if negative_points else [])
        all_labels = [1] * len(positive_points) + ([0] * len(negative_points) if negative_points else [])
        point_coords = np.array(all_points)
        point_labels = np.array(all_labels)
        print(f"Using {len(positive_points)} foreground points and {len(negative_points) if negative_points else 0} background points")

    # utilize SAM to segment the first frame
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    current_mask = masks[2]

    # === IMPROVEMENT: Compute mask centroid as anchor point ===
    # This ensures the anchor is at the object center, not at the clicked edge
    if use_centroid:
        mask_coords = np.argwhere(current_mask)  # Returns [row, col] = [y, x]
        if len(mask_coords) > 0:
            centroid_y = np.mean(mask_coords[:, 0])
            centroid_x = np.mean(mask_coords[:, 1])
            centroid_point = (int(centroid_x), int(centroid_y))
            
            print(f"Original click point: {start_point}")
            print(f"Mask centroid: {centroid_point}")
            print(f"Offset from click: ({centroid_x - u:.1f}, {centroid_y - v:.1f}) pixels")
            
            # Use centroid as the anchor point for tracking
            u, v = centroid_point
        else:
            print("Warning: Empty mask, using original click point")
    else:
        print("Using original click point as anchor")

    # save the result of first frame
    base_name = os.path.splitext(curr_img_data.name)[0]
    ext = os.path.splitext(curr_img_data.name)[1]
    
    # 1. original file - only mask overlay
    h, w = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    masked_overlay = np.ma.masked_where(~current_mask, current_mask.astype(float))
    ax.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    clean_path = os.path.join(result_path, curr_img_data.name)
    plt.savefig(clean_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    
    # 2. debug file with notations
    fig_debug = plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    
    # Visualize all input points
    if positive_points is not None:
        pos_pts = np.array(positive_points)
        plt.scatter(pos_pts[:, 0], pos_pts[:, 1], color='lime', marker='o', s=150, 
                   label=f'Foreground ({len(positive_points)})', edgecolors='white', linewidths=2.5)
        if negative_points and len(negative_points) > 0:
            neg_pts = np.array(negative_points)
            plt.scatter(neg_pts[:, 0], neg_pts[:, 1], color='red', marker='x', s=150, 
                       label=f'Background ({len(negative_points)})', linewidths=3)
    else:
        plt.scatter([start_point[0]], [start_point[1]], color='yellow', marker='o', s=150, 
                    label='Click Point', edgecolors='black', linewidths=2.5)
    
    # Show the anchor point (centroid)
    if use_centroid:
        plt.scatter([u], [v], color='cyan', marker='*', s=250, 
                    label='Anchor (Centroid)', edgecolors='black', linewidths=2)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.title(f"Frame {start_idx}: {curr_img_data.name} (Multi-Point Anchor)", fontsize=14)
    plt.xlabel(f"Resolution: {w}x{h}", fontsize=10)
    debug_path = os.path.join(result_path, f"{base_name}_debug{ext}")
    plt.savefig(debug_path, bbox_inches='tight', dpi=150)
    plt.close(fig_debug)
    
    # 3. npy mask file
    mask_path = os.path.join(result_path, f"{base_name}.npy")
    np.save(mask_path, current_mask)

    # 4. propagate to following frames
    # Prepare 3D anchor points for ALL selected points
    K_full = get_intrinsic_matrix(curr_cam)
    R_start = qvec2rotmat(curr_img_data.qvec)
    t_start = curr_img_data.tvec
    
    # Store 3D anchor points for all positive and negative points
    anchor_3d_points_positive = []
    anchor_3d_points_negative = []
    
    if positive_points is not None:
        # Convert all positive points to 3D
        for pt in positive_points:
            u_pt, v_pt = pt
            u_pt_full = u_pt * scale_factor_x
            v_pt_full = v_pt * scale_factor_y
            depth_pt = depth_map[int(v_pt * d_scale_y), int(u_pt * d_scale_x)]
            P_world_pt = unproject_pixel(u_pt_full, v_pt_full, depth_pt, K_full, R_start, t_start)
            anchor_3d_points_positive.append(P_world_pt)
            print(f"  Positive point {pt} -> 3D: {P_world_pt}")
        
        # Convert all negative points to 3D (if any)
        if negative_points:
            for pt in negative_points:
                u_pt, v_pt = pt
                u_pt_full = u_pt * scale_factor_x
                v_pt_full = v_pt * scale_factor_y
                depth_pt = depth_map[int(v_pt * d_scale_y), int(u_pt * d_scale_x)]
                P_world_pt = unproject_pixel(u_pt_full, v_pt_full, depth_pt, K_full, R_start, t_start)
                anchor_3d_points_negative.append(P_world_pt)
                print(f"  Negative point {pt} -> 3D: {P_world_pt}")
    else:
        # Single point mode: use centroid
        u_full = u * scale_factor_x
        v_full = v * scale_factor_y
        depth_val_centroid = depth_map[int(v * d_scale_y), int(u * d_scale_x)]
        P_world_anchor = unproject_pixel(u_full, v_full, depth_val_centroid, K_full, R_start, t_start)
        anchor_3d_points_positive.append(P_world_anchor)
        print(f"Anchor 3D Point (at object center): {P_world_anchor}")

    print(f"\nTotal 3D anchors: {len(anchor_3d_points_positive)} positive, {len(anchor_3d_points_negative)} negative")

    for i in range(start_idx + 1, len(sorted_img_ids)):
        next_img_id = sorted_img_ids[i]
        next_img_data = images[next_img_id]

        print(f"Propagating to {next_img_data.name}...")

        # Project ALL 3D anchor points to current frame
        R_next = qvec2rotmat(next_img_data.qvec)
        t_next = next_img_data.tvec
        
        # Reproject all positive points
        projected_positive_pts = []
        for P_world_pt in anchor_3d_points_positive:
            projected_uv_full = reproject_point(P_world_pt, K_full, R_next, t_next)
            if projected_uv_full is not None:
                u_proj = projected_uv_full[0] / scale_factor_x
                v_proj = projected_uv_full[1] / scale_factor_y
                # Check bounds
                if 0 <= u_proj < image.shape[1] and 0 <= v_proj < image.shape[0]:
                    projected_positive_pts.append((u_proj, v_proj))
        
        # Reproject all negative points
        projected_negative_pts = []
        for P_world_pt in anchor_3d_points_negative:
            projected_uv_full = reproject_point(P_world_pt, K_full, R_next, t_next)
            if projected_uv_full is not None:
                u_proj = projected_uv_full[0] / scale_factor_x
                v_proj = projected_uv_full[1] / scale_factor_y
                # Check bounds
                if 0 <= u_proj < image.shape[1] and 0 <= v_proj < image.shape[0]:
                    projected_negative_pts.append((u_proj, v_proj))
        
        # Check if we have at least one positive point
        if len(projected_positive_pts) == 0:
            print("  No valid positive points projected, skipping frame.")
            continue
        
        print(f"  Projected {len(projected_positive_pts)} positive, {len(projected_negative_pts)} negative points")

        # SAM

        # Read current frame
        next_img_path = os.path.join(images_dir, next_img_data.name)
        next_image = cv2.imread(next_img_path)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

        # Prepare points for SAM: combine projected positive and negative points
        all_projected_pts = projected_positive_pts + projected_negative_pts
        all_projected_labels = [1] * len(projected_positive_pts) + [0] * len(projected_negative_pts)
        
        # Run SAM with ALL projected points
        predictor.set_image(next_image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(all_projected_pts),
            point_labels=np.array(all_projected_labels),
            multimask_output=True
        )
        
        best_mask = masks[2]
        
        # === IMPROVEMENT: Update 3D anchors using new mask centroid ===
        # This re-centers the tracking points to handle object rotation/perspective changes
        centroid_2d = None  # Initialize
        mask_coords_new = []  # Initialize
        
        if update_anchor:
            mask_coords_new = np.argwhere(best_mask)
            if len(mask_coords_new) > 0:
                new_centroid_y = np.mean(mask_coords_new[:, 0])
                new_centroid_x = np.mean(mask_coords_new[:, 1])
                centroid_2d = (int(new_centroid_x), int(new_centroid_y))
                
                # Calculate average shift from all projected positive points
                if len(projected_positive_pts) > 0:
                    avg_proj_x = np.mean([pt[0] for pt in projected_positive_pts])
                    avg_proj_y = np.mean([pt[1] for pt in projected_positive_pts])
                    shift_x = new_centroid_x - avg_proj_x
                    shift_y = new_centroid_y - avg_proj_y
                    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
                    
                    print(f"  Mask centroid shift: ({shift_x:.1f}, {shift_y:.1f}) px, magnitude: {shift_magnitude:.1f}")
                    
                    # Only update anchors if shift is reasonable
                    if shift_magnitude < 50:
                        # Update ALL 3D anchor points based on centroid shift
                        refined_u_full = new_centroid_x * scale_factor_x
                        refined_v_full = new_centroid_y * scale_factor_y
                        
                        # Get depth at refined position (支持COLMAP和gsplat两种格式)
                        if use_colmap_depth:
                            depth_filename = next_img_data.name + ".geometric.bin"
                            next_depth_path = os.path.join(depths_dir, depth_filename)
                            if not os.path.exists(next_depth_path):
                                depth_filename = next_img_data.name + ".photometric.bin"
                                next_depth_path = os.path.join(depths_dir, depth_filename)
                            next_depth_map = read_colmap_depth(next_depth_path)
                        else:
                            next_depth_path = os.path.join(depths_dir, os.path.splitext(next_img_data.name)[0] + ".npy")
                            next_depth_map = np.load(next_depth_path)
                        
                        refined_depth = next_depth_map[
                            int(new_centroid_y * d_scale_y), 
                            int(new_centroid_x * d_scale_x)
                        ]
                        
                        # Update the primary anchor point
                        P_world_new = unproject_pixel(
                            refined_u_full, refined_v_full, refined_depth,
                            K_full, R_next, t_next
                        )
                        
                        # Replace all anchor points with the new centroid-based anchor
                        # This ensures consistency as the object rotates
                        anchor_3d_points_positive = [P_world_new]
                        anchor_3d_points_negative = []  # Clear negative points after first update
                        
                        print(f"  Updated to single centroid anchor: {P_world_new}")
                    else:
                        print(f"  Shift too large ({shift_magnitude:.1f} px), keeping previous anchors")
            else:
                print("  Warning: Empty mask in current frame")

        # Visualization 
        base_name = os.path.splitext(next_img_data.name)[0]
        ext = os.path.splitext(next_img_data.name)[1]
        
        # 1. original file - green screen replacement (chroma key green)
        green_screen_img = next_image.copy()
        # Use chroma key green color (0, 177, 64) in RGB
        green_screen_img[best_mask] = [0, 177, 64]
        cv2.imwrite(os.path.join(result_path, next_img_data.name), cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))
        
        # 2. debug file with notations
        fig_debug = plt.figure(figsize=(12, 8))
        plt.imshow(next_image)
        plt.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        
        # Show all projected positive points
        if len(projected_positive_pts) > 0:
            pos_pts_arr = np.array(projected_positive_pts)
            plt.scatter(pos_pts_arr[:, 0], pos_pts_arr[:, 1], color='lime', marker='o', s=150, 
                       label=f'Projected Positive ({len(projected_positive_pts)})', 
                       edgecolors='white', linewidths=2.5)
        
        # Show all projected negative points
        if len(projected_negative_pts) > 0:
            neg_pts_arr = np.array(projected_negative_pts)
            plt.scatter(neg_pts_arr[:, 0], neg_pts_arr[:, 1], color='red', marker='x', s=150, 
                       label=f'Projected Negative ({len(projected_negative_pts)})', linewidths=3)
        
        # Show centroid if updated
        if update_anchor and centroid_2d is not None:
            plt.scatter([centroid_2d[0]], [centroid_2d[1]], color='cyan', marker='*', s=250,
                       label='Centroid', edgecolors='black', linewidths=2)
        
        plt.legend(loc='upper right', fontsize=10)
        plt.title(f"Frame {i}: {next_img_data.name} (Multi-Point Anchor)", fontsize=14)
        plt.xlabel(f"Resolution: {w}x{h}", fontsize=10)
        debug_path = os.path.join(result_path, f"{base_name}_debug{ext}")
        plt.savefig(debug_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)
        
        # 3. npy mask file
        mask_path = os.path.join(result_path, f"{base_name}.npy")
        np.save(mask_path, best_mask)

############################################# OPTICAL FLOW ################################################

def compute_optical_flow(gray1, gray2):
    """
    Compute the dense optical flow between two frames.
    Using improved parameters for better accuracy.
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,    # image pyramid scale (0.5 means each layer is half the size)
        levels=5,         # pyramid levels (increased from 3 for better large motion handling)
        winsize=21,       # window size (increased from 15 for smoother flow)
        iterations=5,     # iterations in each level (increased from 3 for better convergence)
        poly_n=7,         # size of pixel neighborhood (increased from 5)
        poly_sigma=1.5,   # Gaussian standard variance (increased from 1.2)
        flags=0
    ) # type: ignore
    return flow

def run_propagation_with_flow(colmap_path, 
                              result_path, 
                              images_dir, 
                              depths_dir, 
                              start_img_name, 
                              start_point, 
                              sam_checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth",
                              method="hybrid",
                              use_colmap_depth=True):
    """
    combine with geometry projection and optical flow
    
    Args:
        method: "hybrid" (default) | "geometry_only" | "flow_only"
    """
    if not os.path.exists(result_path):
        print(f"Creating result directory: {result_path}")
        os.makedirs(result_path, exist_ok=True)

    # 1. load COLMAP data
    print("Loading COLMAP model...")
    cameras, images, _ = read_model(colmap_path)
    sorted_img_ids = sorted(images.keys(), key = lambda k : images[k].name)

    start_idx = -1
    for i, img_id in enumerate(sorted_img_ids):
        if images[img_id].name == start_img_name:
            start_idx = i
            break

    if start_idx == -1:
        raise ValueError(f"Start time {start_img_name} not found in COLMAP model")
    
    # 2. initialize SAM
    print("Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 3. preprocess initial frame
    curr_img_id = sorted_img_ids[start_idx]
    curr_img_data = images[curr_img_id]
    curr_cam = cameras[curr_img_data.camera_id]

    img_path = os.path.join(images_dir, curr_img_data.name)
    
    # Read depth map (支持COLMAP和gsplat两种格式)
    if use_colmap_depth:
        depth_filename = os.path.splitext(curr_img_data.name)[0] + ".geometric.bin"
        depth_path = os.path.join(depths_dir, depth_filename)
        if not os.path.exists(depth_path):
            depth_filename = os.path.splitext(curr_img_data.name)[0] + ".photometric.bin"
            depth_path = os.path.join(depths_dir, depth_filename)
        depth_map = read_colmap_depth(depth_path)
        print(f"Loaded COLMAP depth: {depth_path}")
    else:
        depth_path = os.path.join(depths_dir, os.path.splitext(curr_img_data.name)[0] + ".npy")
        depth_map = np.load(depth_path)
        print(f"Loaded gsplat depth: {depth_path}")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    u, v = start_point
    
    scale_factor_x = curr_cam.width / image.shape[1]
    scale_factor_y = curr_cam.height / image.shape[0]
    d_scale_y = depth_map.shape[0] / curr_cam.height
    d_scale_x = depth_map.shape[1] / curr_cam.width
    depth_val = depth_map[int(v * d_scale_y), int(u * d_scale_x)]

    print(f"Start frame: {curr_img_data.name}, Point: {start_point}, Depth: {depth_val:.4f}")

    ## SAM for first frame
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=np.array([start_point]), point_labels=np.array([1]), multimask_output=True)
    current_mask = masks[2]

    ## Save 
    base_name = os.path.splitext(curr_img_data.name)[0]
    ext = os.path.splitext(curr_img_data.name)[1]
    
    # 1. original file - only mask overlay
    h, w = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    masked_overlay = np.ma.masked_where(~current_mask, current_mask.astype(float))
    ax.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    clean_path = os.path.join(result_path, curr_img_data.name)
    plt.savefig(clean_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    
    # 2. debug file with notations
    fig_debug = plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    plt.scatter([u], [v], color='lime', marker='o', s=150, 
                label='Start Point', edgecolors='white', linewidths=2.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.title(f"Frame {start_idx}: {curr_img_data.name} (Hybrid: Geo+Flow)", fontsize=14)
    plt.xlabel(f"Resolution: {w}x{h} | Method: {method}", fontsize=10)
    debug_path = os.path.join(result_path, f"{base_name}_debug{ext}")
    plt.savefig(debug_path, bbox_inches='tight', dpi=150)
    plt.close(fig_debug)
    
    # 3. npy mask file
    mask_path = os.path.join(result_path, f"{base_name}.npy")
    np.save(mask_path, current_mask)

    u_full = u * scale_factor_x
    v_full = v * scale_factor_y
    K_full = get_intrinsic_matrix(curr_cam)
    R_start = qvec2rotmat(curr_img_data.qvec)
    t_start = curr_img_data.tvec
    P_world_anchor = unproject_pixel(u_full, v_full, depth_val, K_full, R_start, t_start)
    print(f"Anchor 3D point: {P_world_anchor}")

    # variables for optical flow
    prev_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    prev_point_flow = np.array(start_point, dtype=np.float32)

    for i in range(start_idx + 1, len(sorted_img_ids)):
        next_img_id = sorted_img_ids[i]
        next_img_data = images[next_img_id]
        print(f"Propagating to {next_img_data.name}...")

        # Read next frame
        next_img_path = os.path.join(images_dir, next_img_data.name)
        next_image = cv2.imread(next_img_path)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
        next_gray = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)

        # === Method 1: Geometry-based (Anchored) ===
        R_next = qvec2rotmat(next_img_data.qvec)
        t_next = next_img_data.tvec
        projected_uv_full = reproject_point(P_world_anchor, K_full, R_next, t_next)

        if projected_uv_full is not None:
            u_geo = projected_uv_full[0] / scale_factor_x
            v_geo = projected_uv_full[1] / scale_factor_y
            projected_uv_geo = np.array([u_geo, v_geo])
        else:
            projected_uv_geo = None


        # === Method 2: Optical flow ===
        flow = compute_optical_flow(prev_gray, next_gray)

        # get move vectors from optical flow
        u_prev_int = int(np.clip(prev_point_flow[0], 0, flow.shape[1] - 1))
        v_prev_int = int(np.clip(prev_point_flow[1], 0, flow.shape[0] - 1))

        flow_vec = flow[v_prev_int, u_prev_int]
        u_flow = prev_point_flow[0] + flow_vec[0]
        v_flow = prev_point_flow[1] + flow_vec[1]
        projected_uv_flow = np.array([u_flow, v_flow])

        # === Fusion ===
        distance = -1  # Initialize distance variable
        if projected_uv_geo is not None:
            # compute the distance between this two method
            distance = np.linalg.norm(projected_uv_geo - projected_uv_flow)
            print(f" Geo: {projected_uv_geo}, Flow: {projected_uv_flow}, Distance: {distance:.2f}")

            # check blocking: compare the projection depth and actual depth
            # 1. compute the depth of anchor in current frame
            P_cam_expected = np.dot(R_next, P_world_anchor) + t_next
            expected_depth = P_cam_expected[2]

            # 2. read actual depth (支持COLMAP和gsplat两种格式)
            if use_colmap_depth:
                depth_filename = next_img_data.name + ".geometric.bin"
                next_depth_path = os.path.join(depths_dir, depth_filename)
                if not os.path.exists(next_depth_path):
                    depth_filename = next_img_data.name + ".photometric.bin"
                    next_depth_path = os.path.join(depths_dir, depth_filename)
                next_depth_map = read_colmap_depth(next_depth_path)
            else:
                next_depth_path = os.path.join(depths_dir, os.path.splitext(next_img_data.name)[0] + ".npy")
                next_depth_map = np.load(next_depth_path)

            # read depth by using reprojection (make sure it is in the image)
            u_geo_int = int(np.clip(projected_uv_geo[0], 0, next_depth_map.shape[1] - 1))
            v_geo_int = int(np.clip(projected_uv_geo[1], 0, next_depth_map.shape[0] - 1))

            # consider the scaling 
            u_depth = int(u_geo_int * d_scale_x)
            v_depth = int(v_geo_int * d_scale_y)
            u_depth = np.clip(u_depth, 0, next_depth_map.shape[1] - 1)
            v_depth = np.clip(v_depth, 0, next_depth_map.shape[0] - 1)

            actual_depth = next_depth_map[v_depth, u_depth]

            # blocking checking: if the actual depth is less than expected depth, there is a block
            depth_diff = expected_depth - actual_depth
            is_occluded = depth_diff > 0.5 # adjustable

            print(f"Expected depth: {expected_depth:.2f}, Actual depth: {actual_depth:.2f}, Diff: {depth_diff:.2f}")

            if is_occluded:
                print(f"OCCULUSION DETECTED! Skipping this frame.")
                prev_gray = next_gray
                prev_point_flow = projected_uv_flow
                continue

            # Method selection based on parameter
            if method == "geometry_only":
                final_point = projected_uv_geo
                print(f"Geometry only mode -> Using geometry-based point: {final_point}")
            elif method == "flow_only":
                final_point = projected_uv_flow
                print(f"Flow only mode -> Using flow-based point: {final_point}")
            else:  # method == "hybrid"
                if distance < 5:
                    # High confidence: both methods agree, use weighted average
                    final_point = 0.7 * projected_uv_geo + 0.3 * projected_uv_flow
                    print(f"High confidence -> Using weighted average: {final_point}")
                elif distance < 20:
                    # Medium discrepancy: prefer geometry but check depth validity
                    final_point = projected_uv_geo
                    print(f"Medium discrepancy -> Using geometry-based point.")
                else:
                    # Large discrepancy: need careful decision
                    if actual_depth > 0 and abs(depth_diff) < 2.0:
                        # Depth valid, likely flow drift
                        final_point = projected_uv_geo
                        print("Large discrepancy but depth valid -> Using geometry.")
                    else:
                        # Depth invalid, geometry might be wrong, try flow
                        final_point = projected_uv_flow
                        print("Large discrepancy and depth invalid -> Using flow (geometry may be unreliable).")
                        # Alternative: skip frame if too uncertain
                        # prev_gray = next_gray
                        # prev_point_flow = projected_uv_flow
                        # continue
        
        else:
            # Geometry failed (behind camera)
            if method == "geometry_only":
                print(f"Geometry only mode but point behind camera -> Skipping frame.")
                prev_gray = next_gray
                prev_point_flow = projected_uv_flow
                continue
            else:  # flow_only or hybrid
                print(f"Geometry failed (behind camera) -> Using flow-based point: {projected_uv_flow}")
                final_point = projected_uv_flow

        # Check bounds
        if not (0 <= final_point[0] < image.shape[1] and 0 <= final_point[1] < image.shape[0]):
            print(f"Point out of bounds: {final_point}")
            prev_gray = next_gray
            prev_point_flow = projected_uv_flow
            continue


        # SAM segmentation
        predictor.set_image(next_image)
        masks, scores, _ = predictor.predict(point_coords=np.array([final_point]), point_labels=np.array([1]), multimask_output=True)
        
        current_mask = masks[2]
        base_name = os.path.splitext(next_img_data.name)[0]
        ext = os.path.splitext(next_img_data.name)[1]

        # Save result 
        # 1. original file - green screen replacement (chroma key green)
        green_screen_img = next_image.copy()
        # Use chroma key green color (0, 177, 64) in RGB
        green_screen_img[current_mask] = [0, 177, 64]
        cv2.imwrite(os.path.join(result_path, next_img_data.name), cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))
        
        # 2. debug file with notations
        fig_debug = plt.figure(figsize=(12, 8))
        plt.imshow(next_image)
        plt.imshow(masked_overlay, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        plt.scatter([final_point[0]], [final_point[1]], color='lime', marker='o', s=150, 
                    label='Final Point', edgecolors='white', linewidths=2.5)
        if projected_uv_geo is not None:
            plt.scatter([projected_uv_geo[0]], [projected_uv_geo[1]], color='cyan', marker='s', s=100, 
                        label='Geometry', edgecolors='white', linewidths=2)
        plt.scatter([projected_uv_flow[0]], [projected_uv_flow[1]], color='yellow', marker='^', s=100, 
                    label='Flow', edgecolors='white', linewidths=2)
        if distance >= 0:
            plt.text(10, 30, f'Geo-Flow Distance: {distance:.1f}px', 
                    color='white', fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        plt.legend(loc='upper right', fontsize=10)
        plt.title(f"Frame {i}: {next_img_data.name} (Hybrid: Geo+Flow)", fontsize=14)
        plt.xlabel(f"Resolution: {w}x{h} | Method: {method}", fontsize=10)
        debug_path = os.path.join(result_path, f"{base_name}_debug{ext}")
        plt.savefig(debug_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)
        
        # 3. npy mask file
        mask_path = os.path.join(result_path, f"{base_name}.npy")
        np.save(mask_path, current_mask)

        # Update for next iteration
        prev_gray = next_gray
        
        # IMPORTANT: Use geometry point to reset flow tracking when available
        # This prevents cumulative drift in optical flow
        if projected_uv_geo is not None and distance < 10:
            # If geometry is reliable, use it to reset flow tracker
            prev_point_flow = projected_uv_geo
            print(f"  -> Resetting flow tracker to geometry point to prevent drift")
        else:
            # Otherwise use final point (may accumulate error)
            prev_point_flow = final_point 

################################################# SAM2 #################################################### 

"""
Args:
    images_dir: Directory containing video frames
    result_dir: Directory to save segmentation results
    sam2_checkpoint: Path to SAM2 checkpoint
    model_cfg: SAM2 config name
    positive_points: List of (x, y) foreground points
    negative_points: List of (x, y) foreground points
"""
def propagation_with_sam2(images_dir,
                          result_dir,
                          sam2_checkpoint,
                          model_cfg,
                          positive_points=None,
                          negative_points=None):
    """ SAM2 video object segmentation with multi-point selection """

    # Create result directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
        print(f"Created result directory: {result_dir}")

    # Validate images directory
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Load frames
    frame_names = [
        p for p in os.listdir(images_dir) 
        if os.path.splitext(p)[-1] in [".jpg", "jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    if len(frame_names) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    
    frame_names.sort(key=lambda p : int(os.path.splitext(p)[0]))

    # Initialize SAM2 video predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=images_dir)

    # Prepare points for first frame
    frame_idx = 0

    # Prepare points for SAM
    all_points = positive_points + (negative_points if negative_points else []) # type: ignore
    all_labels = [1] * len(positive_points) + ([0] * len(negative_points) if negative_points else []) # type: ignore
    print(f"Using {len(positive_points)} foreground points and {len(negative_points) if negative_points else 0} background points") # type: ignore

    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=1,
        points=np.array(all_points),
        labels=np.array(all_labels),
    )

    frames_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frames_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for frame_idx in tqdm(sorted(frames_segments.keys()), desc="Saving frames"):
        frame_name = frame_names[frame_idx]
        base_name = os.path.splitext(frame_name)[0]
        ext = os.path.splitext(frame_name)[1]

        # Load original image (RGB default)
        img_path = os.path.join(images_dir, frame_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get mask for object ID 1
        mask = frames_segments[frame_idx][1][0]  # [0] to get first mask dimension
        
        # 1. original file - green screen replacement (chroma key green)
        green_screen_img = image.copy()
        # Use chroma key green color (0, 177, 64) in RGB
        green_screen_img[mask] = [0, 177, 64]
        cv2.imwrite(os.path.join(result_dir, frame_name), cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))

        # 2. debug file with notations
        h, w = image.shape[:2]
        fig_debug = plt.figure(figsize=(12, 8))
        plt.imshow(image)
        masked_overlay_debug = np.ma.masked_where(~mask, mask.astype(float))
        plt.imshow(masked_overlay_debug, alpha=0.5, cmap='jet')

        if frame_idx == 0:
            pos_pts = np.array(positive_points)
            plt.scatter(pos_pts[:, 0], pos_pts[:, 1], color='lime', marker='o',
                        s=150, label=f'Foreground ({len(positive_points)})',
                        edgecolors='white', linewidths=2.5)

            if negative_points and len(negative_points) > 0:
                neg_pts = np.array(negative_points)
                plt.scatter(neg_pts[:, 0], neg_pts[:, 1], color='red', marker='x',
                            s=150, label=f'Background ({len(negative_points)})', 
                            linewidths=3)
            plt.legend(loc='upper right', fontsize=10)

        plt.title(f"Frame {frame_idx}/{len(frames_segments)-1}: {frame_name} (SAM2)", fontsize=14)
        plt.xlabel(f"Resolution: {w}x{h}", fontsize=10)

        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        cbar=plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
        cbar.set_label('Mask Confidence', rotation=270, labelpad=15)

        debug_path = os.path.join(result_dir, f"{base_name}_debug{ext}")
        plt.savefig(debug_path, bbox_inches='tight', dpi=150)
        plt.close(fig_debug)
        
        # .npy file
        mask_save_path = os.path.join(result_dir, f"{base_name}.npy")
        np.save(mask_save_path, mask)

    return
    


if __name__ == "__main__":
    COLMAP_PATH = "data/sofa/sparse/0"

    RESULT_PATH = "data/sofa_Marked/images"
    ANCHOR_RESULT_PATH = "data/sofa_Marked_Anchor/images"
    FLOW_RESULT_PATH = "data/sofa_Marked_Flow/images"
    SAM2_RESULT_PATH = "data/STree/mask/Sam2/images_2"

    IMAGES_DIR = "data/STree/images_2"

    USE_COLMAP_DEPTH = True
    if USE_COLMAP_DEPTH:
        DEPTHS_DIR = "data/sofa/dense/stereo/depth_maps" # .bin
    else:
        DEPTHS_DIR = "results/sofa/train_depths" # .npy file folder

    START_IMAGE = "000001.jpg"
    
    # Interactive point selection
    USE_INTERACTIVE = True  # Set to False to use manual points
    START_POINT_MANUAL = (480, 270)  # Fallback manual point (for backward compatibility)
    POSITIVE_POINTS_MANUAL = None  # Or set to [(x1,y1), (x2,y2), ...] for multiple points
    NEGATIVE_POINTS_MANUAL = None  # Or set to [(x1,y1), (x2,y2), ...] for exclusion points
    
    if USE_INTERACTIVE and INTERACTIVE_AVAILABLE:
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE: Select points on the first image")
        print("=" * 60)
        
        # Construct path to first image
        first_image_path = os.path.join(IMAGES_DIR, START_IMAGE)
        
        if not os.path.exists(first_image_path):
            print(f"Error: First image not found: {first_image_path}")
            print("Please check IMAGES_DIR and START_IMAGE settings")
            exit(1)
        
        # Get points interactively (returns lists of positive and negative points)
        POSITIVE_POINTS, NEGATIVE_POINTS = get_start_point_interactive(first_image_path)
        
        if POSITIVE_POINTS is None:
            print("No points selected. Exiting.")
            exit(0)
            
        # For backward compatibility, set START_POINT to first positive point
        START_POINT = POSITIVE_POINTS[0] if POSITIVE_POINTS else START_POINT_MANUAL
    else:
        # Manual mode
        if POSITIVE_POINTS_MANUAL is not None:
            POSITIVE_POINTS = POSITIVE_POINTS_MANUAL
            NEGATIVE_POINTS = NEGATIVE_POINTS_MANUAL
            START_POINT = POSITIVE_POINTS[0]
            print(f"Using manual points: {len(POSITIVE_POINTS)} positive, {len(NEGATIVE_POINTS) if NEGATIVE_POINTS else 0} negative")
        else:
            # Fallback to single point
            START_POINT = START_POINT_MANUAL
            POSITIVE_POINTS = None
            NEGATIVE_POINTS = None
            print(f"Using manual START_POINT: {START_POINT}")


    # run_propagation_anchor(COLMAP_PATH,
    #                        ANCHOR_RESULT_PATH, 
    #                        IMAGES_DIR,
    #                        DEPTHS_DIR,
    #                        "000.jpg",
    #                        START_POINT,
    #                        "sam_checkpoints/sam_vit_l_0b3195.pth",
    #                        "vit_l",
    #                        False,   # use_centroid=True: use centroid instead of clicked point
    #                        False,   # update_anchor=True: update anchor every frame to avoid shifting
    #                        POSITIVE_POINTS,
    #                        NEGATIVE_POINTS,
    #                        USE_COLMAP_DEPTH)


    propagation_with_sam2(IMAGES_DIR, 
                          SAM2_RESULT_PATH,
                          "sam2_checkpoints/sam2_hiera_tiny.pt", 
                          "sam2_hiera_t.yaml", 
                          POSITIVE_POINTS,
                          NEGATIVE_POINTS)


