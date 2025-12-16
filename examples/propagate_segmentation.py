import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# pipeline
# 1.(Single pixel segmentation) user click -> get pixel coordinate(u1, v1) -> SAM -> Mask1
# 2.(Unprojection) pixel coordinate(u1, v1) + depth(d1) + camera paramaters(K, R1, t1) -> Hit point coordinates(P_world)
# 3.(Reprojection) P_world + camera parameters(K, R2, t2) -> pixel coordinate(u2, v2)
# 4.(Propogation segmentation) pixel coordinate(u2, v2) -> SAM -> Mask2

from BinaryReader import read_model, get_camera_center_and_rotation, qvec2rotmat

from segment_anything import sam_model_registry, SamPredictor

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
    sam_checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth"
):
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
    depth_path = os.path.join(depths_dir, os.path.splitext(curr_img_data.name)[0] + ".npy")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = np.load(depth_path)

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
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(current_mask, alpha=0.5)
    plt.scatter([u], [v], color='red', marker='*', s=100)
    plt.title(f"Frame {start_idx}: {curr_img_data.name}")
    save_path = os.path.join(result_path, curr_img_data.name)
    plt.savefig(save_path)
    plt.close()

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

        plt.figure(figsize=(10, 10))
        plt.imshow(next_image)
        plt.imshow(masks[2], alpha=0.5)
        plt.scatter([u_new], [v_new], color='red', marker='*', s=100)
        plt.title(f"Frame {i}: {next_img_data.name}")
        save_path = os.path.join(result_path, next_img_data.name)
        plt.savefig(save_path)
        plt.close()

        # update state for follow propagation
        prev_img_data = next_img_data
        prev_point = projected_uv
        prev_depth = d_val_new


def run_propagation_anchor(
    colmap_path,
    result_path,
    images_dir,
    depths_dir,
    start_img_name,
    start_point, 
    sam_checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth"
):
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
    depth_path = os.path.join(depths_dir, os.path.splitext(curr_img_data.name)[0] + ".npy")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = np.load(depth_path)

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

    # utilize SAM to segment the first frame
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=np.array([start_point]),
        point_labels=np.array([1]), # 1 denotes foreground
        multimask_output=True
    )
    current_mask = masks[2]

    # save the result of first frame
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(current_mask, alpha=0.5)
    plt.scatter([u], [v], color='red', marker='*', s=100)
    plt.title(f"Frame {start_idx}: {curr_img_data.name}")
    save_path = os.path.join(result_path, curr_img_data.name)
    plt.savefig(save_path)
    plt.close()

    # 4. propagate to following frames
    # a. map the click point to original resolution
    u_full = u * scale_factor_x
    v_full = v * scale_factor_y

    # b. get the intrinsic camera parameters of full resolution
    K_full = get_intrinsic_matrix(curr_cam)

    # c. unproject to fixed 3D point
    R_start = qvec2rotmat(curr_img_data.qvec)
    t_start = curr_img_data.tvec
    P_world_anchor = unproject_pixel(u_full, v_full, depth_val, K_full, R_start, t_start)

    print(f"Anchor 3D Point: {P_world_anchor}")

    for i in range(start_idx + 1, len(sorted_img_ids)):
        next_img_id = sorted_img_ids[i]
        next_img_data = images[next_img_id]

        print(f"Propagating to {next_img_data.name}...")

        # core: 3D projectio

        # reprojection: project 3D point into current frame
        R_next = qvec2rotmat(next_img_data.qvec)
        t_next = next_img_data.tvec
        projected_uv_full = reproject_point(P_world_anchor, K_full, R_next, t_next)

        if projected_uv_full is None:
            print("Point projected behind camera, stopping tracking.")
            continue

        u_new = projected_uv_full[0] / scale_factor_x
        v_new = projected_uv_full[1] / scale_factor_y
        
        projected_uv = np.array([u_new, v_new])

        # check if out of range
        if not (0 <= projected_uv[0] < image.shape[1] and 0 <= projected_uv[1] < image.shape[0]):
            print(f"Point projected outside image, {projected_uv}")
            continue

        print(f" Projected point: {projected_uv}")

        # SAM

        # read current frame
        next_img_path = os.path.join(images_dir, next_img_data.name)
        next_image = cv2.imread(next_img_path)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

        # run SAM
        predictor.set_image(next_image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([projected_uv]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(next_image)
        plt.imshow(masks[2], alpha=0.5)
        plt.scatter([u_new], [v_new], color='red', marker='*', s=100)
        plt.title(f"Frame {i}: {next_img_data.name}")
        save_path = os.path.join(result_path, next_img_data.name)
        plt.savefig(save_path)
        plt.close()



if __name__ == "__main__":
    COLMAP_PATH = "data/Tree/sparse/0"
    RESULT_PATH = "data/Tree_Marked/images"
    ANCHOR_RESULT_PATH = "data/Tree_Marked_Anchor/images"
    IMAGES_DIR = "data/Tree/images_2"
    DEPTHS_DIR = "results/Tree/train_depths" # .npy file folder

    START_IMAGE = "000001.jpg"
    START_POINT = (480, 270) # uv

    # run_propagation(
    #     COLMAP_PATH,
    #     RESULT_PATH,
    #     IMAGES_DIR,
    #     DEPTHS_DIR,
    #     START_IMAGE,
    #     START_POINT
    # )

    run_propagation_anchor(
        COLMAP_PATH,
        ANCHOR_RESULT_PATH,
        IMAGES_DIR,
        DEPTHS_DIR,
        START_IMAGE,
        START_POINT
    )



