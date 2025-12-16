import os
import struct
import numpy as np
import collections

# --- data struct definition ---

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# --- utilities ---

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

def get_camera_center_and_rotation(image):
    # compute the camera position and rotation matrix in world coordinates
    # 1. get rotation matrix R (World-to-Camera)
    R = qvec2rotmat(image.qvec)
    # 2. get translation vector t (World-to-Camera)
    t = image.tvec
    # 3. camera center C = -R^T * t
    camera_center = -np.dot(R.T, t)
    # 4. camera rotation matrix (Camera-to-World)
    rotation_c2w = R.T
    
    return camera_center, rotation_c2w

# --- read ---

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            if model_id == 0: params = struct.unpack("<3d", fid.read(24)) # SIMPLE_PINHOLE
            elif model_id == 1: params = struct.unpack("<4d", fid.read(32)) # PINHOLE
            else: 
                # simplified，suppose we use SIMPLE_PINHOLE, actually we should reference number of paramters by model_id
                # TODO: if use complex model
                params = struct.unpack("<3d", fid.read(24)) 
            
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_id,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            points2D = struct.unpack("<" + "ddq" * num_points2D, fid.read(24 * num_points2D))
            points2D = np.array(points2D).reshape((num_points2D, 3))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=points2D[:, :2], point3D_ids=points2D[:, 2].astype(int))
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_properties = struct.unpack("<QdddBBBd", fid.read(43))
            point3D_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4])
            rgb = np.array(binary_point_properties[4:7])
            error = binary_point_properties[7]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack("<" + "ii" * track_length, fid.read(8 * track_length))
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def read_model(path, ext=".bin"):
    if ext == ".txt":
        # currently no .txt files
        pass
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D" + ext))
        return cameras, images, points3D

if __name__ == "__main__":
    model_path = "data/Tree_Filled/sparse/0"
    
    print(f"reading model: {model_path} ...")
    cameras, images, points3D = read_model(model_path) 
    
    print(f"\n--- statistics ---")
    print(f"#cameras: {len(cameras)}")
    print(f"#images: {len(images)}")
    print(f"#3D points: {len(points3D)}")
    
    # information of camera 0
    first_cam_id = list(cameras.keys())[0]
    print(f"\n--- camera example (ID: {first_cam_id}) ---")
    print(cameras[first_cam_id])
    
    # information of image 0
    first_img_id = list(images.keys())[0]
    img = images[first_img_id]
    print(f"\n--- image example (ID: {first_img_id}) ---")
    print(f"file name: {img.name}")
    print(f"posture (transformation): {img.tvec}")
    
    center, rot_c2w = get_camera_center_and_rotation(img)
    print(f"Camera Center (World Coords): {center}")
    print(f"Camera Forward Vector: {rot_c2w[:, 2]}") # z-axis-> camera view direction

    print(f"#related 3D points: {np.sum(img.point3D_ids != -1)}")
    
    # information of 3D point 0
    first_point_id = list(points3D.keys())[0]
    pt = points3D[first_point_id]
    print(f"\n--- 3D point example (ID: {first_point_id}) ---")
    print(f"coordinate: {pt.xyz}")
    print(f"color: {pt.rgb}")
    print(f"reprojection error: {pt.error}")