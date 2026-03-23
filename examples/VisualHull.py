# type: ignore

import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import open3d as o3d
import struct


class VisualHullGenerator:
    """ Generate Visual Hull from multiple-view mask and colmap geometric info """
    def __init__(self, mask_dir: str, colmap_dir: str, voxel_resolution: int = 256):
        """
        Args:
            mask_dir: .npy mask file
            colmap_dir: COLMAP file
            voxel_resolution: 
        """
        self.mask_dir = Path(mask_dir)
        self.colmap_dir = Path(colmap_dir)
        self.voxel_resolution = voxel_resolution
        
        # camera config
        self.cameras = self.load_colmap_cameras()
        
    def load_colmap_cameras(self):
        from gsplat.datasets.colmap import Parser
        
        parser = Parser(data_dir=str(self.colmap_dir))
        cameras = {
            'K': [],  # intrinsic matrix
            'R': [],  # rotation matrix
            't': [],  # translation vector
            'width': parser.imsize[0],
            'height': parser.imsize[1]
        }
        
        for i in range(len(parser.camtoworlds)):
            # retrieve camera extrinsic parameters
            c2w = parser.camtoworlds[i]
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            cameras['R'].append(R)
            cameras['t'].append(t)
            cameras['K'].append(parser.Ks[i])
            
        return cameras
    
    def load_masks(self) -> List[np.ndarray]:
        """ load .npy mask file"""
        mask_files = sorted(self.mask_dir.glob("*.npy"))
        masks = [np.load(f) for f in mask_files]
        return masks
    
    def load_depth_normal(self):
        """ load COLMAP generated depth and normal info"""
        # TODO change file format
        depth_dir = self.colmap_dir / "stereo" / "depth_maps"
        normal_dir = self.colmap_dir / "stereo" / "normal_maps"
        
        depths = []
        normals = []
        
        if depth_dir.exists():
            for depth_file in sorted(depth_dir.glob("*.bin")):
                depth = self.read_colmap_array(depth_file)
                depths.append(depth)
                
        if normal_dir.exists():
            for normal_file in sorted(normal_dir.glob("*.bin")):
                normal = self.read_colmap_array(normal_file)
                normals.append(normal)
                
        return depths, normals
    
    def read_colmap_array(self, path):
        """ read COLMAP depth/normal data in binary format """
        with open(path, "rb") as f:
            width, height, channels = np.fromfile(f, dtype=np.int32, count=3)
            arr = np.fromfile(f, dtype=np.float32, count=width*height*channels)
            arr = arr.reshape((height, width, channels))
        return arr
    
    def create_voxel_grid(self, bounds: np.ndarray) -> np.ndarray:
        """ create 3D voxel grid """
        x = np.linspace(bounds[0, 0], bounds[1, 0], self.voxel_resolution)
        y = np.linspace(bounds[0, 1], bounds[1, 1], self.voxel_resolution)
        z = np.linspace(bounds[0, 2], bounds[1, 2], self.voxel_resolution)
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        voxels = np.stack([xv, yv, zv], axis=-1)
        return voxels
    
    def project_point(self, point_3d: np.ndarray, K: np.ndarray, 
                     R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """ project 3D point onto 2D screen """
        # WC -> VC
        point_cam = R @ point_3d + t
        
        if point_cam[2] <= 0:
            return None
            
        # project
        point_2d = K @ point_cam
        point_2d = point_2d[:2] / point_2d[2]
        
        return point_2d
    
    def check_visibility(self, point_2d: np.ndarray, mask: np.ndarray) -> bool:
        """ check if the projected 2D point is in the mask """
        h, w = mask.shape[:2]
        x, y = int(point_2d[0]), int(point_2d[1])
        
        if 0 <= x < w and 0 <= y < h:
            return mask[y, x] > 0
        return False
    
    def generate_visual_hull(self, use_depth: bool = True):
        """ generate Visual Hull"""
        # load data
        masks = self.load_masks()
        depths, normals = self.load_depth_normal()
        
        print(f"Loaded {len(masks)} masks")
        
        # scene boundry（can be set by depth or manually）
        bounds = np.array([[-2, -2, -2], [2, 2, 2]], dtype=np.float32)
        
        # create voxel grid
        print("Creating voxel grid...")
        voxels = self.create_voxel_grid(bounds)
        occupancy = np.ones(voxels.shape[:3], dtype=bool)
        
        # do carving for each view
        for i, mask in enumerate(masks):
            print(f"Processing view {i+1}/{len(masks)}...")
            K = self.cameras['K'][i]
            R = self.cameras['R'][i]
            t = self.cameras['t'][i]
            
            # iterate all voxels
            for ix in range(self.voxel_resolution):
                for iy in range(self.voxel_resolution):
                    for iz in range(self.voxel_resolution):
                        if not occupancy[ix, iy, iz]:
                            continue
                            
                        point_3d = voxels[ix, iy, iz]
                        point_2d = self.project_point(point_3d, K, R, t)
                        
                        if point_2d is not None:
                            if not self.check_visibility(point_2d, mask):
                                occupancy[ix, iy, iz] = False
        
        # convert into point cloud / mesh
        points = voxels[occupancy]
        
        return points, occupancy
    
    def save_result(self, points: np.ndarray, output_path: str):
        """ save as point cloud / mesh """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd.estimate_normals()
        
        o3d.io.write_point_cloud(output_path + ".ply", pcd)
        
        # optional: generate mesh by using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        o3d.io.write_triangle_mesh(output_path + "_mesh.ply", mesh)
        
        print(f"Saved results to {output_path}")


def main():
    generator = VisualHullGenerator(
        mask_dir="Tree/mask/Sam2/images_2",
        colmap_dir="Tree/sparse/0/",
        voxel_resolution=256
    )
    
    points, occupancy = generator.generate_visual_hull()
    generator.save_result(points, "visual_hull_output")


if __name__ == "__main__":
    main()

