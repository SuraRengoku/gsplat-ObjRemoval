#!/usr/bin/env python3
"""
测试脚本：检查绿色前景检测是否正常工作
"""

import sys
from pathlib import Path
import imageio
import torch
import matplotlib.pyplot as plt
import numpy as np

def test_green_detection(image_path, output_path="debug_green_mask.png"):
    """测试绿色区域检测"""
    
    # 绿色阈值
    green_threshold_lower = (0.0, 0.5, 0.0)
    green_threshold_upper = (0.5, 1.0, 0.5)
    
    # 加载图片
    print(f"Loading image: {image_path}")
    img = imageio.imread(image_path)
    
    if len(img.shape) == 2:
        print("Error: Image is grayscale")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image range: [{img.min()}, {img.max()}]")
    
    # 归一化到 [0, 1]
    img_norm = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm)
    
    # 检测绿色区域
    lower = torch.tensor(green_threshold_lower)
    upper = torch.tensor(green_threshold_upper)
    
    green_mask = (img_tensor >= lower) & (img_tensor <= upper)
    green_mask = green_mask.all(dim=-1).float()
    
    # 统计
    total_pixels = green_mask.numel()
    green_pixels = green_mask.sum().item()
    green_ratio = green_pixels / total_pixels
    
    print(f"\nDetection Results:")
    print(f"Total pixels: {total_pixels}")
    print(f"Green pixels: {int(green_pixels)}")
    print(f"Green ratio: {green_ratio:.2%}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Green mask
    axes[1].imshow(green_mask.numpy(), cmap='gray')
    axes[1].set_title(f"Green Mask ({green_ratio:.2%} detected)")
    axes[1].axis('off')
    
    # Overlay
    overlay = img.copy()
    mask_np = green_mask.numpy()
    overlay[mask_np > 0.5] = [255, 0, 0]  # 红色标记检测到的区域
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (detected in red)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return green_mask

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_green_detection.py <image_path>")
        print("Example: python test_green_detection.py data/Tree_Filled/images/tree_filled1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    test_green_detection(image_path)
