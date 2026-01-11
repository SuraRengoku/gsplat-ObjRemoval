import os
import numpy as np
import matplotlib.pyplot as plt
from BinaryReader import read_colmap_depth

def compare_depth_maps(colmap_depth_dir, gsplat_depth_dir, image_name):
    """
    Compare COLMAP depth vs gsplat depth for the same image.
    """
    base_name = os.path.splitext(image_name)[0]
    
    # Load COLMAP depth
    colmap_path = os.path.join(colmap_depth_dir, f"{base_name}.geometric.bin")
    if not os.path.exists(colmap_path):
        colmap_path = os.path.join(colmap_depth_dir, f"{base_name}.photometric.bin")
    
    colmap_depth = read_colmap_depth(colmap_path)
    
    # Load gsplat depth
    gsplat_path = os.path.join(gsplat_depth_dir, f"{base_name}.npy")
    gsplat_depth = np.load(gsplat_path)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # COLMAP depth
    im1 = axes[0, 0].imshow(colmap_depth, cmap='turbo')
    axes[0, 0].set_title('COLMAP Depth')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # gsplat depth
    im2 = axes[0, 1].imshow(gsplat_depth, cmap='turbo')
    axes[0, 1].set_title('gsplat Depth')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference (after resampling to same size)
    from scipy.ndimage import zoom
    if colmap_depth.shape != gsplat_depth.shape:
        scale_y = gsplat_depth.shape[0] / colmap_depth.shape[0]
        scale_x = gsplat_depth.shape[1] / colmap_depth.shape[1]
        colmap_depth_resized = zoom(colmap_depth, (scale_y, scale_x), order=1)
    else:
        colmap_depth_resized = colmap_depth
    
    diff = np.abs(colmap_depth_resized - gsplat_depth)
    im3 = axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Histograms
    axes[1, 0].hist(colmap_depth.flatten(), bins=100, alpha=0.7, label='COLMAP')
    axes[1, 0].set_xlabel('Depth Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('COLMAP Depth Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(gsplat_depth.flatten(), bins=100, alpha=0.7, label='gsplat', color='orange')
    axes[1, 1].set_xlabel('Depth Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('gsplat Depth Distribution')
    axes[1, 1].legend()
    
    # Statistics
    stats_text = f"""COLMAP Depth:
    Min: {colmap_depth.min():.4f}
    Max: {colmap_depth.max():.4f}
    Mean: {colmap_depth.mean():.4f}
    Std: {colmap_depth.std():.4f}
    
gsplat Depth:
    Min: {gsplat_depth.min():.4f}
    Max: {gsplat_depth.max():.4f}
    Mean: {gsplat_depth.mean():.4f}
    Std: {gsplat_depth.std():.4f}
    
Difference:
    Mean Error: {diff.mean():.4f}
    Max Error: {diff.max():.4f}"""
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'depth_comparison_{base_name}.png', dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: depth_comparison_{base_name}.png")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print(f"Depth Map Comparison: {image_name}")
    print("="*60)
    print(stats_text)

if __name__ == "__main__":
    compare_depth_maps(
        "data/sofa/stereo/depth_maps",
        "results/sofa/train_depths",
        "000.jpg"
    )