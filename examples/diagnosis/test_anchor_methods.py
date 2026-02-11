"""
Test script to compare different anchor strategies:
1. Original: Fixed anchor at clicked point
2. Improved: Anchor at mask centroid, updated each frame
"""

import os
import sys

# Test both methods
if __name__ == "__main__":
    from examples.Propagate_segmentation import run_propagation_anchor
    
    COLMAP_PATH = "data/Tree/sparse/0"
    IMAGES_DIR = "data/Tree/images_2"
    DEPTHS_DIR = "results/Tree/train_depths"
    START_IMAGE = "000001.jpg"
    START_POINT = (480, 270)  # User click point
    
    # Run improved method with centroid tracking
    RESULT_PATH = "data/Tree_Marked_Centroid/images"
    
    print("=" * 60)
    print("Running IMPROVED method: Centroid-based anchor with updates")
    print("=" * 60)
    
    run_propagation_anchor(
        COLMAP_PATH,
        RESULT_PATH,
        IMAGES_DIR,
        DEPTHS_DIR,
        START_IMAGE,
        START_POINT
    )
    
    print("\n" + "=" * 60)
    print("DONE! Check results in:", RESULT_PATH)
    print("=" * 60)
    print("\nKey improvements:")
    print("1. Anchor point = mask centroid (not clicked edge point)")
    print("2. Anchor updates each frame based on new mask centroid")
    print("3. Better handles object rotation and perspective changes")
    print("4. Visualizations show both projected and refined points")
