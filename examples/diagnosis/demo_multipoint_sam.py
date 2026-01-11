#!/usr/bin/env python3
"""
Quick demo of multi-point interactive selection for SAM segmentation
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from interactive_point_selector import get_start_point_interactive
    from segment_anything import sam_model_registry, SamPredictor
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure SAM and other dependencies are installed")
    sys.exit(1)


def demo_multipoint_selection(image_path, sam_checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth"):
    """
    Demo: Select multiple points and show SAM segmentation result.
    """
    print("=" * 60)
    print("MULTI-POINT SAM SEGMENTATION DEMO")
    print("=" * 60)
    
    # Step 1: Get points interactively
    print("\nStep 1: Interactive point selection...")
    positive_pts, negative_pts = get_start_point_interactive(image_path)
    
    if not positive_pts:
        print("No points selected. Exiting demo.")
        return
    
    print(f"\nSelected {len(positive_pts)} foreground points")
    if negative_pts:
        print(f"Selected {len(negative_pts)} background points")
    
    # Step 2: Load image
    print("\nStep 2: Loading image...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 3: Load SAM
    print("\nStep 3: Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not os.path.exists(sam_checkpoint):
        print(f"Error: SAM checkpoint not found: {sam_checkpoint}")
        print("Please download SAM checkpoint or update the path")
        return
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Step 4: Run SAM
    print("\nStep 4: Running SAM segmentation...")
    predictor.set_image(image)
    
    # Prepare points
    all_points = positive_pts + (negative_pts if negative_pts else [])
    all_labels = [1] * len(positive_pts) + ([0] * len(negative_pts) if negative_pts else [])
    
    point_coords = np.array(all_points)
    point_labels = np.array(all_labels)
    
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # Step 5: Visualize results
    print("\nStep 5: Visualizing results...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show all three masks
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i+1].imshow(image)
        axes[i+1].imshow(mask, alpha=0.5, cmap='jet')
        
        # Draw points
        pos_pts = np.array(positive_pts)
        axes[i+1].scatter(pos_pts[:, 0], pos_pts[:, 1], color='green', marker='o', 
                         s=100, label='Foreground', edgecolors='white', linewidths=2)
        if negative_pts:
            neg_pts = np.array(negative_pts)
            axes[i+1].scatter(neg_pts[:, 0], neg_pts[:, 1], color='red', marker='x', 
                             s=100, label='Background', linewidths=3)
        
        axes[i+1].set_title(f"Mask {i+1} (score: {score:.3f})")
        axes[i+1].legend()
        axes[i+1].axis('off')
    
    plt.tight_layout()
    output_path = "multipoint_sam_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    plt.show()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"Best mask is usually the one with highest score")
    print(f"Scores: {scores}")


if __name__ == "__main__":
    # Default test image
    test_image = "data/Tree/images_2/000001.jpg"
    
    # Allow custom image path
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if not os.path.exists(test_image):
        print(f"Error: Image not found: {test_image}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [image_path]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} data/Tree/images_2/000001.jpg")
        sys.exit(1)
    
    demo_multipoint_selection(test_image)
