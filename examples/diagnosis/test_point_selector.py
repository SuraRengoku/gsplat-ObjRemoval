#!/usr/bin/env python3
"""
Quick test script for interactive point selection
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from interactive_point_selector import get_start_point_interactive

def test_interactive_selector():
    """Test the interactive point selector."""
    
    # Test with Tree dataset
    test_image = "data/Tree/images_2/000001.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("\nPlease update the test_image path in this script")
        print("Or provide an image path as argument:")
        print(f"  python {sys.argv[0]} path/to/image.jpg")
        return
    
    print("Testing interactive point selector...")
    print(f"Image: {test_image}\n")
    
    point = get_start_point_interactive(test_image)
    
    if point:
        print("\n" + "=" * 60)
        print("SUCCESS! Point selected successfully")
        print("=" * 60)
        print(f"Selected coordinates: {point}")
        print(f"\nYou can use this in your code:")
        print(f"  START_POINT = {point}")
        print("=" * 60)
    else:
        print("\nSelection cancelled or failed")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use custom image path if provided
        custom_image = sys.argv[1]
        if os.path.exists(custom_image):
            point = get_start_point_interactive(custom_image)
            if point:
                print(f"\nSelected point: {point}")
        else:
            print(f"Error: Image not found: {custom_image}")
    else:
        test_interactive_selector()
