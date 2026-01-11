"""
Interactive point selector for segmentation propagation.
Allows user to click multiple points for foreground and background (exclusion).
"""

import cv2
import numpy as np
from PIL import Image

class PointSelector:
    def __init__(self, image_path):
        """
        Initialize the point selector with an image.
        
        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.positive_points = []  # Foreground points (left click)
        self.negative_points = []  # Background points (right click)
        self.display_image = None
        self.original_image = None
        
    def draw_points(self):
        """Draw all selected points on the display image."""
        self.display_image = self.original_image.copy()
        
        # Draw positive points (foreground) - Green
        for i, (x, y) in enumerate(self.positive_points):
            cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(self.display_image, (x, y), 10, (0, 200, 0), 2)  # Dark green border
            # Number the points
            cv2.putText(self.display_image, f"+{i+1}", (x + 12, y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw negative points (background) - Red
        for i, (x, y) in enumerate(self.negative_points):
            cv2.circle(self.display_image, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(self.display_image, (x, y), 10, (0, 0, 200), 2)  # Dark red border
            # Number the points
            cv2.putText(self.display_image, f"-{i+1}", (x + 12, y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show statistics
        stats_text = f"Foreground: {len(self.positive_points)} | Background: {len(self.negative_points)}"
        cv2.putText(self.display_image, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: add positive point (foreground)
            self.positive_points.append((x, y))
            self.draw_points()
            cv2.imshow('Select Points', self.display_image)
            print(f"Added foreground point #{len(self.positive_points)}: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: add negative point (background)
            self.negative_points.append((x, y))
            self.draw_points()
            cv2.imshow('Select Points', self.display_image)
            print(f"Added background point #{len(self.negative_points)}: ({x}, {y})")
            
    def select_points(self):
        """
        Open an interactive window to select multiple points.
        
        Returns:
            tuple: (positive_points, negative_points) 
                   positive_points: list of (x, y) for foreground
                   negative_points: list of (x, y) for background
        """
        # Read image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot load image: {self.image_path}")
        
        self.display_image = self.original_image.copy()
        
        # Get image dimensions
        height, width = self.original_image.shape[:2]
        
        # Resize if image is too large
        max_display_width = 1200
        max_display_height = 800
        
        if width > max_display_width or height > max_display_height:
            scale = min(max_display_width / width, max_display_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_for_window = cv2.resize(self.display_image, (new_width, new_height))
            is_resized = True
            resize_scale = scale
        else:
            display_for_window = self.display_image
            is_resized = False
            resize_scale = 1.0
        
        # Create window and set mouse callback
        cv2.namedWindow('Select Points')
        
        # If resized, we need to scale the mouse coordinates
        if is_resized:
            def scaled_mouse_callback(event, x, y, flags, param):
                # Scale coordinates back to original image size
                orig_x = int(x / resize_scale)
                orig_y = int(y / resize_scale)
                self.mouse_callback(event, orig_x, orig_y, flags, param)
                
                # Update the resized display
                if self.display_image is not None:
                    display_updated = cv2.resize(self.display_image, (new_width, new_height))
                    cv2.imshow('Select Points', display_updated)
            
            cv2.setMouseCallback('Select Points', scaled_mouse_callback)
        else:
            cv2.setMouseCallback('Select Points', self.mouse_callback)
        
        # Add instructions
        instructions = self.original_image.copy()
        y_offset = 40
        cv2.putText(instructions, "LEFT CLICK: Add foreground point (object)", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 40
        cv2.putText(instructions, "RIGHT CLICK: Add background point (exclude)", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 40
        cv2.putText(instructions, "U: Undo last point | C: Clear all points", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40
        cv2.putText(instructions, "ENTER: Confirm | ESC: Cancel", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if is_resized:
            instructions = cv2.resize(instructions, (new_width, new_height))
        
        cv2.imshow('Select Points', instructions)
        
        print("=" * 60)
        print("MULTI-POINT INTERACTIVE SELECTION")
        print("=" * 60)
        print("Instructions:")
        print("  LEFT CLICK  - Add foreground point (object)")
        print("  RIGHT CLICK - Add background point (exclude)")
        print("  U key       - Undo last point")
        print("  C key       - Clear all points")
        print("  ENTER       - Confirm selection")
        print("  ESC         - Cancel")
        print("=" * 60)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER key
                if len(self.positive_points) > 0:
                    break
                else:
                    print("Please select at least one foreground point!")
                    
            elif key == 27:  # ESC key
                print("Selection cancelled")
                cv2.destroyAllWindows()
                return None, None
                
            elif key == ord('u') or key == ord('U'):  # Undo
                if len(self.negative_points) > 0:
                    removed = self.negative_points.pop()
                    print(f"Removed background point: {removed}")
                elif len(self.positive_points) > 0:
                    removed = self.positive_points.pop()
                    print(f"Removed foreground point: {removed}")
                else:
                    print("No points to undo")
                self.draw_points()
                if is_resized:
                    display_updated = cv2.resize(self.display_image, (new_width, new_height))
                    cv2.imshow('Select Points', display_updated)
                else:
                    cv2.imshow('Select Points', self.display_image)
                    
            elif key == ord('c') or key == ord('C'):  # Clear all
                self.positive_points = []
                self.negative_points = []
                print("Cleared all points")
                self.draw_points()
                if is_resized:
                    display_updated = cv2.resize(self.display_image, (new_width, new_height))
                    cv2.imshow('Select Points', display_updated)
                else:
                    cv2.imshow('Select Points', self.display_image)
        
        cv2.destroyAllWindows()
        
        print("=" * 60)
        print(f"Foreground points: {len(self.positive_points)}")
        for i, pt in enumerate(self.positive_points):
            print(f"  +{i+1}: {pt}")
        print(f"Background points: {len(self.negative_points)}")
        for i, pt in enumerate(self.negative_points):
            print(f"  -{i+1}: {pt}")
        print("=" * 60)
        
        return self.positive_points, self.negative_points


def get_start_point_interactive(image_path):
    """
    Convenience function to get starting points interactively.
    
    Args:
        image_path: Path to the first image
        
    Returns:
        tuple: (positive_points, negative_points) where:
               - positive_points: list of (x, y) for foreground
               - negative_points: list of (x, y) for background
               Returns (None, None) if cancelled
    """
    selector = PointSelector(image_path)
    return selector.select_points()


if __name__ == "__main__":
    # Test the selector
    import os
    
    test_image = "data/Tree/images_2/000001.jpg"
    
    if os.path.exists(test_image):
        positive_pts, negative_pts = get_start_point_interactive(test_image)
        if positive_pts:
            print(f"\nYou can use these points in your script:")
            print(f"POSITIVE_POINTS = {positive_pts}")
            print(f"NEGATIVE_POINTS = {negative_pts}")
    else:
        print(f"Test image not found: {test_image}")
        print("Please update the test_image path")
