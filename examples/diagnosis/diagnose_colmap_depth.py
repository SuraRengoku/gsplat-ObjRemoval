#!/usr/bin/env python3
"""
Diagnose COLMAP depth map binary file format.
"""

import struct
import numpy as np
import os
import sys

def diagnose_depth_file(filepath):
    """Diagnose the binary format of a COLMAP depth file."""
    
    print("=" * 60)
    print(f"Diagnosing: {filepath}")
    print("=" * 60)
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return
    
    filesize = os.path.getsize(filepath)
    print(f"File size: {filesize} bytes")
    
    with open(filepath, "rb") as f:
        # Read first 32 bytes for inspection
        header_bytes = f.read(32)
        
        print("\nFirst 32 bytes (hex):")
        print(" ".join(f"{b:02x}" for b in header_bytes))
        
        # Try different interpretations
        print("\n--- Trying different header formats ---")
        
        # Try 1: Two int32 (little endian)
        f.seek(0)
        try:
            width_le = struct.unpack('<i', f.read(4))[0]
            height_le = struct.unpack('<i', f.read(4))[0]
            print(f"1. Little-endian int32: width={width_le}, height={height_le}")
            expected_size = 8 + width_le * height_le * 4
            print(f"   Expected file size: {expected_size} bytes")
            if expected_size == filesize:
                print(f"   ✓ MATCH! This is likely the correct format.")
        except:
            print("   ✗ Failed to read as little-endian int32")
        
        # Try 2: Two int32 (big endian)
        f.seek(0)
        try:
            width_be = struct.unpack('>i', f.read(4))[0]
            height_be = struct.unpack('>i', f.read(4))[0]
            print(f"2. Big-endian int32: width={width_be}, height={height_be}")
            expected_size = 8 + width_be * height_be * 4
            print(f"   Expected file size: {expected_size} bytes")
            if expected_size == filesize:
                print(f"   ✓ MATCH! This is likely the correct format.")
        except:
            print("   ✗ Failed to read as big-endian int32")
        
        # Try 3: Two uint32 (little endian)
        f.seek(0)
        try:
            width_ule = struct.unpack('<I', f.read(4))[0]
            height_ule = struct.unpack('<I', f.read(4))[0]
            print(f"3. Little-endian uint32: width={width_ule}, height={height_ule}")
            expected_size = 8 + width_ule * height_ule * 4
            print(f"   Expected file size: {expected_size} bytes")
            if expected_size == filesize:
                print(f"   ✓ MATCH! This is likely the correct format.")
        except:
            print("   ✗ Failed to read as little-endian uint32")
        
        # Try 4: Check if it's a different format entirely
        print("\n--- Checking alternative formats ---")
        
        # Maybe it has no header and is just raw float32?
        num_floats = filesize // 4
        possible_dims = []
        for w in range(100, 2000):
            if num_floats % w == 0:
                h = num_floats // w
                if 100 <= h <= 2000:
                    possible_dims.append((w, h))
        
        if possible_dims:
            print(f"4. Raw float32 array (no header):")
            print(f"   Possible dimensions: {possible_dims[:5]}")  # Show first 5
        
        # Sample some depth values to see if they make sense
        print("\n--- Depth value samples ---")
        f.seek(8)  # Skip header
        sample_values = struct.unpack('<10f', f.read(40))
        print(f"First 10 depth values: {sample_values}")
        
        # Check for special values
        f.seek(8)
        all_depths = np.fromfile(f, dtype=np.float32)
        print(f"\nTotal depth values: {len(all_depths)}")
        print(f"Min depth: {all_depths.min():.4f}")
        print(f"Max depth: {all_depths.max():.4f}")
        print(f"Mean depth: {all_depths.mean():.4f}")
        print(f"Std depth: {all_depths.std():.4f}")
        print(f"Num zeros: {np.sum(all_depths == 0)}")
        print(f"Num negative: {np.sum(all_depths < 0)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_colmap_depth.py <depth_file.bin>")
        print("\nExample:")
        print("  python diagnose_colmap_depth.py data/sofa/dense/stereo/depth_maps/000.jpg.geometric.bin")
        sys.exit(1)
    
    diagnose_depth_file(sys.argv[1])
