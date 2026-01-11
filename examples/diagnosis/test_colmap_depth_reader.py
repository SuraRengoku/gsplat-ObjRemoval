import os
import numpy as np
from BinaryReader import read_colmap_depth

# Test file
depth_file = "data/sofa/dense/stereo/depth_maps/000.jpg.geometric.bin"

print(f"File: {depth_file}")
print(f"Size: {os.path.getsize(depth_file)} bytes\n")

# Print first 64 bytes in hex
with open(depth_file, "rb") as f:
    data = f.read(64)
    print("First 64 bytes (hex):")
    for i in range(0, len(data), 16):
        hex_str = " ".join(f"{b:02x}" for b in data[i:i+16])
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in data[i:i+16])
        print(f"{i:04x}: {hex_str:<48}  {ascii_str}")

print("\n" + "="*60)
print("Testing new read_colmap_depth() function:")
print("="*60)

try:
    depth = read_colmap_depth(depth_file)
    print(f"\n✓ SUCCESS!")
    print(f"  Shape: {depth.shape}")
    print(f"  Dtype: {depth.dtype}")
    print(f"  Min: {depth.min():.4f}")
    print(f"  Max: {depth.max():.4f}")
    print(f"  Mean: {depth.mean():.4f}")
    print(f"  Non-zero: {(depth > 0).sum()} / {depth.size}")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
