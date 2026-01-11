#!/usr/bin/env python3
"""Quick test to read COLMAP depth file header"""

import struct
import os

filepath = "data/sofa/dense/stereo/depth_maps/000.jpg.geometric.bin"

if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    exit(1)

filesize = os.path.getsize(filepath)
print(f"File: {filepath}")
print(f"Size: {filesize} bytes\n")

with open(filepath, "rb") as f:
    # Read first 64 bytes
    header = f.read(64)
    
    print("First 64 bytes (hex):")
    for i in range(0, 64, 16):
        hex_str = " ".join(f"{b:02x}" for b in header[i:i+16])
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in header[i:i+16])
        print(f"{i:04x}: {hex_str:47s}  {ascii_str}")
    
    print("\n" + "="*60)
    print("Trying different interpretations:")
    print("="*60)
    
    # Try format 1: width, height (little-endian int32)
    f.seek(0)
    w1 = struct.unpack('<I', f.read(4))[0]  # unsigned int
    h1 = struct.unpack('<I', f.read(4))[0]
    print(f"\n1. uint32 (little-endian): {w1} x {h1}")
    print(f"   Expected size: {8 + w1*h1*4} bytes")
    
    # Try format 2: width, height, channels
    f.seek(0)
    w2 = struct.unpack('<I', f.read(4))[0]
    h2 = struct.unpack('<I', f.read(4))[0]
    c2 = struct.unpack('<I', f.read(4))[0]
    print(f"\n2. uint32 with channels: {w2} x {h2} x {c2}")
    print(f"   Expected size: {12 + w2*h2*c2*4} bytes")
    
    # Try format 3: Swap byte order
    f.seek(0)
    w3 = struct.unpack('>I', f.read(4))[0]  # big-endian
    h3 = struct.unpack('>I', f.read(4))[0]
    print(f"\n3. uint32 (big-endian): {w3} x {h3}")
    print(f"   Expected size: {8 + w3*h3*4} bytes")
    
    # Try format 4: Maybe it's char + width + height?
    f.seek(0)
    chars = f.read(1)
    w4 = struct.unpack('<I', f.read(4))[0]
    h4 = struct.unpack('<I', f.read(4))[0]
    print(f"\n4. char + uint32: {chars} + {w4} x {h4}")
    print(f"   Expected size: {9 + w4*h4*4} bytes")
    
    # Check which matches file size
    print(f"\n{'='*60}")
    print("Matching file size:")
    print(f"{'='*60}")
    
    for i, (w, h, offset, extra) in enumerate([
        (w1, h1, 8, 0),
        (w2, h2, 12, c2),
        (w3, h3, 8, 0),
        (w4, h4, 9, 0)
    ], 1):
        if extra:
            expected = offset + w * h * extra * 4
        else:
            expected = offset + w * h * 4
        diff = abs(expected - filesize)
        match = "✓ MATCH!" if diff < 100 else ""
        print(f"Format {i}: {w:6d} x {h:6d}  Expected: {expected:10d}  Diff: {diff:6d}  {match}")
