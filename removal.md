# GSplat Object Removal Pipeline

## Reference
https://github.com/jonstephens85/gsplat_3dgut

---

## 1. Environment Setup

### Prerequisites
```bash
# Install dependencies
sudo apt install imagemagick colmap

# Create conda environment
conda create -n gsplat python=3.10
conda activate gsplat
```

---

## 2. Data Preparation

### 2.1 Organize Dataset
```bash
# Create dataset folder structure
mkdir -p data/xxx/images
# Put your images into data/xxx/images/
```

### 2.2 Resize Images (Optional)
```bash
cd data/xxx
mkdir images_2
magick mogrify -path images_2 -resize 50% images/*.jpg
# or for PNG: images/*.png
```

---

## 3. COLMAP Processing

### 3.1 Structure from Motion (SfM)
```bash
cd data/xxx
mkdir sparse

# Feature extraction
colmap feature_extractor \
  --database_path database.db \
  --image_path images \
  --ImageReader.camera_model SIMPLE_PINHOLE \
  --ImageReader.single_camera 1

# Feature matching
colmap exhaustive_matcher --database_path database.db

# Sparse reconstruction
colmap mapper \
  --database_path database.db \
  --image_path images \
  --output_path sparse
```

### 3.2 Dense Reconstruction (Optional, for COLMAP depth maps)
⚠️ **This may take a long time**

```bash
# Undistort images
mkdir dense
colmap image_undistorter \
  --image_path images \
  --input_path sparse/0 \
  --output_path dense \
  --output_type COLMAP

# Generate depth maps
colmap patch_match_stereo \
  --workspace_path dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true
```

**Output**: Depth maps at `dense/stereo/depth_maps/*.geometric.bin`

### 3.3 Point Cloud Fusion (Optional)
```bash
colmap stereo_fusion \
  --workspace_path dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path dense/fused.ply
```

---

## 4. GSplat Training

### 4.1 Basic Training
```bash
python examples/simple_trainer.py default \
  --data_dir data/xxx/ \
  --data_factor 2 \
  --result_dir results/xxx
```

### 4.2 Training with Depth/PLY Output
```bash
python examples/simple_trainer.py default \
  --data_dir data/xxx/ \
  --data_factor 2 \
  --result_dir results/xxx \
  --save_train_depths \
  --save_ply
```

### 4.3 Generate Depth Maps After Training
```bash
python examples/simple_trainer.py default \
  --data_dir data/xxx/ \
  --data_factor 2 \
  --result_dir results/xxx \
  --ckpt results/xxx/ckpts/ckpt_30000_rank0.pt \
  --disable_viewer
```

---

## 5. Object Removal with SAM

### 5.1 Interactive Segmentation Propagation
```bash
python examples/propagate_segmentation.py \
  --use_colmap_depth \
  --max_frames 100
```

**Interactive Controls:**
- `LEFT CLICK` - Add foreground point (object to remove)
- `RIGHT CLICK` - Add background point (keep)
- `U` - Undo last point
- `C` - Clear all points
- `ENTER` - Confirm and start propagation
- `ESC` - Cancel

**Output Files:**
- `{frame}.jpg` - Green screen replacement (RGB: 0, 177, 64)
- `{frame}_debug.jpg` - Visualization with annotations
- `{frame}.npy` - Raw mask data

### 5.2 Propagation Methods
```python
# Method 1: Geometry-based (default in run_propagation_anchor)
# Uses 3D point tracking + COLMAP/GSplat depth

# Method 2: Geometry + Optical Flow (run_propagation_with_flow)
# Hybrid approach for handling occlusions

# Method 3: SAM2 Video (propagation_with_sam2)
# End-to-end video object segmentation
```

---

## 6. Depth Map Formats

### COLMAP Depth Maps
- **Location**: `data/xxx/dense/stereo/depth_maps/`
- **Format**: `{image_name}.geometric.bin` or `{image_name}.photometric.bin`
- **Structure**: ASCII header `"width&height&channels&"` + float32 array
- **Example**: `000.jpg.geometric.bin` for image `000.jpg`

### GSplat Depth Maps
- **Location**: `results/xxx/depths/`
- **Format**: NumPy `.npy` files
- **Structure**: Direct float32 array (H, W)

---

## Common Issues

### 1. COLMAP depth reading errors
- Ensure dense reconstruction completed: `ls data/xxx/dense/stereo/depth_maps/`
- Check file naming: Should be `000.jpg.geometric.bin`, not `000.geometric.bin`

### 2. Propagation drift
- Use `--use_colmap_depth` for more accurate depth maps
- Add more background points to constrain segmentation
- Reduce `--max_frames` for testing

### 3. Green screen artifacts
- Adjust mask threshold in SAM settings
- Check debug images for segmentation quality