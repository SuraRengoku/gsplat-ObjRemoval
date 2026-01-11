# Centroid-based Anchor Tracking

## 问题描述

使用固定锚点进行3D追踪时，当相机旋转后，锚点可能会投影到物体边缘，导致SAM分割效果变差。

## 解决方案

### 1. 使用Mask质心作为锚点 (`use_centroid=True`)

**问题：** 用户点击可能在物体边缘  
**解决：** 在第一帧使用SAM分割后，计算mask的质心（重心）作为锚点

```python
# 计算mask质心
mask_coords = np.argwhere(mask)  # 获取所有mask像素坐标
centroid_x = np.mean(mask_coords[:, 1])  # 平均x坐标
centroid_y = np.mean(mask_coords[:, 0])  # 平均y坐标
```

**优势：** 锚点在物体中心，更稳定

### 2. 动态更新锚点 (`update_anchor=True`)

**问题：** 物体在不同视角下，固定的3D点可能投影到边缘  
**解决：** 每一帧都基于新的mask质心更新3D锚点

```python
# 每帧流程：
1. 投影旧锚点到当前帧 -> projected_point
2. 用projected_point做SAM分割 -> new_mask
3. 计算new_mask的质心 -> refined_point
4. 用refined_point的深度更新3D锚点
5. 下一帧使用更新后的锚点
```

**优势：** 
- 自适应调整到物体中心
- 处理视角变化和物体旋转
- 防止锚点漂移到边缘

**安全措施：**
- 如果质心偏移 > 50px，认为分割错误，不更新锚点
- 避免错误分割导致追踪失败

## 使用方法

```python
from propagate_segmentation import run_propagation_anchor

# 方法1：最优配置（推荐）
run_propagation_anchor(
    colmap_path,
    result_path,
    images_dir,
    depths_dir,
    start_img_name,
    start_point,
    use_centroid=True,    # 使用质心
    update_anchor=True    # 动态更新
)

# 方法2：仅使用质心，不更新
run_propagation_anchor(
    ...,
    use_centroid=True,
    update_anchor=False   # 固定在第一帧质心
)

# 方法3：原始方法（对比用）
run_propagation_anchor(
    ...,
    use_centroid=False,   # 使用点击点
    update_anchor=False   # 固定锚点
)
```

## 可视化说明

改进后的可视化会显示：
- **黄色圆圈 (○)**: 用户点击点
- **红色星号 (★)**: 锚点（质心）
- **蓝色叉号 (×)**: 3D投影点
- **红色星号 (★)**: 优化后的质心点

## 对比实验

```bash
# 测试不同配置
python examples/test_anchor_methods.py
```

结果保存在：
- `data/Tree_Marked_Centroid/` - 改进方法（质心+更新）
- `data/Tree_Marked_Anchor/` - 原始方法（点击点+固定）

## 适用场景

✅ **适合：**
- 相机旋转较大的场景
- 物体在不同视角下形状变化
- 用户点击不在物体中心
- 需要长时间追踪

⚠️ **注意：**
- 需要SAM分割质量好
- 物体需要有明显的边界
- 深度图需要准确

## 参数调整

```python
# 在代码中可以调整：
shift_magnitude < 50  # 质心偏移阈值（默认50像素）
                      # 太小：更新太保守
                      # 太大：可能接受错误分割
```

## 技术原理

1. **质心计算**: 加权平均所有mask像素位置
2. **3D更新**: 使用新质心的深度值反投影到3D空间
3. **稳定性检查**: 限制质心偏移量，防止跳变
4. **迭代优化**: 每帧都向物体中心收敛

## 预期效果

- **减少边缘效应**: 锚点始终接近物体中心
- **提高分割质量**: SAM在物体中心分割更准确
- **增强鲁棒性**: 适应视角变化和物体旋转
- **减少追踪失败**: 避免锚点飘移到背景
