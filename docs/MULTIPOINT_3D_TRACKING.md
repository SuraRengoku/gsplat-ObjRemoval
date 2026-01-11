# 多点3D投影追踪说明

## 🎯 核心改进

现在所有选择的点都会参与3D投影和重投影过程，而不仅仅是在第一帧使用。

## 📐 工作原理

### 第一帧：初始化
```
1. 用户选择多个点：
   - 左键：前景点 (绿色) +1, +2, +3...
   - 右键：背景点 (红色) -1, -2, -3...

2. SAM分割：
   输入：所有选择的点 + 标签
   输出：初始mask

3. 转换为3D锚点：
   每个2D点 → (深度图) → 3D世界坐标
   
   正例: [(u1,v1), (u2,v2), ...] → [P1_world, P2_world, ...]
   负例: [(u3,v3), (u4,v4), ...] → [P3_world, P4_world, ...]
```

### 后续帧：投影和追踪
```
1. 3D投影到当前帧：
   所有3D锚点 → (相机参数) → 2D投影点
   
   P1_world → (u1', v1')  ✓ 在图像内
   P2_world → (u2', v2')  ✓ 在图像内
   P3_world → behind camera ✗ 丢弃
   P4_world → (u4', v4')  ✓ 在图像内

2. SAM分割：
   输入：所有有效的投影点 + 对应标签
   点: [(u1',v1'), (u2',v2'), (u4',v4')]
   标签: [1, 1, 0]  # 1=前景, 0=背景
   
   输出：当前帧的mask

3. 更新锚点（可选）：
   计算mask质心 → 转换为新的3D锚点
   下一帧使用更新后的锚点
```

## 🔄 两种追踪模式

### 模式1: 固定多点追踪 (`update_anchor=False`)
```python
run_propagation_anchor(..., update_anchor=False)
```

**特点**：
- 保持初始选择的所有3D点不变
- 每帧投影所有初始点
- 适合静止物体或小幅度运动

**流程**：
```
Frame 1: Points [P1, P2, P3] → Mask1
Frame 2: Points [P1, P2, P3] → Project → SAM → Mask2
Frame 3: Points [P1, P2, P3] → Project → SAM → Mask3
...
```

### 模式2: 自适应质心追踪 (`update_anchor=True`)
```python
run_propagation_anchor(..., update_anchor=True)
```

**特点**：
- 第一帧使用所有选择点
- 从第二帧开始，切换到质心追踪
- 锚点自动调整到物体中心
- 适合旋转或视角变化大的场景

**流程**：
```
Frame 1: Points [P1, P2, P3] → Mask1 → Centroid C1
Frame 2: Point [C1] → Project → SAM → Mask2 → Centroid C2
Frame 3: Point [C2] → Project → SAM → Mask3 → Centroid C3
...
```

**原因**：
- 多点在初始帧提供更准确的分割
- 质心追踪在后续帧更稳定（单点，减少噪声）
- 自动适应物体旋转和视角变化

## 📊 可视化说明

### 第一帧
- **绿色圆圈**: 用户选择的前景点
- **红色X**: 用户选择的背景点
- **青色星号**: 计算出的质心（如果use_centroid=True）

### 后续帧

#### update_anchor=False（固定多点）
- **绿色圆圈**: 投影的前景点
- **红色X**: 投影的背景点
- 图例显示点数量

#### update_anchor=True（自适应质心）
- **青色星号**: 投影的质心点
- 图例显示"Centroid"

## 💡 使用建议

### 何时使用多个前景点？
```
✓ 物体形状复杂（如：手、树、动物）
✓ 物体颜色不均匀
✓ 需要精确的初始分割
✓ 物体部分被遮挡

✗ 物体形状简单（如：球、盒子）
✗ 对比度明显
✗ 追求速度（少点=快）
```

### 何时使用背景排除点？
```
✓ 物体和背景颜色相似
✓ 有反光或阴影干扰
✓ 附近有相似物体
✓ 初始分割包含了不想要的区域

示例：
- 绿色树在草地上 → 添加草地排除点
- 白色物体+白色背景 → 添加背景排除点
```

### 推荐配置

**场景1：简单静止物体**
```python
操作: 左键点击1次
配置: use_centroid=True, update_anchor=False
```

**场景2：复杂物体，小运动**
```python
操作: 左键点击3-5次 + （可选）右键排除
配置: use_centroid=True, update_anchor=False
说明: 多点提供准确初始分割，固定追踪保持精度
```

**场景3：复杂物体，大幅旋转/视角变化**
```python
操作: 左键点击3-5次 + （可选）右键排除
配置: use_centroid=True, update_anchor=True
说明: 多点初始化，质心自适应追踪
```

## 🔧 技术细节

### 3D点过滤规则
```python
# 投影时，点会被过滤如果：
1. 投影到相机后面 (Z <= 0)
2. 投影到图像外 (x < 0 或 x >= width 或 y < 0 或 y >= height)

# 至少需要1个有效前景点才能继续
if len(projected_positive_pts) == 0:
    skip_frame()
```

### 质心更新逻辑
```python
if update_anchor:
    # 计算mask质心
    centroid = mask.mean(axis=0)
    
    # 检查偏移量
    shift = distance(centroid, average_projected_points)
    
    if shift < 50:  # 合理范围
        # 用质心替换所有锚点
        anchor_3d_points = [centroid_3d]
    else:
        # 保持原锚点
        keep_previous_anchors()
```

### SAM输入格式
```python
# 多点输入
point_coords = np.array([
    [u1, v1],  # 前景点1
    [u2, v2],  # 前景点2
    [u3, v3],  # 背景点1
])

point_labels = np.array([
    1,  # 前景
    1,  # 前景
    0,  # 背景
])

predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)
```

## 📈 预期效果

### 优势
✅ 更准确的初始分割（多点比单点好）
✅ 自动处理点在相机后面的情况
✅ 自动处理点超出图像范围
✅ 可以使用排除点提高分割质量
✅ 自适应追踪适应视角变化

### 注意事项
⚠️ 更多点 = 更慢（每帧需要投影更多点）
⚠️ 质心更新可能在大跳跃时失败
⚠️ 背景点在update_anchor=True时会被清除

## 🚀 快速开始

```bash
# 运行交互式多点选择
python examples/propagate_segmentation.py
```

**操作**：
1. 左键点击物体多个位置（3-5次）
2. （可选）右键点击背景排除区域
3. 按ENTER确认
4. 观察结果中每帧的投影点

**检查可视化**：
- 第一帧：应该看到所有选择的点
- 后续帧（固定模式）：应该看到投影的多个点
- 后续帧（自适应模式）：应该看到质心点

## 🔬 调试技巧

### 查看投影信息
```
控制台输出：
  Projected 3 positive, 1 negative points
  
说明：
- 3个前景点成功投影
- 1个背景点成功投影
- 其他点可能在相机后面或图像外
```

### 查看3D锚点
```
控制台输出：
  Positive point (480, 270) -> 3D: [1.2, 3.4, 5.6]
  Positive point (500, 300) -> 3D: [1.3, 3.5, 5.5]
  
说明：显示每个2D点及其对应的3D世界坐标
```

### 检查质心更新
```
控制台输出：
  Mask centroid shift: (5.3, -2.1) px, magnitude: 5.7
  Updated to single centroid anchor: [1.25, 3.45, 5.55]
  
说明：
- 质心偏移了5.7像素（<50，可接受）
- 已更新为新的3D锚点
```

## 🎓 总结

这个改进实现了真正的**多点3D追踪**：
1. ✅ 所有选择点转为3D锚点
2. ✅ 每帧投影所有3D锚点
3. ✅ SAM接收所有投影点进行分割
4. ✅ 可选的质心自适应更新
5. ✅ 自动过滤无效投影点

相比之前只在第一帧使用多点，现在**每一帧都利用多点信息**，获得更准确和稳定的追踪效果！
