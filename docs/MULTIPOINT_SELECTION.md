# 多点选择功能更新总结

## ✅ 新功能

### 1. 多点前景选择
- 可以选择多个点来确保完整分割物体
- 每个点用绿色圆圈标记（+1, +2, +3...）
- 适用于复杂形状或颜色不均匀的物体

### 2. 背景点排除
- 右键点击可以添加排除点（negative points）
- 每个点用红色X标记（-1, -2, -3...）
- 告诉SAM哪些区域不属于目标物体

### 3. 编辑功能
- **U键**: 撤销最后一个点
- **C键**: 清空所有点，重新开始
- 实时显示当前选择的点数量

## 🎯 使用场景

### 场景1: 简单物体（单点足够）
```
操作: 左键点击物体中心 → ENTER
示例: 分割一个球、一棵树
```

### 场景2: 复杂物体（需要多个前景点）
```
操作: 左键点击物体多个位置 → ENTER
示例: 分割一只手（点击手指、手掌等多个位置）
```

### 场景3: 需要排除背景（使用排除点）
```
操作: 左键点击物体 + 右键点击背景 → ENTER
示例: 绿色的树在绿色草地上
      - 左键点击树干和树叶（前景）
      - 右键点击草地（排除）
```

## 📊 快速对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 前景点 | 1个（单击） | 多个（左键多次点击） |
| 背景点 | ❌ 不支持 | ✅ 支持（右键点击） |
| 撤销 | ❌ 重新点击覆盖 | ✅ U键撤销 |
| 清空 | ❌ 不支持 | ✅ C键清空 |
| 可视化 | 黄色圆圈 | 绿色圆圈(前景) + 红色X(背景) |

## 🚀 快速开始

### 基本使用
```bash
# 直接运行，会弹出交互窗口
python examples/propagate_segmentation.py
```

操作：
1. 左键点击物体（可以多次）
2. （可选）右键点击背景排除区域
3. 按ENTER确认

### 测试多点选择
```bash
# 单独测试选择器
python examples/interactive_point_selector.py

# 测试完整SAM分割效果
python examples/demo_multipoint_sam.py
```

### 手动指定多点
```python
# 在 propagate_segmentation.py 中
USE_INTERACTIVE = False
POSITIVE_POINTS_MANUAL = [(480, 270), (500, 300), (450, 280)]  # 前景
NEGATIVE_POINTS_MANUAL = [(200, 150)]  # 背景（可选）
```

## 🔧 技术实现

### SAM多点API
```python
# 准备点坐标和标签
positive_points = [(x1, y1), (x2, y2), ...]  # 前景点
negative_points = [(x3, y3), (x4, y4), ...]  # 背景点

all_points = positive_points + negative_points
all_labels = [1, 1, ...] + [0, 0, ...]  # 1=前景, 0=背景

# 调用SAM
predictor.predict(
    point_coords=np.array(all_points),
    point_labels=np.array(all_labels),
    multimask_output=True
)
```

### 向后兼容性
- ✅ 保持单点模式（`start_point`参数）
- ✅ 新增多点模式（`positive_points`, `negative_points`参数）
- ✅ 两种模式可以切换

## 📁 更新的文件

1. **interactive_point_selector.py**
   - 重写为多点选择模式
   - 添加左键/右键区分
   - 添加U键撤销、C键清空

2. **propagate_segmentation.py**
   - `run_propagation_anchor()` 添加多点参数
   - 更新可视化显示所有选择点
   - `__main__` 部分支持多点模式

3. **demo_multipoint_sam.py** (新增)
   - 演示多点选择的效果
   - 显示3个mask候选和分数

4. **INTERACTIVE_SELECTION.md**
   - 更新使用说明
   - 添加多点和排除点的文档

## 🎨 可视化增强

### 选择界面
- 实时显示统计：`Foreground: 3 | Background: 1`
- 清晰的颜色区分：绿色(前景) vs 红色(背景)
- 编号标记：便于追踪每个点

### 结果图像
第一帧保存的图像显示：
- **绿色圆圈**: 所有前景点
- **红色X**: 所有背景点
- **青色星号**: 计算出的锚点（mask质心）

## 💡 使用建议

### 最佳实践
1. **从少到多**: 先试试单点，不行再加点
2. **关键位置**: 前景点选择物体的关键特征位置
3. **明确排除**: 只在物体和背景难以区分时使用排除点
4. **及时撤销**: 点错了立即按U键撤销

### 常见问题

**Q: 需要选几个前景点？**
A: 至少1个。简单物体1-2个，复杂物体3-5个。

**Q: 什么时候需要排除点？**
A: 当分割结果包含了不想要的背景区域时。

**Q: 可以先选排除点再选前景点吗？**
A: 可以，顺序无所谓。但必须至少有1个前景点。

**Q: 撤销是按什么顺序？**
A: 先撤销背景点，背景点撤完再撤销前景点（后添加先撤销）。

## 🔬 效果示例

### 单点 vs 多点
```
场景: 分割一棵形状不规则的树

单点模式:
  点击: 树干中心
  结果: 可能只分割到树干，树叶丢失

多点模式:
  点击: 树干 + 树冠左侧 + 树冠右侧
  结果: 完整分割整棵树
```

### 无排除点 vs 有排除点
```
场景: 树在草地上（都是绿色）

无排除点:
  点击: 树干和树叶
  结果: 可能包含部分草地

有排除点:
  点击: 树干和树叶（前景）+ 草地（背景）
  结果: 只分割树，排除草地
```

## 📚 相关文档
- [INTERACTIVE_SELECTION.md](INTERACTIVE_SELECTION.md) - 详细使用指南
- [CENTROID_TRACKING.md](CENTROID_TRACKING.md) - 质心追踪说明

## 🎉 试试看！
```bash
cd /home/jonas/Code/gsplat
python examples/propagate_segmentation.py
```

选择多个点，体验更精确的分割效果！
