# 交互式多点选择使用指南

## 功能说明

现在支持在第一张图片上选择**多个前景点**和**多个背景点**（排除点），以获得更精确的SAM分割效果。

## 使用方法

### 1. 启用交互式选择（默认）

直接运行脚本：

```bash
python examples/propagate_segmentation.py
```

### 2. 操作步骤

1. **等待图像窗口打开**
   - 会自动打开第一张图片
   - 图片上方会显示使用说明

2. **选择前景点（物体）**
   - 用鼠标**左键点击**物体上的多个位置
   - 每次点击会显示**绿色圆圈**和编号（+1, +2, +3...）
   - 可以点击多个位置来确保完整分割物体

3. **选择背景点（排除区域）**
   - 用鼠标**右键点击**你想排除的区域
   - 每次点击会显示**红色X**和编号（-1, -2, -3...）
   - 用于排除不想要的部分

4. **调整选择（可选）**
   - 按 **U** 键撤销最后一个点
   - 按 **C** 键清空所有点，重新开始

5. **确认选择**
   - 按 **ENTER** 键确认选择
   - 程序会继续执行分割传播

6. **取消操作**
   - 按 **ESC** 键取消并退出

### 3. 禁用交互式选择

如果想使用手动坐标，修改 `propagate_segmentation.py`：

```python
# 在 __main__ 中设置
USE_INTERACTIVE = False  # 改为 False

# 选项1: 使用单点（向后兼容）
START_POINT_MANUAL = (480, 270)
POSITIVE_POINTS_MANUAL = None
NEGATIVE_POINTS_MANUAL = None

# 选项2: 使用多点
POSITIVE_POINTS_MANUAL = [(480, 270), (500, 300), (450, 280)]  # 前景点
NEGATIVE_POINTS_MANUAL = [(200, 150), (600, 400)]  # 背景点
```

## 示例输出

```
============================================================
MULTI-POINT INTERACTIVE SELECTION
============================================================
Instructions:
  LEFT CLICK  - Add foreground point (object)
  RIGHT CLICK - Add background point (exclude)
  U key       - Undo last point
  C key       - Clear all points
  ENTER       - Confirm selection
  ESC         - Cancel
============================================================
Added foreground point #1: (480, 270)
Added foreground point #2: (510, 285)
Added background point #1: (200, 150)
============================================================
Foreground points: 2
  +1: (480, 270)
  +2: (510, 285)
Background points: 1
  -1: (200, 150)
============================================================
Using 2 foreground points and 1 background points
```

## 可视化说明

### 选择界面

- **绿色圆圈 (●)**: 前景点（物体）
- **红色X (✕)**: 背景点（排除）
- **编号**: +1, +2... 表示前景，-1, -2... 表示背景
- **顶部统计**: 显示当前选择的点数量

### 结果图像

第一帧保存的图像会显示：
- **绿色圆圈**: 所有前景点
- **红色X**: 所有背景点（如果有）
- **青色星号 (★)**: 计算出的锚点（mask质心）

## 快捷键总结

| 键 | 功能 |
|---|---|
| **鼠标左键** | 添加前景点（物体） |
| **鼠标右键** | 添加背景点（排除） |
| **U** | 撤销最后一个点 |
| **C** | 清空所有点 |
| **ENTER** | 确认选择 |
| **ESC** | 取消退出 |

## 使用技巧

### 1. 多点选择的优势

**单点**：
```
左键点击物体中心一次 → ENTER
```
- 简单快速
- 适合形状简单、对比明显的物体

**多点**：
```
左键点击物体多个位置 → ENTER
```
- 更准确的分割
- 适合复杂形状、颜色不均匀的物体
- 可以确保整个物体被包含

### 2. 排除点的使用

```
左键点击物体 → 右键点击不想要的区域 → ENTER
```

**适用场景**：
- 物体与背景颜色相似
- 需要排除阴影或反光
- 物体附近有干扰物

**示例**：
```
物体: 树 (绿色)
背景: 草地 (也是绿色)
操作: 左键点击树干、树叶 + 右键点击草地
结果: SAM会分割树，但排除草地
```

### 3. 推荐工作流程

```
1. 左键点击物体中心（至少1个前景点）
2. 如果分割不完整，再添加几个前景点
3. 如果包含了不想要的区域，添加背景点
4. 按U撤销错误的点
5. 按ENTER确认
```

## 技术细节

### 自动缩放

- 如果图片太大（>1200x800），会自动缩放到合适大小显示
- 但返回的坐标始终是**原始图片的坐标**

### 可视化

- **黄色实心圆**: 选中的点
- **红色边框**: 选中点的高亮
- **文本标签**: 显示坐标 `(x, y)`

### 兼容性

- 使用 OpenCV 的 `cv2.imshow` 和鼠标回调
- 需要图形界面支持（不支持纯命令行环境）
- 如果在SSH远程环境，需要X11转发

## 故障排除

### 图像窗口没有打开

**可能原因**: 
- 在远程SSH环境没有X11转发
- 没有图形界面

**解决方案**:
```python
USE_INTERACTIVE = False  # 使用手动模式
```

### 导入错误

如果看到：
```
Warning: interactive_point_selector not available. Using manual point selection.
```

**解决方案**: 
确保 `interactive_point_selector.py` 在同一目录

### 图片路径错误

如果看到：
```
Error: First image not found: data/Tree/images_2/000001.jpg
```

**解决方案**:
检查 `IMAGES_DIR` 和 `START_IMAGE` 设置

## 完整配置示例

```python
if __name__ == "__main__":
    # ===== 基本路径配置 =====
    COLMAP_PATH = "data/Tree/sparse/0"
    IMAGES_DIR = "data/Tree/images_2"
    DEPTHS_DIR = "results/Tree/train_depths"
    START_IMAGE = "000001.jpg"
    
    # ===== 交互式选择配置 =====
    USE_INTERACTIVE = True           # 启用交互式选择
    START_POINT_MANUAL = (480, 270)  # 手动备用坐标
    
    # ===== 输出路径 =====
    ANCHOR_CENTROID_PATH = "data/Tree_Marked_Centroid/images"
    
    # 程序会自动处理点选择...
```

## 快捷键总结

| 键 | 功能 |
|---|---|
| **鼠标左键** | 选择/更改点 |
| **ENTER** | 确认选择 |
| **ESC** | 取消退出 |

## 测试交互式选择器

单独测试选择器：

```bash
python examples/interactive_point_selector.py
```

这会打开测试图片让你练习点选择。
