# 前景遮罩训练模式

## 功能说明

这个功能允许你使用已经训练好的高斯数据和带有绿色前景标记的图片进行二次训练。训练过程中，当检测到高斯球落在图片的前景区域（绿色标记区域）时，会将该高斯球的颜色强制设置为黑色。

## 使用步骤

### 1. 准备数据

确保你有：
- **已训练的高斯模型**：例如 `results/Tree/ckpts/ckpt_29999_rank0.pt`
- **带绿色标记的图片**：在 `data/Tree_Filled/images/` 目录下，前景区域用纯绿色标记

### 2. 配置参数

在 `OR_trainer.py` 的 `Config` 类中，有以下相关配置：

```python
foreground_mask_to_black: bool = False  # 启用前景遮罩模式
green_threshold_lower: Tuple[float, float, float] = (0.0, 0.5, 0.0)  # 绿色下界 RGB
green_threshold_upper: Tuple[float, float, float] = (0.5, 1.0, 0.5)  # 绿色上界 RGB
```

### 3. 运行训练

#### 方法 1：使用提供的脚本

```bash
cd /home/jonas/Code/gsplat
./examples/run_foreground_masking.sh
```

#### 方法 2：自定义命令

```bash
python examples/OR_trainer.py \
    --data_dir data/Tree_Filled \
    --result_dir results/Tree_Masked \
    --data_factor 4 \
    --ckpt results/Tree/ckpts/ckpt_29999_rank0.pt \
    --foreground_mask_to_black True \
    --max_steps 5000 \
    --eval_steps 1000 5000 \
    --save_steps 1000 5000 \
    --disable_viewer \
    --disable_video
```

### 4. 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data_dir` | 带绿色标记图片的目录 | `data/Tree_Filled` |
| `--result_dir` | 结果保存目录 | `results/Tree_Masked` |
| `--ckpt` | 已训练的checkpoint路径 | `results/Tree/ckpts/ckpt_29999_rank0.pt` |
| `--foreground_mask_to_black` | 启用前景遮罩模式 | `True` |
| `--green_threshold_lower` | 绿色检测下界 | `0.0 0.5 0.0` |
| `--green_threshold_upper` | 绿色检测上界 | `0.5 1.0 0.5` |
| `--max_steps` | 训练步数 | `5000` |

### 5. 工作原理

1. **加载已训练模型**：从checkpoint加载高斯球参数（位置、缩放、旋转、颜色等）
2. **检测绿色前景**：从带标记的图片中，检测RGB值在绿色阈值范围内的像素
3. **遮罩高斯球**：
   - 在每个训练步，渲染图像时检测哪些高斯球落在绿色前景区域
   - 将这些高斯球的颜色参数强制设为黑色（接近0）
4. **继续训练**：模型会学习在保持前景为黑色的同时优化其他参数

### 6. 输出结果

训练完成后，在 `results/Tree_Masked/` 目录下会有：
- `ckpts/`：保存的checkpoint
- `renders/`：渲染的图像
- `stats/`：训练统计数据
- `tb/`：Tensorboard日志

### 7. 注意事项

1. **绿色检测阈值**：默认检测的绿色范围是 RGB 中 G通道在 0.5-1.0，R和B通道在 0.0-0.5。如果你的绿色标记不在这个范围，需要调整 `green_threshold_lower` 和 `green_threshold_upper`。

2. **Packed模式**：代码使用 packed 模式来高效追踪哪些高斯球落在前景。确保 `--packed True`（默认值）。

3. **checkpoint兼容性**：确保checkpoint文件与当前代码版本兼容。

### 8. 调试

如果绿色区域没有被正确检测，可以：

1. 打印检测到的绿色mask：
```python
# 在 _load_foreground_masks 函数中添加
import matplotlib.pyplot as plt
plt.imshow(green_mask.cpu().numpy())
plt.savefig(f"debug_mask_{image_id}.png")
```

2. 调整绿色阈值：
```bash
python examples/OR_trainer.py \
    --green_threshold_lower 0.0 0.6 0.0 \
    --green_threshold_upper 0.4 1.0 0.4 \
    ...
```

## 示例

假设原始训练结果在 `results/Tree/`，带绿色标记的图片在 `data/Tree_Filled/images/`：

```bash
# 运行前景遮罩训练
python examples/OR_trainer.py \
    --data_dir data/Tree_Filled \
    --result_dir results/Tree_Masked \
    --data_factor 4 \
    --ckpt results/Tree/ckpts/ckpt_29999_rank0.pt \
    --foreground_mask_to_black True \
    --max_steps 3000 \
    --eval_steps 1000 3000 \
    --save_steps 1000 3000
```

训练完成后，前景区域的高斯球颜色将变为黑色。
