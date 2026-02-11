#!/usr/bin/env python3
"""
检查训练图像和 mask 文件的对齐关系
"""

import os
import sys
from pathlib import Path
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.datasets.colmap import Parser

def check_alignment(data_dir: str, data_factor: int = 2, mask_type: str = "Sam2"):
    """检查图像和 mask 的对应关系"""
    
    print("=" * 80)
    print("检查图像和 Mask 对齐关系")
    print("=" * 80)
    
    # 初始化 Parser
    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=True,
        test_every=8
    )
    
    print(f"\n数据目录: {data_dir}")
    print(f"缩放因子: {data_factor}")
    print(f"总图像数: {len(parser.image_names)}")
    print(f"Test every: {parser.test_every}")
    
    # 获取训练集索引
    indices = np.arange(len(parser.image_names))
    train_indices = indices[indices % parser.test_every != 0]
    test_indices = indices[indices % parser.test_every == 0]
    
    print(f"\n训练集图像数: {len(train_indices)}")
    print(f"测试集图像数: {len(test_indices)}")
    
    # Mask 目录
    if data_factor == 1:
        mask_dir = Path(data_dir) / "mask" / mask_type / "images"
    else:
        mask_dir = Path(data_dir) / "mask" / mask_type / f"images_{data_factor}"
    
    print(f"\nMask 目录: {mask_dir}")
    print(f"目录存在: {mask_dir.exists()}")
    
    if not mask_dir.exists():
        print(f"\n❌ Mask 目录不存在！")
        return
    
    # 列出所有 mask 文件
    mask_files = sorted(mask_dir.glob("*.npy"))
    print(f"\nMask 文件数: {len(mask_files)}")
    
    # 检查对齐
    print("\n" + "=" * 80)
    print("图像 ID → 图像名称 → Mask 文件对应关系")
    print("=" * 80)
    print(f"{'ID':<6} {'训练/测试':<8} {'图像名称':<30} {'Mask文件':<30} {'状态':<10}")
    print("-" * 80)
    
    missing_masks = []
    matched_masks = []
    
    for idx in range(len(parser.image_names)):
        image_name = parser.image_names[idx]
        base_name = Path(image_name).stem
        is_train = idx % parser.test_every != 0
        split = "训练" if is_train else "测试"
        
        # 检查 mask 文件
        mask_path = mask_dir / f"{base_name}.npy"
        
        if mask_path.exists():
            status = "✓ 存在"
            if is_train:
                matched_masks.append((idx, image_name, mask_path.name))
        else:
            status = "✗ 缺失"
            if is_train:
                missing_masks.append((idx, image_name))
        
        # 只打印前 15 个和缺失的
        if idx < 15 or not mask_path.exists():
            print(f"{idx:<6} {split:<8} {image_name:<30} {base_name}.npy{'':<15} {status:<10}")
    
    if len(parser.image_names) > 15:
        print(f"... (省略 {len(parser.image_names) - 15} 行)")
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"训练集图像: {len(train_indices)} 个")
    print(f"  - 有对应 mask: {len(matched_masks)} 个")
    print(f"  - 缺失 mask: {len(missing_masks)} 个")
    
    if missing_masks:
        print(f"\n❌ 缺失的训练 mask:")
        for idx, img_name in missing_masks[:10]:
            print(f"  - ID {idx}: {img_name}")
        if len(missing_masks) > 10:
            print(f"  ... 还有 {len(missing_masks) - 10} 个")
    
    # 检查命名模式
    print("\n" + "=" * 80)
    print("命名模式分析")
    print("=" * 80)
    
    if len(mask_files) > 0:
        print(f"\n前 10 个 mask 文件:")
        for mf in mask_files[:10]:
            print(f"  - {mf.name}")
        
        print(f"\n前 10 个图像文件:")
        for idx in range(min(10, len(parser.image_names))):
            print(f"  - ID {idx}: {parser.image_names[idx]}")
    
    # 检查是否是编号不匹配
    print("\n" + "=" * 80)
    print("可能的问题诊断")
    print("=" * 80)
    
    # 提取 mask 文件的编号
    mask_numbers = []
    for mf in mask_files:
        stem = mf.stem
        # 尝试提取数字
        import re
        numbers = re.findall(r'\d+', stem)
        if numbers:
            mask_numbers.append(int(numbers[0]))
    
    if mask_numbers:
        print(f"\nMask 文件编号范围: {min(mask_numbers)} - {max(mask_numbers)}")
        print(f"Mask 文件数量: {len(mask_numbers)}")
    
    # 检查图像文件名的编号
    image_numbers = []
    for img_name in parser.image_names:
        import re
        numbers = re.findall(r'\d+', Path(img_name).stem)
        if numbers:
            image_numbers.append(int(numbers[0]))
    
    if image_numbers:
        print(f"\n图像文件编号范围: {min(image_numbers)} - {max(image_numbers)}")
        print(f"图像文件数量: {len(image_numbers)}")
    
    # 诊断结论
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    
    if len(missing_masks) == 0:
        print("✅ 所有训练图像都有对应的 mask 文件")
        print("✅ 图像名称和 mask 文件名完全匹配")
        print("\n建议:")
        print("  - 问题可能在渲染时的相机参数对齐")
        print("  - 检查 camtoworlds 是否使用了正确的 image_id")
    elif len(matched_masks) > 0:
        print(f"⚠️  部分训练图像缺失 mask ({len(missing_masks)}/{len(train_indices)})")
        print("\n建议:")
        print("  - 重新生成缺失的 mask 文件")
        print("  - 或检查 mask 文件命名是否正确")
    else:
        print("❌ 所有训练图像都缺失 mask")
        print("\n可能原因:")
        print("  1. Mask 目录路径错误")
        print("  2. 图像文件名和 mask 文件名格式不匹配")
        print("  3. Mask 文件扩展名不是 .npy")
    
    # 显示实际使用的 image_id 示例
    print("\n" + "=" * 80)
    print("训练时的 image_id 示例")
    print("=" * 80)
    print("\n在训练循环中，image_id 的取值:")
    for i, idx in enumerate(train_indices[:10]):
        img_name = parser.image_names[idx]
        base_name = Path(img_name).stem
        print(f"  Step {i}: image_id={idx} → 图像={img_name} → mask={base_name}.npy")
    
    if len(train_indices) > 10:
        print(f"  ... (还有 {len(train_indices) - 10} 个训练图像)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Tree", help="数据目录")
    parser.add_argument("--data_factor", type=int, default=2, help="图像缩放因子")
    parser.add_argument("--mask_type", type=str, default="Sam2", help="Mask 类型")
    
    args = parser.parse_args()
    
    check_alignment(args.data_dir, args.data_factor, args.mask_type)
