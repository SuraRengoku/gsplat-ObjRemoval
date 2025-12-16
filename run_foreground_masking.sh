#!/bin/bash

# 使用已训练的高斯数据和带绿色前景标记的图片进行训练
# 当高斯球落在前景区域时，将其颜色设为黑色

# config
TRAINED_CKPT="results/Tree/ckpts/ckpt_29999_rank0.pt"
DATA_DIR="data/Tree_Filled"
RESULT_DIR="results/Tree_Masked"
DATA_FACTOR=2

python examples/OR_trainer.py \
    --data_dir ${DATA_DIR} \
    --result_dir ${RESULT_DIR} \
    --data_factor ${DATA_FACTOR} \
    --ckpt ${TRAINED_CKPT} \
    --foreground_mask_to_black \
    --max_steps 5000 \
    --eval_steps 1000 5000 \
    --save_steps 1000 5000 \
    --disable_viewer \
    --disable_video

echo "训练完成！结果保存在 ${RESULT_DIR}"
