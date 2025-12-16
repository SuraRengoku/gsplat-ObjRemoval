#!/bin/bash

# two stage training scripts - both with foreground masks
# 
# stage 1：training 7000 iterations with foreground masks, using gaussian primitives from initial training
# stage 2：training to 30000 iterations with foreground masks, using gaussian primitives from stage 1

# config
DATA_DIR="data/Tree_Filled"
STAGE1_DIR="results/Tree_Stage1"
STAGE2_DIR="results/Tree_Stage2"
DATA_FACTOR=2

# ==================== first stage ====================
echo "======================================"
echo "first stage running (0 -> 7000 steps)"
echo "training from scratch with foreground mask"
echo "======================================"

python examples/OR_trainer.py \
  --data_dir ${DATA_DIR} \
  --result_dir ${STAGE1_DIR} \
  --data_factor ${DATA_FACTOR} \
  --ckpt results/Tree/ckpts/ckpt_29999_rank0.pt \
  --foreground_mask_to_black \
  --max_steps 7000 \
  --eval_steps 3500 7000 \
  --save_steps 3500 7000 \
  --ply_steps 7000 \
  --save_ply \
  --disable_video

# check state
if [ $? -ne 0 ]; then
    echo "first stage failed"
    exit 1
fi

echo "======================================"
echo "first stage completed"
echo "Checkpoint saved to: ${STAGE1_DIR}/ckpts/ckpt_6999_rank0.pt"
echo "======================================"
echo ""

# ==================== second stage ====================
echo "======================================"
echo "second stage running (continue to 30000 steps)"
echo "training from first stage's checkpoint with foreground mask"
echo "======================================"

python examples/OR_trainer.py \
  --data_dir ${DATA_DIR} \
  --result_dir ${STAGE2_DIR} \
  --data_factor ${DATA_FACTOR} \
  --ckpt ${STAGE1_DIR}/ckpts/ckpt_6999_rank0.pt \
  --foreground_mask_to_black \
  --max_steps 30000 \
  --eval_steps 15000 30000 \
  --save_steps 15000 30000 \
  --ply_steps 30000 \
  --save_ply \
  --disable_video

# check state
if [ $? -ne 0 ]; then
    echo "second stage failed"
    exit 1
fi

echo "======================================"
echo "two stages completed"
echo "first stage result: ${STAGE1_DIR}"
echo "second stage result: ${STAGE2_DIR}"
echo "======================================"
