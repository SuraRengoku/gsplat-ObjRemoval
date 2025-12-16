#!/bin/bash

# initial training to get original gaussian primitives and depth maps for each training image
# NOTE: in training part, gsplat will pick a eval image every 8 images in the input set.

# config 
DATA_DIR="data/Tree"
RESULT_DIR="results/Tree"
CHECK_POINT="results/Tree/ckpts/ckpt_29999_rank0.pt" 
DATA_FACTOR=2

# if there are already pt files in the target result fold, the initial training part is finished. 
# just use the pt file to obtain depth maps
# if there are nothing in the target result fold, we should first start training.

echo "Checking for checkpoint: $CHECK_POINT"

if [ -f "$CHECK_POINT" ]; then
    echo "Checkpoint found! Skipping training..."
    echo "Generating depth maps from existing checkpoint..."

    python examples/simple_trainer.py default \
        --data_dir ${DATA_DIR} \
        --result_dir ${RESULT_DIR} \
        --data_factor ${DATA_FACTOR} \
        --ckpt ${CHECK_POINT} \
        --save_train_depths \
        --disable_viewer
else
    python examples/simple_trainer.py default \
        --data_dir ${DATA_DIR} \
        --data_factor ${DATA_FACTOR} \
        --result_dir ${RESULT_DIR} \
        --save_train_depths \
        --save_ply
fi

