#!/bin/bash

# Iterate Drop ratio
# DROP_RATIO=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
DROP_RATIO=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for drop_ratio in "${DROP_RATIO[@]}"
do
    echo "Fine-tuning with pruning"
    echo "Reducing percent = $drop_ratio"
    MASTER_ADDR="localhost" MASTER_PORT="15085" NODE_RANK="0" WORLD_SIZE=2 \
    sscd/train_pr_ft.py --nodes=1 --gpus=2 --batch_size=128 \
    --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/images/train_20k/ \
    --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
    --entropy_weight=30 --epochs=30 --augmentations=ADVANCED --mixup=true  \
    --output_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/prft_order_from_first \
    --backbone=OFFL_VIT_TINY \
    --head_drop_ratio=$drop_ratio \
    --csv_in_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/file/ogft_order_pre.csv
done
