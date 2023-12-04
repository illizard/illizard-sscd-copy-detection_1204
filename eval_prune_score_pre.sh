#!/bin/bash

# Block 인덱스 값의 배열 정의
BLOCK_IDX=(0 1 2 3 4 5 6 7 8 9 10 11)
# Head 인덱스 값의 배열 정의
HEAD_IDX=(0 1 2)

# Block 인덱스에 대해 반복
for current_block in "${BLOCK_IDX[@]}"
do
    # Head 인덱스에 대해 반복
    for current_head in "${HEAD_IDX[@]}"
    do
        echo "Reducing with block = $current_block and head = $current_head"
        sscd/disc_eval_prune_score_pre.py --disc_path /hdd/wi/dataset/DISC2021_exp/ --gpus=2 \
        --output_path=./ \
        --output_csv_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/file \
        --size=224 --preserve_aspect_ratio=true \
        --workers=28 \
        --block_idx=$current_block --head_idx=$current_head \
        --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/og[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1128_184838/vlhyu7mu/checkpoints/epoch=99-step=15599.ckpt \
    
    done
done
