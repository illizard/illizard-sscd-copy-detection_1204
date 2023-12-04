#!/bin/bash
# score
# Python 스크립트와 아규먼트 실행
python3 utils/cls_visualize_attention_map_mean_redbox.py \
    --csv_in_path "/hdd/wi/sscd-copy-detection/result/1130_real_final/file/ogft_score_pre.csv" \
    --checkpoint_path "/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/prft_score/cls_score_ratio_0.0[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep30_1130_133609/zomq6kb0/checkpoints/epoch=29-step=4679.ckpt" \
    --save_path "/hdd/wi/sscd-copy-detection/result/1130_real_final/images" \
    --head_drop_ratio 0.4 \
    --based "score"

# # order
# # Python 스크립트와 아규먼트 실행
# python3 utils/cls_visualize_attention_map_mean_redbox.py \
#     --csv_in_path "/hdd/wi/sscd-copy-detection/result/1130_real_final/file/ogft_order_pre.csv" \
#     --checkpoint_path "/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/prft_order_from_last/cls_orderback_ratio_0.3[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep30_1201_022035/wgi4agqs/checkpoints/epoch=29-step=4679.ckpt" \
#     --save_path "/hdd/wi/sscd-copy-detection/result/1130_real_final/images" \
#     --head_drop_ratio 0.3 \
#     --based "order"
