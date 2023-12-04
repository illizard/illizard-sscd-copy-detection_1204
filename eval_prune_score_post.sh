#!/bin/bash
# disc_eval_head_ab_post의 csv_out_path == make_file의 csv_in_path와 동일해야 함
# Iterate Drop ratio
DROP_RATIO=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# DROP_RATIO=(0.3)

# for drop_ratio in "${DROP_RATIO[@]}"
# do
#     echo "Reducing percent = $drop_ratio"
#     sscd/disc_eval_head_ab_post.py --disc_path /hdd/wi/dataset/DISC2021_exp/ --gpus=2 \
#     --output_path=./ \
#     --csv_in_path=/hdd/wi/sscd-copy-detection/result/cleaned_head_cutting_metrics.csv \
#     --csv_out_path=/hdd/wi/sscd-copy-detection/result/head_cutting_ratio_vit_tiny_sim_based.csv \
#     --head_drop_ratio=$drop_ratio \
#     --size=224 --preserve_aspect_ratio=true \
#     --workers=0 \
#     --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/ckpt/[og100][model_3]_OFFL_VIT_TINY_last_false_cls+pat_384_1107_062800/4ja5ssy0/checkpoints/epoch=99-step=19499.ckpt
# done

#########################################################
# score based
#########################################################
# 지금 집가서 돌리기 , 근데 먼저 make file에서 고쳐야함 어센딩 -> 디센딩 (헤드를 앞에서부터 지워보기)
for drop_ratio in "${DROP_RATIO[@]}"
do
    echo "SCORE BASED"
    echo "Reducing percent = $drop_ratio"
    sscd/disc_eval_prune_score_post.py --disc_path /hdd/wi/dataset/DISC2021_exp/ --gpus=2 \
    --output_path=./ \
    --csv_in_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/file/ogft_order_pre.csv \
    --csv_out_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/file/ogft_order_post.csv \
    --head_drop_ratio=$drop_ratio \
    --size=224 --preserve_aspect_ratio=true \
    --workers=28 \
    --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/og[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1128_184838/vlhyu7mu/checkpoints/epoch=99-step=15599.ckpt \
    --based=score
done

echo "Make histogram"
python3 utils/make_file.py --csv_in_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/file/ogft_order_post.csv \
    --hist_out_path=/hdd/wi/sscd-copy-detection/result/1130_real_final/images \
    --based=score

#########################################################
# similarity based
# 실행하기 전에 utils/cosine_similarity_all_blocks.py를 돌려서 유사도별 블록, 헤드가 있는 csv 파일을 만들어야 함 
# csv_in_path를 바꿔줘야 함
#########################################################

# for drop_ratio in "${DROP_RATIO[@]}"
# do
#     echo "SIMILARITY BASED"
#     echo "Reducing percent = $drop_ratio"
#     sscd/disc_eval_prune_score_post.py --disc_path /hdd/wi/dataset/DISC2021_exp/ --gpus=2 \
#     --output_path=./ \
#     --csv_in_path=/hdd/wi/sscd-copy-detection/result/1118_evalset/sim_pre_metrics.csv \
#     --csv_out_path=/hdd/wi/sscd-copy-detection/result/1118_mini/sim_post_metrics_test.csv \
#     --head_drop_ratio=$drop_ratio \
#     --size=224 --preserve_aspect_ratio=true \
#     --workers=0 \
#     --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/ckpt/exp_paper/og[model_3]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1119_121055/3xq6fqe3/checkpoints/epoch=99-step=15599.ckpt \
#     --based=similarity
# done

# echo "Make histogram"
# python3 utils/make_file.py --csv_in_path=/hdd/wi/sscd-copy-detection/result/1118_mini/sim_post_metrics_test.csv \
#     --hist_out_path=/hdd/wi/sscd-copy-detection/result/1118_mini/ \
#     --based=similarity

#########################################################
# random based
#########################################################

# for drop_ratio in "${DROP_RATIO[@]}"
# do
#     echo "RANDOM BASED"
#     echo "Reducing percent = $drop_ratio"
#     sscd/disc_eval_prune_scosim_post.py --disc_path /hdd/wi/dataset/DISC2021_exp/ --gpus=2 \
#     --output_path=./ \
#     --csv_in_path=/hdd/wi/sscd-copy-detection/result/1118_evalset/sim_pre_metrics.csv \
#     --csv_out_path=/hdd/wi/sscd-copy-detection/result/1118_evalset/random_post_metrics.csv \
#     --head_drop_ratio=$drop_ratio \
#     --size=224 --preserve_aspect_ratio=true \
#     --workers=0 \
#     --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/ckpt/exp_paper/og[model_3]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1119_121055/3xq6fqe3/checkpoints/epoch=99-step=15599.ckpt \
#     --based=random
# done

# echo "Make histogram"
# python3 utils/make_file.py --csv_in_path=/hdd/wi/sscd-copy-detection/result/1118_evalset/random_post_metrics.csv \
#     --hist_out_path=/hdd/wi/sscd-copy-detection/result/1118_evalset/ \
#     --based=random