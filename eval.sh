sscd/disc_eval_sscd_og.py --disc_path /hdd/wi/dataset/DISC2021_exp/  --gpus=2 \
  --output_path=./ \
  --size=288 --preserve_aspect_ratio=true \
  --backbone=CV_RESNET50 --dims=512 --model_state=./sscd_disc_mixup.classy.pt

# sscd/disc_eval.py --disc_path /hdd/wi/dataset/DISC2021/ --gpus=2 \
#   --output_path=./ \
#   --size=224 --preserve_aspect_ratio=true \
#   --workers=28 \
#   --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/og[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1128_184838/vlhyu7mu/checkpoints/epoch=99-step=15599.ckpt \