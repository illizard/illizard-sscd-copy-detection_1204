### EXEC ### 
echo "MODEL 1"
MASTER_ADDR="localhost" MASTER_PORT="15081" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train_og_ft.py --nodes=1 --gpus=2 --batch_size=128 \
  --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/images/train_20k/ \
  --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
  --entropy_weight=30 --epochs=100 --augmentations=ADVANCED --mixup=true  \
  --output_path=/hdd/wi/sscd-copy-detection/result/test/ \
  --backbone=OFFL_VIT_TINY