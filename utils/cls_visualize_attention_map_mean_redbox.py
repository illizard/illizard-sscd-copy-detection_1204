import os
import glob
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pdb
from make_file import make_dict_score_based, make_dict_similarity_based, make_dict_random_based
import argparse
import numpy as np

# ViT 모델 로드 및 체크포인트 로드
def load_model(checkpoint_path):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = {}

    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '').replace('backbone.', '')  # 모든 접두사 제거
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    return model

# 어텐션 맵 추출
def get_attention_maps(model, img_tensor):
    attention_maps = []

    def hook_fn(module, input, output):
        B, N, C = input[0].shape
        qkv = module.qkv(input[0]).detach().reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = module.qkv(input[0]).detach().reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4) # Reorder to get q, k, v
        q, k, _ = qkv[0], qkv[1], qkv[2]
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * module.scale
        attn_weights = attn_score.softmax(dim=-1)
        attention_maps.append(attn_weights.squeeze().detach())

    hooks = []
    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(img_tensor)

    for hook in hooks:
        hook.remove()

    return attention_maps

# 이미지 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

# 평균 어텐션 맵 계산 및 시각화
def calculate_and_visualize_average_attention_maps(args, image_path, model, set_name, highlighted_blocks_heads):
    all_attention_maps = []  # 모든 이미지의 어텐션 맵을 저장할 리스트

    # 이미지 경로에서 모든 이미지에 대해 어텐션 맵 계산
    for img_file in glob.glob(os.path.join(image_path, "*.jpg")):
        img_tensor = preprocess_image(img_file)
        attention_maps = get_attention_maps(model, img_tensor)

        if len(all_attention_maps) == 0:
            # 첫 번째 이미지의 어텐션 맵으로 리스트 초기화
            all_attention_maps = attention_maps
        else:
            # 이후 이미지의 어텐션 맵을 누적하여 더함
            for block_index, block_maps in enumerate(attention_maps):
                all_attention_maps[block_index] += block_maps

    # 평균을 계산하기 위해 이미지 수로 나눔
    average_attention_maps = [block_maps / len(glob.glob(os.path.join(image_path, "*.jpg")))
                              for block_maps in all_attention_maps]

    # 평균 어텐션 맵 시각화
    visualize_attention_maps(args, average_attention_maps, set_name, highlighted_blocks_heads)

# 개별 어텐션 맵 시각화 (특정 블록/헤드에 빨간색 테두리 추가)
def visualize_attention_maps(args, attention_maps, set_name, highlighted_blocks_heads):
    # Create the save_path directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    num_blocks = len(attention_maps)
    num_heads = attention_maps[0].size(0)

    fig, axes = plt.subplots(num_heads, num_blocks, figsize=(num_blocks * 2, num_heads * 2))
    for i in range(num_blocks):
        for j in range(num_heads):
            cls_attn = attention_maps[i][j][0, 1:]  # 첫 번째 토큰인 [CLS]는 제외
            attn_map = cls_attn.reshape((14, 14))  # (14, 14)로 재구성
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())  # 정규화
            axes[j, i].imshow(attn_map, cmap='viridis')
            axes[j, i].axis('off')
            
            # 특정 블록과 헤드에 빨간색 테두리 추가
            # if (i, j) in highlighted_blocks_heads:
            #     # 패치의 크기와 위치 조정
            #     rect = patches.Rectangle((-0.5, -0.5), 14, 14, linewidth=5, edgecolor='r', facecolor='none')
            #     axes[j, i].add_patch(rect)

    plt.suptitle(f"Attention Maps for Average {set_name}")
    ### check
    plt.savefig(os.path.join(args.save_path, f"{set_name}_attention_maps_redbox_10k_{args.based}_og_{args.head_drop_ratio}.png"))
    plt.close(fig)
    
# 이미지 경로에서 모든 이미지에 대해 어텐션 맵 처리 및 시각화
def process_images(image_path, model, save_path, set_name, highlighted_blocks_heads):
    for img_file in glob.glob(os.path.join(image_path, "*.jpg")):  # JPG 파일을 대상으로 함
        img_tensor = preprocess_image(img_file)
        attention_maps = get_attention_maps(model, img_tensor)
        # image_id = os.path.basename(img_file).split('.')[0]  # 파일 이름에서 이미지 ID 추출
        visualize_attention_maps(attention_maps, save_path, set_name, highlighted_blocks_heads)

def seed_everything(seed: int = 42):
    # random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

# 메인 함수
def main(args):
    # checkpoint_path = '/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/og[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1128_184838/vlhyu7mu/checkpoints/epoch=99-step=15599.ckpt'
    query_path = '/hdd/wi/dataset/DISC2021_exp/images/queries_100/'
    ref_path = '/hdd/wi/dataset/DISC2021_exp/images/references_100/'
    # save_path = '/hdd/wi/sscd-copy-detection/result/1130_real_final/images/'  # 결과 저장 경로

    model = load_model(args.checkpoint_path)
    if args.based == 'score':
        sorted_data_dict = make_dict_score_based(args.head_drop_ratio, args.csv_in_path)
    elif args.based == 'order':
        sorted_data_dict = make_dict_score_based(args.head_drop_ratio, args.csv_in_path)
        pass
    else:
        pass
    highlighted_blocks_heads = list(sorted_data_dict.keys())  # 특정 블록과 헤드 인덱스
    
    calculate_and_visualize_average_attention_maps(args, query_path, model, "Query", highlighted_blocks_heads)
    calculate_and_visualize_average_attention_maps(args, ref_path, model, "Reference", highlighted_blocks_heads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv_in_path', type=str, help='Path to input CSV file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to input checkpoint')
    parser.add_argument('--save_path', type=str, help='Path to output images')
    parser.add_argument('--head_drop_ratio', type=float, help='Ratio of head drop')
    parser.add_argument('--based', type=str, help='Pruning Mode score or similarity')
    args = parser.parse_args()
    
    seed_everything(42)
    main(args)