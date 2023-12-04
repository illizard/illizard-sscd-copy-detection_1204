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

# 평균 패치 임베딩 어텐션 맵 계산 및 시각화
def calculate_and_visualize_average_attention_maps(image_path, model, save_path, set_name, highlighted_blocks_heads):
    all_attention_maps = []

    for img_file in glob.glob(os.path.join(image_path, "*.jpg")):
        img_tensor = preprocess_image(img_file)
        attention_maps = get_attention_maps(model, img_tensor)

        if len(all_attention_maps) == 0:
            all_attention_maps = [[attn_map.clone() for attn_map in block] for block in attention_maps]
        else:
            for block_index, block_maps in enumerate(attention_maps):
                for head_index, head_map in enumerate(block_maps):
                    all_attention_maps[block_index][head_index] += head_map

    num_images = len(glob.glob(os.path.join(image_path, "*.jpg")))
    average_attention_maps = [[head_map / num_images for head_map in block] for block in all_attention_maps]

    visualize_attention_maps(average_attention_maps, "average", save_path, set_name, highlighted_blocks_heads)



# 평균 패치 임베딩 어텐션 맵 시각화
def visualize_attention_maps(attention_maps, image_id, save_path, set_name, highlighted_blocks_heads):
    os.makedirs(save_path, exist_ok=True)
    
    num_blocks = len(attention_maps)
    num_heads = len(attention_maps[0])

    fig, axes = plt.subplots(num_heads, num_blocks, figsize=(num_blocks * 2, num_heads * 2))
    for i in range(num_blocks):
        for j in range(num_heads):
            block_head_attention = attention_maps[i][j]  # Get the attention map for the current block and head
            if block_head_attention.dim() == 2:  # Check if it's a 2D tensor
                patch_attn = block_head_attention  # Use it directly
            else:
                # If it's not 2D, calculate the average attention map for the patch
                patch_attn = block_head_attention[0, 1:, 1:].mean(dim=0).reshape((14, 14))
                
            patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min())
            axes[j, i].imshow(patch_attn, cmap='viridis')
            axes[j, i].axis('off')
            
            if (i, j) in highlighted_blocks_heads:
                rect = patches.Rectangle((-0.5, -0.5), 14, 14, linewidth=5, edgecolor='r', facecolor='none')
                axes[j, i].add_patch(rect)

    plt.suptitle(f"Patch Attention Maps for Image ID {set_name}_{image_id}")
    plt.savefig(os.path.join(save_path, f"{set_name}_{image_id}_patch_attention_maps.png"))
    plt.close(fig)


# 메인 함수
def main():
    # checkpoint_path = '/hdd/wi/sscd-copy-detection/ckpt/exp_head/ogloss_tr20k[model_3]_OFFL_VIT_TINY_last_False_0.0_cls+pat_192_1115_081737/c0x32x74/checkpoints/epoch=49-step=7799.ckpt'
    checkpoint_path = '/hdd/wi/sscd-copy-detection/ckpt/exp_ortho4/contl_xent_ortho_tr20k[model_3]_OFFL_VIT_TINY_last_True_0.0_cls+pat_192_1115_235602/rq0uncxo/checkpoints/epoch=49-step=7799.ckpt'
    query_path = '/hdd/wi/dataset/DISC2021_mini/queries/images/queries/'
    ref_path = '/hdd/wi/dataset/DISC2021_mini/references/images/references/'
    save_path = '/hdd/wi/sscd-copy-detection/result/pat_attn_maps_grid_mean'  # 결과 저장 경로

    model = load_model(checkpoint_path)
    sorted_dict = {(9, 2): 0.65145711, (7, 1): 0.651187705, (8, 2): 0.649433325, (7, 0): 0.649101942, (6, 2): 0.649025248, (4, 0): 0.647900028, (9, 1): 0.64716865}

    highlighted_blocks_heads = list(sorted_dict.keys())  # 특정 블록과 헤드 인덱스
    # process_images(query_path, model, save_path, "Query", highlighted_blocks_heads)
    # process_images(ref_path, model, save_path, "Reference", highlighted_blocks_heads)

    calculate_and_visualize_average_attention_maps(query_path, model, save_path, "Query", highlighted_blocks_heads)
    calculate_and_visualize_average_attention_maps(ref_path, model, save_path, "Reference", highlighted_blocks_heads)

if __name__ == "__main__":
    main()
