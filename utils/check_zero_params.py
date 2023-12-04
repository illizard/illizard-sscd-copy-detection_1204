import torch
import timm

#### 인자값 설정 ####
block_index = 10
head_index = 2

# 사전 훈련된 모델 불러오기 (예: 'resnet50')
model = timm.create_model("vit_tiny_patch16_224", num_classes=0, pretrained=False)

# 사용자의 체크포인트 불러오기
checkpoint_path = '/hdd/wi/sscd-copy-detection/ckpt/1125_final/score_ratio_0.1[model_3]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep30_1126_204438/iqfxtd73/checkpoints/epoch=4-step=779.ckpt'
# checkpoint_path_fp32 = '/hdd/wi/sscd-copy-detection/ckpt/exp_paper/og[model_3]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1119_121055/3xq6fqe3/checkpoints/epoch=99-step=15599.ckpt'
# checkpoint_path_int8 = '/hdd/wi/sscd-copy-detection/result/1118_mini/quantized_model.pth'
# checkpoint_path ='/hdd/wi/sscd-copy-detection/ckpt/exp_test/score_ratio_1.0[model_3]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep1_1126_180213/m4i9u2on/checkpoints/epoch=0-step=9.ckpt'

checkpoint = torch.load(checkpoint_path, map_location='cpu')
# checkpoint_32 = torch.load(checkpoint_path_fp32, map_location='cpu')
# checkpoint_8 = torch.load(checkpoint_path_int8, map_location='cpu')

# 체크포인트 키 수정
for key in list(checkpoint['state_dict'].keys()):
    new_key = key.replace("model.backbone.", "")
    checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)

# 수정된 체크포인트를 모델에 로드
model.load_state_dict(checkpoint['state_dict'])

# QKV 가중치와 바이어스 프루닝
# attn_module = model.blocks[block_index].attn
# qkv_weight = attn_module.qkv.weight
# qkv_bias = attn_module.qkv.bias

# # 가중치와 바이어스를 Query, Key, Value로 분할
# dim = qkv_weight.shape[0] // 3
# q_weight, k_weight, v_weight = qkv_weight[:dim], qkv_weight[dim:2*dim], qkv_weight[2*dim:]
# q_bias, k_bias, v_bias = qkv_bias[:dim], qkv_bias[dim:2*dim], qkv_bias[2*dim:]

# head_size = 64
# start_idx = head_index * head_size
# end_idx = (head_index + 1) * head_size

# qw = q_weight[start_idx:end_idx]
# kw = k_weight[start_idx:end_idx]
# vw = v_weight[start_idx:end_idx]
# qb = q_bias[start_idx:end_idx]
# kb = k_bias[start_idx:end_idx]
# kv = v_bias[start_idx:end_idx]

# 특정 인코더 블록 상태 확인 (예: 첫 번째 블록)
# encoder_block1 = model.blocks[block_index].attn.qkv.weight[]


print(f"q:\n{checkpoint['state_dict']['blocks.7.attn.qkv.weight'][64:128]}\n\n")
print(f"q:\n{checkpoint['state_dict']['blocks.7.attn.qkv.weight'][128:256]}\n\n")
print(f"k:\n{checkpoint['state_dict']['blocks.7.attn.qkv.weight'][256:320]}\n\n")
print(f"v:\n{checkpoint['state_dict']['blocks.7.attn.qkv.weight'][448:512]}\n\n")
#print(checkpoint['state_dict']['blocks.0.attn.qkv.weight'].shape)

# print(checkpoint['state_dict']['model.backbone.blocks.11.attn.qkv.bias'])
# print(checkpoint_32['state_dict']['model.backbone.blocks.11.attn.qkv.bias'])
# print(checkpoint_8['backbone.blocks.11.attn.qkv.bias'])
