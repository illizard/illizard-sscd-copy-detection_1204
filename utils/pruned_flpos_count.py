import torch
import timm

from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_str, flop_count_table
from ptflops import get_model_complexity_info

@torch.no_grad()
def perform_head_pruning_and_cal_flops(sorted_data_dict):
    # model.eval()
    # 프루닝된 부분에 해당하는 가중치의 개수를 추적하기 위한 변수
    # pruned_weights_count = 0
    # for block_index, head_index in sorted_data_dict:

    #     # QKV 가중치와 바이어스 프루닝
    #     attn_module = model.backbone.blocks[block_index].attn
    #     qkv_weight = attn_module.qkv.weight()
    #     qkv_bias = attn_module.qkv.bias()
        
    #     # 가중치와 바이어스를 Query, Key, Value로 분할
    #     dim = qkv_weight.shape[0] // 3
    #     q_weight, k_weight, v_weight = qkv_weight[:dim], qkv_weight[dim:2*dim], qkv_weight[2*dim:]
    #     q_bias, k_bias, v_bias = qkv_bias[:dim], qkv_bias[dim:2*dim], qkv_bias[2*dim:]
       
    #     start_idx = head_index * head_size
    #     end_idx = (head_index + 1) * head_size
        
    #     q_weight[start_idx:end_idx] = 0
    #     k_weight[start_idx:end_idx] = 0
    #     v_weight[start_idx:end_idx] = 0
    #     q_bias[start_idx:end_idx] = 0
    #     k_bias[start_idx:end_idx] = 0
    #     v_bias[start_idx:end_idx] = 0

    
    return calculate_flops(sorted_data_dict)

def calculate_flops(sorted_data_dict):
    # 모델의 전체 FLOPs 계산
    model2 = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
    macs, _ = get_model_complexity_info(model2, (3, 224, 224), as_strings=False, print_per_layer_stat=False)

    # 프루닝 비율과 각 비율에서의 FLOPs 계산
    # sorted_data_dict는 (block_index, head_index)의 튜플로 이루어진 리스트
    # 개수 * 연산량 (9,682,944 MAC)
    flops_before_pruning = len(sorted_data_dict) * 9682944
    flops_after_pruning = macs - flops_before_pruning
    # GMAC으로 변환
    flops_after_pruning = flops_after_pruning / 1e9
    
    return flops_after_pruning
    
# def calculate_flops_2(model):
#     total_flops = 0

#     for layer in model.modules():
#         if isinstance(layer, torch.nn.Linear):
#             # Calculate FLOPs for Linear layer (used in Attention layers)
#             active_elements_count = torch.count_nonzero(layer.weight.data)
#             flops = active_elements_count * layer.in_features
#             total_flops += flops
#     # FLOPs 값을 정수형으로 변환
#     total_flops = total_flops.item()

    # MFLOPs와 GFLOPs로 변환
    # mflops = round(total_flops / 1e6, 2)
    # gflops = round(total_flops / 1e9, 2)

    
    return total_flops

def calculate_effective_flops(model):
    
    # 전체 FLOPs 계산
    input = torch.randn(1, 3, 224, 224)  # 예시 입력
    flops = FlopCountAnalysis(model, input)
    total_flops_1 = flops.total()
    total_flops = 0
    total_inac_flops = 0
    # 0인 가중치의 FLOPs 계산
    
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            # Calculate FLOPs for Linear layer (used in Attention layers)
            active_elements_count = torch.count_nonzero(layer.weight.data)
            addandmul = 2 * active_elements_count 
            flops = addandmul * layer.in_features
            total_flops += flops
            
            inactive_weights_count = torch.sum(layer.weight.data == 0)
            inac_addandmul = 2 * inactive_weights_count
            inac_flops = inac_addandmul * layer.in_features
            total_inac_flops += inac_flops

    # FLOPs 값을 정수형으로 변환
    # total_flops = total_flops.item()
    total_flops = total_flops_1 - total_inac_flops.item()
    
    return total_flops


   
    
# if __name__ == "__main__":
#     model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
#     head_size = 64
#     sorted_data_dict = make_dict(0.1,'/hdd/wi/sscd-copy-detection/result/cleaned_head_cutting_metrics.csv')
#     flops_before_pruning = calculate_flops(model)
#     print("FLOPs before pruning:", flops_before_pruning)

#     perform_head_pruning_and_cal_flops(model, sorted_data_dict, head_size)
    
#     flops_after_pruning = calculate_flops(model)
#     print("FLOPs after pruning:", flops_after_pruning)
#     print(sorted_data_dict.keys())

#     main()



