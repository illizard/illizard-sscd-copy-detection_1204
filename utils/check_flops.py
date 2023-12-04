import torch
import timm
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_str, flop_count_table
from ptflops import get_model_complexity_info

def check_and_compare_flops():
    # 모델 불러오기
    # model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0, num_heads=3)
    model = timm.create_model("mobilevit_xxs", pretrained=True, num_classes=1000)

    model.eval()

    # fvcore를 사용한 FLOPs 계산
    input = torch.randn(1, 3, 224, 224)
    fvcore_flops = FlopCountAnalysis(model, input)
    print("FLOPs (fvcore):", fvcore_flops.total())

    # ptflops를 사용한 FLOPs 계산
    ptflops_macs, _ = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False, verbose=False)
    ptflops_flops = 2 * ptflops_macs  # MACs를 FLOPs로 변환 (대략적으로 MACs * 2)
    print("FLOPs (ptflops):", ptflops_flops)

    # fvcore와 ptflops 결과 비교
    print("\nComparison:")
    print(f"fvcore: {fvcore_flops.total():.2f} FLOPs")
    print(f"ptflops: {ptflops_flops:.2f} FLOPs")
    
def check():

    # 모델 불러오기
    # model = timm.create_model("vit_vit_patch16_224", pretrained=True, num_classes=0, num_heads=3)
    model = timm.create_model("mobilevit_xxs", pretrained=True, num_classes=1000)

    model.eval()
    
    # 더미 입력 데이터
    input = torch.randn(1, 3, 224, 224)

    # 원본 모델의 FLOPs 계산
    original_flops = FlopCountAnalysis(model, input)
    print('Original FLOPs:', original_flops.total())
    
    # FLOPs 문자열 및 테이블 출력
    flops_str = flop_count_str(original_flops)
    flops_table = flop_count_table(original_flops)

    print("FLOPs Summary:\n", flops_str)
    print("FLOPs by Layer:\n", flops_table)
    
    # 특정 어텐션 헤드의 가중치를 0으로 설정
    # head_idx = 0  # 제거하려는 헤드의 인덱스
    # num_heads = model.blocks[0].attn.num_heads
    # head_dim = model.blocks[0].attn.head_dim

    # with torch.no_grad():
    #     # model.blocks[0].attn.qkv.weight[head_idx * head_dim:(head_idx + 1) * head_dim, :].zero_()
    #     # model.blocks[0].attn.qkv.bias[head_idx * head_dim:(head_idx + 1) * head_dim].zero_()
    #     print(f"model.blocks[0].attn.qkv.weight shape is {model.blocks[0].attn.qkv.weight.shape}")
    # # 변경된 모델의 FLOPs 계산
    # modified_flops = FlopCountAnalysis(model, input)
    
    ## 결과 출력
    # print('Modified FLOPs:', modified_flops.total())
    # print("FLOPs:\n", flop_count_str(modified_flops))
    # print("\nFLOPs:\n", flop_count_table(modified_flops))
    
    # 헤드가 전체 FLOPs에 미치는 영향 추정
    # estimated_effect = original_flops / num_heads
    # estimated_modified_flops = original_flops - estimated_effect
    # # print('Estimated Modified FLOPs:', estimated_modified_flops)
    
if __name__ == "__main__":
    # check()
    check()
# attention flops: 21.79M