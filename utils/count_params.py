import torch
import timm
from ptflops import get_model_complexity_info


def count_params(input_tensor, model):

    # 모델의 FLOPs와 파라미터 수를 계산합니다.
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)

    print(f"모델의 FLOPs: {macs}")
    print(f"모델의 파라미터 수: {params}")

    return macs, params

if __name__ == "__main__":

    dummy = torch.randn(1, 3, 224, 224)
    # model = timm.create_model("resnet50", num_classes=0, pretrained=True)
    model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0)
    
    macs, params = count_params(dummy, model)