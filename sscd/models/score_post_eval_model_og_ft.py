# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#230804
from collections import OrderedDict 

import argparse
import enum
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d
from classy_vision.models import build_model
from .gem_pooling import GlobalGeMPool2d

import timm
import torch.nn.utils.prune as prune

from utils.make_file import make_dict_score_based, make_dict_similarity_based, make_dict_random_based

class Implementation(enum.Enum):
    CLASSY_VISION = enum.auto()
    TORCHVISION = enum.auto()
    TORCHVISION_ISC = enum.auto()
    OFFICIAL = enum.auto() ##230708##
    MOBILE = enum.auto() ##230708##
    MY = enum.auto() ##230925##
    
class Backbone(enum.Enum):
    CV_RESNET18 = ("resnet18", 512, Implementation.CLASSY_VISION)
    CV_RESNET50 = ("resnet50", 2048, Implementation.CLASSY_VISION)
    CV_RESNEXT101 = ("resnext101_32x4d", 2048, Implementation.CLASSY_VISION)

    TV_RESNET18 = (resnet18, 512, Implementation.TORCHVISION)
    TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
    TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)
    
    MULTI_RESNET50 = ("multigrain_resnet50", 2048, Implementation.TORCHVISION_ISC)

    OFFL_VIT = ('vit_patch_16_base', 768, Implementation.OFFICIAL)     ##230831#    
    OFFL_VIT_TINY = ('vit_patch_16_tiny', 192, Implementation.OFFICIAL)     ##230831#    
    OFFL_HYVIT_TINY = ('vit_tiny_r_s16_p8_224', 192, Implementation.OFFICIAL)

    OFFL_FAST_TINY_T8 = ('fastvit_t8', 192, Implementation.OFFICIAL)
    OFFL_FAST_TINY_T12 = ('fastvit_t12', 192, Implementation.OFFICIAL)
    OFFL_FAST_TINY_SA12 = ('fastvit_sa12', 192, Implementation.OFFICIAL)

    OFFL_DINO = ('dino_patch_16_base', 768, Implementation.OFFICIAL)     ##230708##
    OFFL_MAE = ('mae_patch_16_base', 768, Implementation.OFFICIAL)  
    OFFL_MOBVIT = ('mobilevit_xxs', 192, Implementation.MOBILE)
    
    MY_DTOP_VIT_192 = ('dtop_vit_tiny_192', 192, Implementation.MY)  
    MY_DTOP_VIT_384 = ('dtop_vit_tiny_384', 384, Implementation.MY)  
    MY_XCIT = ('xcit_retrievalv2_small_12_p16', 384, Implementation.MY)  


    def build(self, dims: int, head_drop_ratio: float, csv_in_path: str):
        impl = self.value[2]
        
        if impl == Implementation.OFFICIAL: #### modi 0722 ###
            if self.value[0] == "vit_patch_16_base":
                model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=False, num_classes=0)
                return model           

            elif self.value[0] == "vit_patch_16_tiny":
                
                sorted_data_dict = make_dict_score_based(head_drop_ratio, csv_in_path)
                
                model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=False, num_classes=0)
                #check
                state_dict = "/hdd/wi/sscd-copy-detection/result/1130_real_final/ckpt/og[model_1]_OFFL_VIT_TINY_last_no_0.0_cls_192_ep100_1128_184838/vlhyu7mu/checkpoints/epoch=99-step=15599.ckpt"
                state_tmp = torch.load(state_dict, map_location=torch.device("cpu"))
                new_state_dict = {key.replace('model.backbone.', ''): value for key, value in state_tmp['state_dict'].items()}
                model.load_state_dict(new_state_dict)
                
                head_size = 64   # 헤드당 디멘션 크기
  

                for block_idx, head_idx in sorted_data_dict:

                    # 특정 인코더 블록의 QKV 가중치와 편향 가져오기
                    qkv_weight = model.blocks[block_idx].attn.qkv.weight.data
                    qkv_bias = model.blocks[block_idx].attn.qkv.bias.data

                    #가중치와 바이어스를 Q, K, V로 분할
                    dim = qkv_weight.shape[0] // 3
                    q_weight, k_weight, v_weight = qkv_weight[:dim], qkv_weight[dim:2*dim], qkv_weight[2*dim:]
                    q_bias, k_bias, v_bias = qkv_bias[:dim], qkv_bias[dim:2*dim], qkv_bias[2*dim:]
                    
                    #특정 헤드의 가중치와 편향을 0으로 설정하고 그래디언트 업데이트 제외
                    start_index = head_idx * head_size
                    end_index = start_index + head_size        

                    for weight, bias in [(q_weight, q_bias), (k_weight, k_bias), (v_weight, v_bias)]:
                        # detach()를 호출하여 원본 텐서에서 분리
                        detached_weight = weight.detach()
                        detached_bias = bias.detach()

                        #분리된 텐서에 대해 변형 수행
                        detached_weight[start_index:end_index, :].zero_()
                        detached_bias[start_index:end_index].zero_()

                        #원본 텐서에 분리된 텐서의 값을 다시 할당
                        weight[start_index:end_index, :] = detached_weight[start_index:end_index, :]
                        bias[start_index:end_index] = detached_bias[start_index:end_index]
                        # weight[start_index:end_index, :].zero_()
                        # bias[start_index:end_index].zero_()
                        # 그래디언트 업데이트 중지
                        weight[start_index:end_index, :].requires_grad = False
                        bias[start_index:end_index].requires_grad = False
                        
                        weight[start_index:end_index, :].requires_grad_(False)
                        bias[start_index:end_index].requires_grad_(False)
                        
                    
                return model

        else:
            raise AssertionError("Unsupported OFFICIAL model: %s" % (self.value[0]))
        

    
class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)


class Model(nn.Module):
    # def __init__(self, backbone: str, dims: int, pool_param: float): # og
    def __init__(self, backbone: str, dims: int, pool_param: float, head_drop_ratio: float, csv_in_path: str): # modii
        super().__init__()
        self.backbone_type = Backbone[backbone] # self.backbone_type = <Backbone.CV_RESNET50>
                                                #<Backbone.CV_RESNET50: ('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)>
                                                #Backbone <enum 'Backbone'> // 'CV_RESNET50'
        # MODI
        # print(f"backbone is {backbone}")
        self.dims = self.backbone_type.value[1]
        # self.block_idx = block_idx
        # self.head_idx = head_idx
        self.head_drop_ratio = head_drop_ratio
        selfcsv_in_path = csv_in_path
        self.backbone = self.backbone_type.build(dims=dims, head_drop_ratio=head_drop_ratio, csv_in_path=csv_in_path) # <class 'torchvision.models.resnet.ResNet'>
        impl = self.backbone_type.value[2]
        if impl == Implementation.OFFICIAL:
            if self.backbone_type.value[0] == "vit_patch_16_base":
                self.backbone.head_drop = nn.Identity()
                self.embeddings = L2Norm() 
            elif self.backbone_type.value[0] == "vit_patch_16_tiny":
                self.backbone.head_drop = nn.Identity()
                self.embeddings = L2Norm()
        
    def forward(self, x): # 배치당 처리 
        ##################################################################
        x = self.backbone(x)
        return self.embeddings(x)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group("Model")
        parser.add_argument(
            "--backbone", default="TV_RESNET50", choices=[b.name for b in Backbone]            
        )
        parser.add_argument("--dims", default=512, type=int)
        parser.add_argument("--pool_param", default=3, type=float)
        # parser.add_argument("--block_idx", default=0, type=int)
        # parser.add_argument("--head_idx", default=0, type=int)
