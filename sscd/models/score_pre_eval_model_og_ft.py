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
from timm.models import create_model

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
    
    def build(self, dims: int):
        impl = self.value[2]
        
        # print(self.value) #('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)
        if impl == Implementation.CLASSY_VISION:
            model = build_model({"name": self.value[0]})
            # Remove head exec wrapper, which we don't need, and breaks pickling
            # (needed for spawn dataloaders).
            return model.classy_model
        
        if impl == Implementation.TORCHVISION:
            return self.value[0](num_classes=dims, zero_init_residual=True)
        
        if impl == Implementation.TORCHVISION_ISC:
            model = resnet50(pretrained=False)
            st = torch.load("/hdd/wi/isc2021/models/multigrain_joint_3B_0.5.pth")
            state_dict = OrderedDict([
                (name[9:], v)
                for name, v in st["model_state"].items() if name.startswith("features.")
            ])
            model.avgpool = nn.Identity()     
            model.fc = nn.Identity()
            # model.avgpool = None # None으로 하면 forward에서 호출하는 게 none이어서 
            # model.fc = None
            model.load_state_dict(state_dict, strict=True)
            return model

        if impl == Implementation.OFFICIAL: #### modi 0722 ###
            if self.value[0] == "vit_patch_16_base":
                model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=False, num_classes=0)
                return model           
            elif self.value[0] == "vit_patch_16_tiny":
                # check eval 모드, ft 모드 일 때 다르니까 잘 확인해야 함
                model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=False, num_classes=0)
                return model
    
        else:
            raise AssertionError("Unsupported OFFICIAL model: %s" % (self.value[0]))
        

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class Model(nn.Module):
    def __init__(self, backbone: str, dims: int, pool_param: float): # og
        super().__init__()
        self.backbone_type = Backbone[backbone] # self.backbone_type = <Backbone.CV_RESNET50>
                                                #<Backbone.CV_RESNET50: ('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)>
                                                #Backbone <enum 'Backbone'> // 'CV_RESNET50'
        # MODI
        # print(f"backbone is {backbone}")
        self.dims = self.backbone_type.value[1]
        self.backbone = self.backbone_type.build(dims=dims)
        impl = self.backbone_type.value[2]
        if impl == Implementation.CLASSY_VISION:
            self.embeddings = nn.Sequential(
                GlobalGeMPool2d(pool_param),
                nn.Linear(self.backbone_type.value[1], dims),
                L2Norm(),
            )
        elif impl == Implementation.TORCHVISION:
            if pool_param > 1:
                self.backbone.avgpool = GlobalGeMPool2d(pooling_param=3.0)
                fc = self.backbone.fc
                nn.init.xavier_uniform_(fc.weight)
                nn.init.constant_(fc.bias, 0)
            self.embeddings = L2Norm()
            # self.embeddings = nn.Identity()
        
        ## MODIFIED 230804##
        elif impl == Implementation.TORCHVISION_ISC:
            if pool_param > 1:
                self.backbone.avgpool = GlobalGeMPool2d(pooling_param=3.0)
                self.backbone.fc = nn.Linear(self.backbone_type.value[1], dims)
            self.embeddings = L2Norm()
        #classy vision은 모델에 pooling, fc없는데 torch vision이랑 pooling이 avg로 달려있음
            # self.embeddings = nn.Sequential(
            #     GlobalGeMPool2d(pooling_param=3.0),
            #     L2Norm(),
            # )
        ## MODIFIED 230724##
        elif impl == Implementation.OFFICIAL:
            if self.backbone_type.value[0] == "vit_patch_16_base":
                self.backbone.head_drop = nn.Identity()
                self.embeddings = L2Norm() 
            elif self.backbone_type.value[0] == "vit_patch_16_tiny":
                self.backbone.head_drop = nn.Identity()
                self.embeddings = L2Norm()
            elif self.backbone_type.value[0] == "vit_tiny_r_s16_p8_224":
                self.embeddings = L2Norm()    

        elif impl == Implementation.MOBILE:
            self.embeddings = L2Norm()
        
        elif impl == Implementation.MY:
            self.backbone.head_drop = nn.Identity()
            self.dropout = nn.Dropout(p=0.5)
            # self.last_fc_layer = nn.Linear(self.dims*2, self.dims)  # 차원 축소 레이어 추가
            self.bn = nn.BatchNorm1d(self.dims)
            self.relu = nn.ReLU()
            self.embeddings = L2Norm()


    def forward(self, x): # 배치당 처리 
        ##################################################################
        #check
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
