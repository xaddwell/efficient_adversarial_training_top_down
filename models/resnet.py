import torchvision
from torch.nn import Linear
import torch
from torch import nn
from .myresnet import resnet18

class ResNet18_30(nn.Module):
    def __init__(self,train=True,pretrained=True,feature_map = False):
        super(ResNet18_30, self).__init__()
        self.model=torchvision.models.resnet18(pretrained=True,feature_map = feature_map)
        self.model.fc.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map

        x = self.model(x)
        return x

class ResNet18_with_feature(nn.Module):
    def __init__(self,pretrained=True,feature_map = False):
        super(ResNet18_with_feature, self).__init__()
        self.model=resnet18(pretrained=True,feature_map = feature_map)
        self.model.fc.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        return self.model(x)