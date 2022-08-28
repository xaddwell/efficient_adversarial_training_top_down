import torchvision
from torch.nn import Linear
import torch
from torch import nn

class Densenet121_30(nn.Module):
    def __init__(self,train=True,pretrained=True,feature_map = False):
        super(Densenet121_30, self).__init__()
        self.model=torchvision.models.densenet121(pretrained=True,feature_map = feature_map)
        self.model.classifier.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map
        x = self.model(x)
        return x


