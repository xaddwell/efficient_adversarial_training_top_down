import torchvision
from torch.nn import Linear
import torch
from torch import nn

class Mobilenet_v2_30(nn.Module):
    def __init__(self,train=True,pretrained=True,feature_map = False):
        super(Mobilenet_v2_30, self).__init__()
        self.model=torchvision.models.mobilenet_v2(pretrained=True,feature_map = feature_map)
        self.model.classifier.add_module(name='2',module=nn.Dropout(p=0.2,inplace=False))
        self.model.classifier.add_module(name='3', module=nn.Linear(in_features=1000,out_features=30))
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map
        x=self.model(x)
        return x



