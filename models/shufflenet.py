import torchvision
from torch.nn import Linear
import torch
from torch import nn
from .myshufflenet import shufflenet_v2_x1_0

class ShuffleNet_v2_30(nn.Module):
    def __init__(self,train=True,pretrained=True,feature_map = False):
        super(ShuffleNet_v2_30, self).__init__()
        self.model=torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained,feature_map = feature_map)
        self.model.fc.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            x, fe_map = self.model(x)
            return x, fe_map

        x = self.model(x)
        return x


class ShuffleNet_with_feature(nn.Module):
    def __init__(self,train=True,pretrained=True,feature_map = False):
        super(ShuffleNet_with_feature, self).__init__()
        self.model=shufflenet_v2_x1_0(pretrained=pretrained,feature_map = feature_map)
        self.model.fc.out_features=30
        self.feature_map = feature_map

    def forward(self,x):
        if self.feature_map:
            return self.model(x)
        return self.model(x)

if __name__ == "__main__":
    net = ShuffleNet_with_feature()
    print(net)

