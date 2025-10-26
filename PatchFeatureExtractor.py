from collections import OrderedDict

from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch

class BaseNet(nn.Module):
    def __init__(self, feature_depth):
        super(BaseNet, self).__init__()
        self.feature_depth = feature_depth
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, self.feature_depth[0], kernel_size=4)),
            ('norm1', nn.BatchNorm3d(self.feature_depth[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(self.feature_depth[0], self.feature_depth[1], kernel_size=3)),
            ('norm2', nn.BatchNorm3d(self.feature_depth[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=2)),
            ('conv3', nn.Conv3d(self.feature_depth[1], self.feature_depth[2], kernel_size=3)),
            ('norm3', nn.BatchNorm3d(self.feature_depth[2])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv3d(self.feature_depth[2], self.feature_depth[3], kernel_size=3)),
            ('norm4', nn.BatchNorm3d(self.feature_depth[3])),
            ('relu4', nn.ReLU(inplace=True)),
        ]))
        self.classify = nn.Sequential(
            nn.Linear(self.feature_depth[3], 1),
            nn.Sigmoid()
        )
        self.GAP = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.GMP = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, input):
        local_feature = self.features(input)
        feature_ = F.adaptive_avg_pool3d(local_feature, (1, 1, 1))
        score = self.classify(feature_.flatten(1, -1))
        return [local_feature, score]



if __name__ == '__main__':
    model = BaseNet([32, 64, 128, 1])
    x = torch.randn(6, 1, 25, 25, 25)
    # input:: [batch_size, channel_size, patch_w, patch_h, patch_d]
    print(model(x)[0].shape)
    # torch.Size([6, 128, 1, 1, 1])
    print(model(x)[1].shape)
    # torch.Size([6, 1])
