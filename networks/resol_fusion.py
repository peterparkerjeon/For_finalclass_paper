import torch
import torch.nn as nn

class MultiResFusion(nn.Module):
    def __init__(self, in_channels=32): # Here, we put things that we can learn.
        super(MultiResFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1) # do 1x1 convolution to resize

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True) # do batchnorm, Relu.


    def forward(self, High_feature, Low_feature_up):
        combined = torch.cat([High_feature, Low_feature_up], dim=1) # do concat

        out = self.relu(self.bn(self.conv(combined))) # Here, really do conv, bn, relu
        return out

