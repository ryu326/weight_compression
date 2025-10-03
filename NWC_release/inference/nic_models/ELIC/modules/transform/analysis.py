import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import AttentionBlock, subpel_conv3x3
from models.ELIC.modules.layers.conv import conv, conv1x1, conv3x3, deconv
from models.ELIC.modules.layers.res_blk import *


class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, M),
            AttentionBlock(M),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x


class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(conv3x3(M, N), act(), conv(N, N), act(), conv(N, N))

    def forward(self, x):
        x = self.reduction(x)
        return x
