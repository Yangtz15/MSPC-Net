# -*- coding: utf-8 -*-
"""
Coordinate Attention（CVPR 2021, Hou et al.）
"Coordinate Attention for Efficient Mobile Network Design"

用途：替换 MSHNet 残差块内原有的 SE 式 ChannelAttention（原文即
"可直接替换现有网络中的通道注意力模块"）。

适配说明：MSHNet 的用法是 `out = self.ca(out) * out`（模块返回权重），
因此本实现返回 a_h * a_w 的权重图，保持调用处一行不改。
"""

import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttentionWeight(nn.Module):
    """坐标注意力，返回权重图 (B,C,H,W)，与 SE 版 ChannelAttention 接口一致。"""

    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # (B,C,1,W)
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                                  # (B,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)              # (B,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)                      # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                         # (B,C,1,W)
        a_h = self.conv_h(x_h).sigmoid()                      # (B,C,H,1)
        a_w = self.conv_w(x_w).sigmoid()                      # (B,C,1,W)
        return a_h * a_w                                      # 广播成 (B,C,H,W)
