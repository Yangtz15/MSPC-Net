# -*- coding: utf-8 -*-
"""
Pinwheel-shaped Convolution (PConv) — 复现自 AAAI 2025
"Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared
Small Target Detection"（Yang et al.，官方仓库 JN-Yang/PConv-SDloss-Data，
model/APConv.py）。本文件仅做最小适配（激活开关），结构保持与官方一致。

用途：作为引用的对比锚点组件（非本文贡献）。
"""

import torch
import torch.nn as nn


class ConvNS(nn.Module):
    """官方 Conv 的去激活变体（MSHNet 块内自己控制 BN/ReLU）。"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvA(ConvNS):
    """官方 Conv（SiLU 激活）等价实现。"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__(c1, c2, k, s, p, g)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(super().forward(x))


class PConv(nn.Module):
    """Pinwheel-shaped Convolution using the Asymmetric Padding method.

    官方版 forward:
        yw0/yw1: (1,k) 水平方向两组（不同的非对称 pad）
        yh0/yh1: (k,1) 垂直方向两组
        cat 后经 2x2 卷积融合到 c2
    act=False 时输出为 BN 后特征（不带激活），供 MSHNet 块内接自身 ReLU。
    """

    def __init__(self, c1, c2, k, s=1, act=True):
        super().__init__()
        assert c2 % 4 == 0, "PConv requires c2 divisible by 4"
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(p[g]) for g in range(4)])
        cls = ConvA if act else ConvNS
        self.cw = cls(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = cls(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = cls(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
