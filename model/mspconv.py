# -*- coding: utf-8 -*-
"""
MS-PConv（Multi-Scale Pinwheel Convolution）—— 改进 A
针对 AAAI 2025 PConv 的问题：单一 k 的风车感受野只覆盖一种"风车臂长"，
而红外小目标尺寸跨度极大（1x1 ~ 20x20 像素），单一臂长在两端都不最优。

官方 PConv(k) 的真实结构（容易看错的地方：是**权重共享**）
----------------------------------------------------------
    self.cw = Conv(c1, c2/4, (1,k))    # 一个模块
    self.ch = Conv(c1, c2/4, (k,1))    # 一个模块
    yw0 = cw(pad(k,0,1,0)(x));  yw1 = cw(pad(0,k,0,1)(x))   # 同一个 cw 用两次
    yh0 = ch(pad(0,1,k,0)(x));  yh1 = ch(pad(1,0,0,k)(x))   # 同一个 ch 用两次
    cat(4 支) -> Conv2d(c2, c2, 2) 融合回 c2
非对称 pad 使每支输出 (H/s+1, W/s+1)，2x2 卷积正好收回 (H/s, W/s)。

MS-PConv 的构造（参数与 PConv(k=4) **完全相等**）
----------------------------------------------
把 c2 切成 8 份（每份 cw = c2/8）：k=3 的四支 + k=5 的四支。
每个 k 只要两个卷积模块（h/v 各一，各自 width=cw），各用两次不同的 pad：
    参数 = Σ_k 2 * c1 * cw * k = 2 * c1 * (c2/8) * (3+5) = 2*c1*c2
    PConv(k=4): 2 * c1 * (c2/4) * 4              = 2*c1*c2     ← 完全相同
    cat 卷积 c2->c2 也完全相同
=> 参数、FLOPs、每层张量形状全部对齐，唯一变量是"风车臂长是否多样"。
   这是干净的受控消融：同样预算下，感受野多样性本身值多少分。

可选 gate=True：cat 之前按 k 分组做通道门控（SE 式，末层零初始化 -> 初始恒等），
默认关闭以保持与 PConv 严格对齐。
"""

import torch
import torch.nn as nn

from .pconv import ConvA, ConvNS


class MSPConv(nn.Module):
    """多尺度风车卷积：并联 k=3 与 k=5 两组风车臂，共用 2x2 融合卷积。

    Args:
        c1, c2: 输入/输出通道。要求 c2 % (4*len(ks)) == 0。
        ks:     风车臂长集合，默认 (3, 5)（均值 4，对齐原论文 k=4 的预算）。
        s:      步长（与 PConv 一致，作用在非对称卷积上）。
        act:    是否在卷积后接 SiLU；MSHNet 块内传 act=False。
        gate:   按 k 分组的通道门控（默认 False）。
        share:  同一 k 内 h/v 是否共享（无法共享，保留参数以兼容旧脚本）。
    """

    def __init__(self, c1, c2, ks=(3, 5), s=1, act=True, gate=False, share=True):
        super().__init__()
        ks = tuple(ks)
        n_br = 4 * len(ks)          # 支路数
        assert c2 % n_br == 0, f"MSPConv requires c2 divisible by {n_br}, got {c2}"
        cw = c2 // n_br             # 每支输出宽度

        cls = ConvA if act else ConvNS
        self.ks = ks
        self.cw = cw

        self.pads = nn.ModuleList()
        self.h_convs = nn.ModuleList()
        self.v_convs = nn.ModuleList()
        for k in ks:
            # pad 顺序：h0(k,0,1,0) h1(0,k,0,1) v0(0,1,k,0) v1(1,0,0,k)，与官方一致
            for p in [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]:
                self.pads.append(nn.ZeroPad2d(p))
            self.h_convs.append(cls(c1, cw, (1, k), s=s, p=0))
            self.v_convs.append(cls(c1, cw, (k, 1), s=s, p=0))

        self.gate = None
        if gate:
            hidden = max(c2 // 8, 8)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, c2, 1, bias=True),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

        self.cat = cls(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        outs = []
        for gi in range(len(self.ks)):
            pi = 4 * gi
            hc, vc = self.h_convs[gi], self.v_convs[gi]
            outs.append(hc(self.pads[pi + 0](x)))   # 共享 hc
            outs.append(hc(self.pads[pi + 1](x)))   # 共享 hc
            outs.append(vc(self.pads[pi + 2](x)))   # 共享 vc
            outs.append(vc(self.pads[pi + 3](x)))   # 共享 vc
        y = torch.cat(outs, dim=1)
        if self.gate is not None:
            g = torch.sigmoid(self.gate(y))
            y = y * (0.5 + g)   # 零初始化 -> 1.0，初始恒等
        return self.cat(y)
