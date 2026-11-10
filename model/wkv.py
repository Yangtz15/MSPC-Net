# -*- coding: utf-8 -*-
"""
WKV 线性注意力模块（加权键值线性注意力，RWKV/VRWKV 风格）

Spatial Mix: X -> Norm -> 1x1 conv -> R(门控)/K(键)/V(值)
             WKV 算子（可学习衰减 w、当前 token 加成 u，双向聚合）
             输出 = WKV(K,V) * Sigmoid(R)，残差连接
Channel Mix: X -> Norm -> Rc/Kc -> ReLU^2 MLP * Sigmoid(Rc)，残差连接

实现要点：
  - 矩阵形式（parallel WKV），对 16x16=256 token 的 middle 层极轻量；
    按通道分块以约束峰值显存。
  - k 做逐 (batch, channel) 最大值平移：分子分母同用一组权重，平移在
    WKV 归一化中严格抵消，数值上保证 exp 不上溢。
  - 双向 = 正向序列一遍 + 翻转序列一遍再翻回，取平均。
  - 衰减参数化: w_eff = exp(w_raw) > 0，距离 d 的权重 exp(-d*w_eff)；
    当前 token 权重 exp(u)。
  - 零初始化残差门控: out = x + gamma * mix(x)，gamma 初始 0，
    最坏情况退化为恒等映射（不拖垮 baseline）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def wkv_forward(k, v, w_eff, u, chunk=32, eps=1e-8):
    """单向 WKV（矩阵形式）。

    Args:
        k: [B, T, C] 键（logits）
        v: [B, T, C] 值
        w_eff: [C] 正衰减率
        u: [C] 当前 token 加成（log 域）
    Returns:
        [B, T, C] 凸组合结果（量级与 v 相当）
    """
    B, T, C = k.shape
    device, dtype = k.device, k.dtype

    kmax = k.max(dim=1, keepdim=True).values          # [B,1,C]
    ek = torch.exp(k - kmax)                           # (0,1]
    ekv = ek * v

    idx = torch.arange(T, device=device, dtype=torch.float32)
    dist = (idx[:, None] - idx[None, :]).clamp_min(0)  # [T,T] 仅 j<=t 保留
    offdiag = (1.0 - torch.eye(T, device=device, dtype=torch.float32))  # 非对角掩码

    out = torch.empty_like(k)
    for c0 in range(0, C, chunk):
        c1 = min(c0 + chunk, C)
        w = w_eff[c0:c1].to(torch.float32)             # [g]
        W = torch.exp(-dist[None] * w[:, None, None]) * offdiag  # [g,T,T]
        W = W + torch.eye(T, device=device, dtype=torch.float32) * \
            torch.exp(u[c0:c1].to(torch.float32))[:, None, None]  # 对角 = exp(u)
        W = W.to(dtype)
        kc = ek[:, :, c0:c1].permute(0, 2, 1)          # [B,g,T]
        vc = ekv[:, :, c0:c1].permute(0, 2, 1)         # [B,g,T]
        den = torch.einsum("gtj,bgj->btg", W, kc)      # [B,T,g]
        num = torch.einsum("gtj,bgj->btg", W, vc)
        out[:, :, c0:c1] = num / (den + eps)           # [B,T,g] 已对齐通道片
    return out


def wkv_bidirectional(k, v, w_eff, u):
    fwd = wkv_forward(k, v, w_eff, u)
    bwd = wkv_forward(k.flip(1), v.flip(1), w_eff, u).flip(1)
    return 0.5 * (fwd + bwd)


class WKVSpatialMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)   # 通道维 LayerNorm（NHWC 语义）
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Conv2d(dim, dim, 1)
        self.w_raw = nn.Parameter(torch.zeros(dim))    # w_eff = exp(w_raw)
        self.u = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gate.weight, 0.0)
        nn.init.constant_(self.gate.bias, 0.0)         # 门控初始为 0.5
        # w_raw 初始 0 -> w_eff=1，距离 1 即衰减到 0.37，偏"局部"

    def forward(self, x):
        B, C, H, W = x.shape
        xn = self.norm(x)
        k = self.key(xn)
        v = self.value(xn)
        r = torch.sigmoid(self.gate(xn))

        k_seq = k.flatten(2).transpose(1, 2)           # [B, HW, C]
        v_seq = v.flatten(2).transpose(1, 2)
        w_eff = torch.exp(self.w_raw)
        y = wkv_bidirectional(k_seq, v_seq, w_eff, self.u)
        y = y.transpose(1, 2).reshape(B, C, H, W)
        return y * r


class WKVChannelMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.kc = nn.Conv2d(dim, dim, 1)
        self.rc = nn.Conv2d(dim, dim, 1)
        self.vc = nn.Conv2d(dim, dim, 1)
        nn.init.constant_(self.rc.weight, 0.0)
        nn.init.constant_(self.rc.bias, 0.0)

    def forward(self, x):
        xn = self.norm(x)
        h = F.relu(self.kc(xn)) ** 2
        return self.vc(h) * torch.sigmoid(self.rc(xn))


class WKVUnit(nn.Module):
    """即插即用单元：Spatial Mix -> Channel Mix，各自零初始化残差门控。"""

    def __init__(self, dim):
        super().__init__()
        self.spatial = WKVSpatialMix(dim)
        self.channel = WKVChannelMix(dim)
        self.gamma_s = nn.Parameter(torch.zeros(1))
        self.gamma_c = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x + self.gamma_s * self.spatial(x)
        x = x + self.gamma_c * self.channel(x)
        return x
