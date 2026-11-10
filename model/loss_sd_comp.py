# -*- coding: utf-8 -*-
"""
SDM 损失的改进 B：逐连通域位置项（Component-aware SDM Loss）

问题诊断（原论文自述 + 我们的复核）
------------------------------------
AAAI 2025 的 LLoss 用的是 **全图质心**：
    pred_centerx = (x_index * pred[i]).mean()      # 全图所有预测的质心
    target_centerx = (x_index * target[i]).mean()  # 全图所有 GT 的质心
原论文 Limitations 明确承认："Ll_loss ... considers the average position of
all objects in an image, making it difficult to converge when one object is
missed."

对 IRSTD-1K 尤其致命：40.8% 的图是多目标（最多 8 个）。多目标图里
- 一个目标漏检，它的缺失被其它目标"平均"掉 -> 位置项几乎不动；
- 目标 A 的预测偏到目标 B 附近，全图质心可能反而更接近 GT 质心 -> 位置项发假信号。

改进
----
把位置项从"1 张图 1 个质心"改成"1 个 GT 连通域 1 个质心"：
  1. 对每张 GT mask 做 8 连通域标记（scipy.ndimage.label，detach，无梯度）；
  2. 每个连通域 k 取质心 g_k 与面积 a_k，窗口半径 r_k = max(r_min, scale*sqrt(a_k))；
  3. 只在该窗口内算预测的质心 c_k（局部化，避免被邻居目标拉走）；
  4. pos_k = ||c_k - g_k||^2 / r_k^2，裁剪到 [0,1]；
  5. 窗口内无预测质量时 pos_k = 1（最差常数，无梯度；漏检的质量缺口交给 SLS-IoU 项）。
  6. 可选 cov 项：1 - min(1, m_k / a_k)，给漏检目标提供可导的"质量覆盖"梯度。

Loss 值域与原来可比（都是 [0, 2] 量级），β 动态加权的语义不变。
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

_STRUCT8 = np.ones((3, 3), dtype=np.int32)  # 8 连通，与 skimage 默认 connectivity=2 一致


def gt_components(mask, max_comp=16):
    """单个 GT mask (H,W) -> [(cx, cy, area), ...]，按面积降序，最多 max_comp 个。"""
    m = mask.detach().cpu().numpy() > 0.5 if torch.is_tensor(mask) else np.asarray(mask) > 0.5
    if not m.any():
        return []
    lab, n = ndimage.label(m, structure=_STRUCT8)
    if n == 0:
        return []
    flat = lab.ravel()
    areas = np.bincount(flat, minlength=n + 1)[1:].astype(np.float64)
    ys, xs = np.indices(m.shape)
    cx = np.bincount(flat, weights=xs.ravel(), minlength=n + 1)[1:] / areas
    cy = np.bincount(flat, weights=ys.ravel(), minlength=n + 1)[1:] / areas
    comps = sorted(zip(cx.tolist(), cy.tolist(), areas.tolist()), key=lambda t: -t[2])
    return comps[:max_comp]


def component_lloss(pred, target, radius_scale=4.0, min_radius=12.0,
                    max_comp=16, cov_weight=0.5):
    """逐连通域位置损失（替代官方 LLoss 的全图质心版本）。

    Args:
        pred:  (B,1,H,W) 已 sigmoid 的概率图（需要梯度）
        target:(B,1,H,W) 0/1 mask
        radius_scale: 窗口半径 = radius_scale * sqrt(area)
        min_radius:   窗口半径下限（像素）
        max_comp:     每图最多处理的连通域数
        cov_weight:   >0 时加入可导的局部覆盖项（默认 0，等价纯位置项）
    """
    assert pred.dim() == 4 and pred.shape[1] == 1, pred.shape
    B, _, H, W = pred.shape
    losses = []

    for i in range(B):
        comps = gt_components(target[i, 0], max_comp=max_comp)
        if not comps:
            # 保持计算图连通（空图 backward 会报 "does not require grad"）
            losses.append(pred[i].sum() * 0.0)
            continue
        pi = pred[i, 0]
        terms = []
        for (gx, gy, area) in comps:
            r = max(float(min_radius), float(radius_scale) * math.sqrt(max(area, 1.0)))
            x0 = int(max(0, math.floor(gx - r)))
            x1 = int(min(W, math.ceil(gx + r) + 1))
            y0 = int(max(0, math.floor(gy - r)))
            y1 = int(min(H, math.ceil(gy + r) + 1))
            crop = pi[y0:y1, x0:x1]
            m = crop.sum()

            if float(m.detach()) > 1e-6:
                hh, ww = y1 - y0, x1 - x0
                xs = torch.arange(x0, x1, device=pred.device, dtype=pred.dtype).view(1, ww)
                ys = torch.arange(y0, y1, device=pred.device, dtype=pred.dtype).view(hh, 1)
                cx = (crop * xs).sum() / (m + 1e-8)
                cy = (crop * ys).sum() / (m + 1e-8)
                pos = ((cx - gx) ** 2 + (cy - gy) ** 2) / (r * r)
                pos = pos.clamp(0.0, 1.0)
            else:
                pos = pred.new_tensor(1.0)

            if cov_weight > 0:
                # 目标面积越大，要求的窗口内预测质量越多；用 0.5*a 作为"合格线"
                cov = 1.0 - torch.clamp(m / (0.5 * max(area, 1.0)), max=1.0)
                terms.append(pos + cov_weight * cov)
            else:
                terms.append(pos)

        losses.append(torch.stack(terms).mean())

    return torch.stack(losses).mean()


class SLSIoULossComp(nn.Module):
    """AAAI 2025 SDM 损失 + 改进 B（逐连通域位置项）。

    SLS-IoU / α 加权 / 动态 β 与原版逐行一致，仅把 LLoss 换成 component_lloss。
    另外把原版的
        beta = torch.where(beta > delta, torch.tensor(delta), beta)   # CPU 标量张量
    改成 beta.clamp(max=delta)，避免 GPU 上 device 不匹配，且梯度更干净。
    """

    def __init__(self, radius_scale=4.0, min_radius=12.0, max_comp=16, cov_weight=0.5,
                 bg_weight=0.0, gll_weight=0.0):
        super().__init__()
        self.radius_scale = radius_scale
        self.min_radius = min_radius
        self.max_comp = max_comp
        self.cov_weight = cov_weight
        # 修复项：窗口化牺牲了官方 LLoss 的"隐式全局背景抑制"（全图矩量把背景
        # 响应也算进质心），导致 Fa 上升。两条可选补偿：
        #   bg_weight  : 背景 BCE（只在非 GT 像素上，对高置信虚警惩罚最大）
        #   gll_weight : 保留一路原版全图 LLoss 作为辅助项（直接复原抑制来源）
        self.bg_weight = bg_weight
        self.gll_weight = gll_weight

    def lloss(self, pred, target):
        return component_lloss(pred, target,
                               radius_scale=self.radius_scale,
                               min_radius=self.min_radius,
                               max_comp=self.max_comp,
                               cov_weight=self.cov_weight)

    def forward(self, pred_log, target, warm_epoch, epoch,
                with_distance=True, dynamic=True, delta=0.5):
        pred = torch.sigmoid(pred_log)
        h, w = pred.shape[2], pred.shape[3]
        smooth = 0.0

        R_oc = 512 * 512 / (w * h)
        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / \
                (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if dynamic:
                beta = (target_sum * delta * R_oc) / 81
                beta = beta.clamp(max=delta).mean()
                if with_distance:
                    loss = (1 + beta) * (1 - siou_loss.mean()) + (1 - beta) * self.lloss(pred, target)
                else:
                    loss = 1 - siou_loss.mean()
            else:
                if with_distance:
                    loss = 1 - siou_loss.mean() + self.lloss(pred, target)
                else:
                    loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()

        # ---- 可选的全局背景抑制补偿项 ----
        if self.bg_weight > 0:
            bg = 1.0 - target
            bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6),
                                         torch.zeros_like(pred), reduction='none')
            loss = loss + self.bg_weight * (bce * bg).sum() / (bg.sum() + 1e-6)
        if self.gll_weight > 0:
            from .loss_sd import LLoss as _GlobalLLoss
            loss = loss + self.gll_weight * _GlobalLLoss(pred, target)
        return loss
