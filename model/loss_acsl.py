"""AC-SLSIoU 复现 (arXiv 2607.01555)

Adaptive-Contrastive SLSIoU = SLS IoU 损失 + 三个即插即用组件:
  1. LDMC  Logit-Domain Margin Constraint
           OHNM 挑出 top-k 高概率背景像素(hard negatives), 与正样本在
           **未激活的 logit 域**做 all-to-all softplus 排序损失, 规避
           sigmoid 饱和导致的梯度消失。
  2. ABS   Adaptive Boundary Suppression
           对每个 GT 连通域生成尺度自适应的环形惩罚带(带宽 ∝ sqrt(area)),
           抑制热扩散造成的 halo 溢出响应。
  3. FAFL  False-Alarm Focal Loss
           只对预测概率超过 tau 的背景像素施加 focal 惩罚, 权重 p^gamma,
           专门打压高置信虚警。

注意: 原论文代码未开源, 超参(三个权重/margin/OHNM 比例/带宽系数)为本复现
自行设定的默认值, 与论文本体可能存在差异; 论文中报告的数字在论文里按
"引用值"处理, 本实现的结果按"our reproduction"标注。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

_S8 = np.ones((3, 3), np.int8)


def _annulus_mask(target_np, alpha=0.8, max_w=10):
    """尺度自适应环形惩罚带(CPU, numpy)。

    target_np: (H, W) bool
    返回: (H, W) bool, 需要被抑制的环形背景区域
    """
    t = target_np.astype(bool)
    if not t.any():
        return np.zeros_like(t, dtype=bool)
    # 背景像素到最近目标像素的欧氏距离
    dist = ndimage.distance_transform_edt(~t)
    lab, n = ndimage.label(t, structure=_S8)
    if n == 0:
        return np.zeros_like(t, dtype=bool)
    # 每个背景像素"最近的目标"所属连通域 -> 该域面积 -> 带宽
    idx = ndimage.distance_transform_edt(~t, return_indices=True)[1]
    nearest = lab[idx[0], idx[1]]
    areas = ndimage.sum(t, lab, range(1, n + 1))
    lut = np.zeros(n + 1, dtype=np.float64)
    lut[1:] = np.asarray(areas, dtype=np.float64)
    a_map = lut[nearest]
    w_map = np.clip(np.round(alpha * np.sqrt(a_map / np.pi)), 1, max_w)
    # 紧贴目标的一圈(dist<=1.5)不惩罚(那是目标边缘), 再往外 w 个像素为惩罚带
    ring = (dist > 1.5) & (dist <= 1.5 + w_map)
    ring &= ~t
    return ring


class ACSLSIoULoss(nn.Module):
    """SLS IoU 损失(官方 with_shape=True, 含全局质心 LLoss) + LDMC + ABS + FAFL"""

    def __init__(self, w_ldmc=1.0, w_abs=1.0, w_fafl=1.0,
                 margin=1.0, ohnm_frac=0.05, ohnm_max=512,
                 abs_alpha=0.8, abs_max_w=10,
                 fafl_tau=0.5, fafl_gamma=2.0,
                 with_shape=True):
        super().__init__()
        self.w_ldmc = w_ldmc
        self.w_abs = w_abs
        self.w_fafl = w_fafl
        self.margin = margin
        self.ohnm_frac = ohnm_frac
        self.ohnm_max = ohnm_max
        self.abs_alpha = abs_alpha
        self.abs_max_w = abs_max_w
        self.fafl_tau = fafl_tau
        self.fafl_gamma = fafl_gamma
        self.with_shape = with_shape

    # ---------- 三个组件 ----------
    def _ldmc(self, z, t):
        """z: logits (B,1,H,W);  t: bool (B,1,H,W)"""
        out = []
        B = z.shape[0]
        for i in range(B):
            zi = z[i, 0]
            ti = t[i, 0]
            zp = zi[ti]
            bg = zi[~ti]
            if zp.numel() == 0 or bg.numel() == 0:
                out.append(zp.sum() * 0.0 + bg.sum() * 0.0)
                continue
            with torch.no_grad():
                pb = torch.sigmoid(bg)
                k = int(self.ohnm_frac * bg.numel())
                k = max(1, min(k, self.ohnm_max, bg.numel()))
                _, idx = torch.topk(pb.reshape(-1), k)
            zn = bg.reshape(-1)[idx]
            # all-to-all:  softplus(z_neg - z_pos + m)
            d = zn.unsqueeze(0) - zp.unsqueeze(1) + self.margin
            out.append(F.softplus(d).mean())
        return sum(out) / B

    def _abs(self, pred, t_np):
        """pred: 概率 (B,1,H,W); t_np: (B,H,W) bool (numpy)"""
        out = []
        B = pred.shape[0]
        for i in range(B):
            ring = _annulus_mask(t_np[i], self.abs_alpha, self.abs_max_w)
            if not ring.any():
                out.append(pred[i, 0].sum() * 0.0)
                continue
            r = torch.as_tensor(ring, device=pred.device)
            out.append(pred[i, 0][r].mean())
        return sum(out) / B

    def _fafl(self, pred, t):
        """只对高概率背景像素的 focal 惩罚"""
        out = []
        B = pred.shape[0]
        for i in range(B):
            pi = pred[i, 0]
            with torch.no_grad():
                m = (~t[i, 0]) & (pi > self.fafl_tau)
            if not bool(m.any()):
                out.append(pi.sum() * 0.0)
                continue
            pn = pi[m]
            out.append(-(pn ** self.fafl_gamma * torch.log(1 - pn + 1e-6)).mean())
        return sum(out) / B

    # ---------- 主入口 ----------
    def forward(self, pred_log, target, warm_epoch, epoch):
        pred = torch.sigmoid(pred_log)
        smooth = 1e-8
        intersection = pred * target
        inter_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        dis = torch.pow((pred_sum - target_sum) / 2, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / \
                (torch.max(pred_sum, target_sum) + dis + smooth)
        iou = (inter_sum + smooth) / (pred_sum + target_sum - inter_sum + smooth)

        t_bool = target > 0.5
        t_np = t_bool[:, 0].detach().cpu().numpy().astype(bool)

        if epoch <= warm_epoch:
            # warmup 阶段保持与官方一致, 只优化 IoU
            return 1 - iou.mean()

        sls = 1 - (alpha * iou).mean()
        if self.with_shape:
            from model.loss import LLoss
            sls = sls + LLoss(pred, target)

        loss = sls
        if self.w_ldmc > 0:
            loss = loss + self.w_ldmc * self._ldmc(pred_log, t_bool)
        if self.w_abs > 0:
            loss = loss + self.w_abs * self._abs(pred, t_np)
        if self.w_fafl > 0:
            loss = loss + self.w_fafl * self._fafl(pred, t_bool)
        return loss
