# -*- coding: utf-8 -*-
"""
MSHNet-WKV 服务器训练脚本（AutoDL / RTX 4090）

基于官方 main.py 改造：
  - 组件开关 --wkv / --pconv（四臂实验矩阵）
  - 每 epoch 追加 metrics.jsonl（一行一 epoch，含子集分指标）
  - checkpoint 续跑（torch>=2.6 weights_only 兼容 + 写回原目录）
  - num_workers=0（Windows/容器 DataLoader 多进程静默死亡规避）
  - 独立评测实例（PD_FA 每次 val 新建，规避 reset 漏 target 的上游 bug——
    utils/metric.py 已修复，这里双保险）
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import json
import time
import os.path as osp
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adagrad
from tqdm import tqdm

from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, mIoU
from model.MSHNet import MSHNet
from model.loss import SLSIoULoss
from model.loss_sd import SLSIoULoss as SLSIoULossSD
from model.loss_sd_comp import SLSIoULossComp
from model.loss_acsl import ACSLSIoULoss


def parse_args():
    p = ArgumentParser()
    p.add_argument('--dataset-dir', type=str, default='datasets/SIRST3',
                   help='dataset folder containing trainval.txt / test.txt / images / masks')
    p.add_argument('--save-root', type=str, default='runs',
                   help='folder for checkpoints, logs and summaries')
    p.add_argument('--tag', type=str, required=True,
                   help='run tag, e.g. base / msp / sdcomp_bg16 / msp_sdcomp_bg16')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--warm-epoch', type=int, default=5)
    p.add_argument('--base-size', type=int, default=256)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--wkv', action='store_true')
    p.add_argument('--pconv', action='store_true')
    p.add_argument('--pconv-k', type=int, default=4)
    p.add_argument('--mspconv', action='store_true', help='改进A：多尺度风车卷积（参数与 pconv 相等）')
    p.add_argument('--msp-ks', type=str, default='3,5', help='风车臂长集合，如 3,5')
    p.add_argument('--msp-gate', action='store_true', help='改进A可选：按 k 分组的通道门控')
    p.add_argument('--ca', action='store_true')
    p.add_argument('--sdloss', action='store_true', help='用 AAAI25 的 SDM 损失（复现参照臂）')
    p.add_argument('--acsl', action='store_true', help='AC-SLSIoU 复现（arXiv 2607.01555 对比方法）')
    p.add_argument('--ldmc-w', type=float, default=1.0, help='AC-SLSIoU: 对数域 margin 项权重')
    p.add_argument('--abs-w', type=float, default=1.0, help='AC-SLSIoU: 尺度自适应边界抑制权重')
    p.add_argument('--fafl-w', type=float, default=1.0, help='AC-SLSIoU: 高置信虚警 focal 权重')
    p.add_argument('--acsl-margin', type=float, default=1.0, help='AC-SLSIoU: LDMC 的 logit margin')
    p.add_argument('--ohnm-frac', type=float, default=0.05, help='AC-SLSIoU: OHNM 取背景 top 比例')
    p.add_argument('--abs-alpha', type=float, default=0.8, help='AC-SLSIoU: 环带带宽系数')
    p.add_argument('--fafl-tau', type=float, default=0.5, help='AC-SLSIoU: FAFL 概率阈值')
    p.add_argument('--sdcomp', action='store_true', help='改进B：逐连通域位置项的 SDM 损失')
    p.add_argument('--cov-weight', type=float, default=0.5, help='改进B的局部覆盖项权重')
    p.add_argument('--min-radius', type=float, default=12.0, help='改进B的连通域窗口半径下限')
    p.add_argument('--bg-weight', type=float, default=0.0,
                   help='改进B修复项：背景 BCE 权重（补偿窗口化丢失的全局背景抑制）')
    p.add_argument('--gll-weight', type=float, default=0.0,
                   help='改进B修复项：保留一路原版全图 LLoss 辅助项的权重')
    p.add_argument('--eval-interval', type=int, default=10)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--persistent', action='store_true', default=True)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--seed', type=int, default=-1, help='-1 = 不设 seed（框架默认随机初始化）')
    p.add_argument('--source-map', type=str, default='',
                   help='name<TAB>source 映射文件，用于子集分指标；空则跳过子集评测')
    return p.parse_args()


def load_eval_lists(args):
    """返回 {source: [names]} 用于子集分指标。"""
    if not args.source_map or not osp.exists(args.source_map):
        return {}
    m = {}
    with open(args.source_map) as f:
        for line in f:
            if not line.strip():
                continue
            name, src = line.strip().rsplit('\t', 1)
            m.setdefault(src, []).append(name)
    return m


class Evaluator:
    def __init__(self, base_size):
        self.base_size = base_size

    def run(self, model, loader, device, warm_tag):
        miou = mIoU(1)
        pd_fa = PD_FA(1, 10, self.base_size)  # 每次新建，天然免疫 reset bug
        model.eval()
        with torch.no_grad():
            for data, mask in loader:
                data, mask = data.to(device), mask.to(device)
                _, pred = model(data, warm_tag)
                miou.update(pred, mask)
                pd_fa.update(pred, mask)
        _, mean_iou = miou.get()
        FA, PD = pd_fa.get(len(loader))
        return mean_iou * 100, PD[0] * 100, FA[0] * 1e6


@torch.no_grad()
def eval_by_source_stub():
    """子集分指标在独立评测脚本 eval_sirst3.py 中实现（支持任意文件名列表）。"""
    pass


def main():
    args = parse_args()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    if args.seed >= 0:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tag = args.tag
    save_folder = osp.join(args.save_root, f'{tag}-{time.strftime("%Y-%m-%d-%H-%M-%S")}')
    if args.resume:
        save_folder = osp.dirname(osp.abspath(args.resume))  # 续跑写回原目录
    os.makedirs(save_folder, exist_ok=True)

    trainset = IRSTD_Dataset(args, mode='train')
    valset = IRSTD_Dataset(args, mode='val')
    train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True,
                                   drop_last=True, num_workers=args.workers,
                                   pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = Data.DataLoader(valset, 1, drop_last=False,
                                 num_workers=max(4, args.workers // 2), pin_memory=True)

    msp_ks = tuple(int(x) for x in args.msp_ks.split(',') if x.strip())
    model = MSHNet(3, use_wkv=args.wkv, use_pconv=args.pconv, use_ca=args.ca, wkv_dim=256,
                   use_mspconv=args.mspconv, msp_ks=msp_ks, msp_gate=args.msp_gate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{tag}] params={n_params/1e6:.2f}M wkv={args.wkv} conv={model.conv_type} '
          f'msp_ks={msp_ks if args.mspconv else "-"} gate={args.msp_gate} '
          f'ca={args.ca} sdloss={args.sdloss} sdcomp={args.sdcomp}', flush=True)

    optimizer = Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.sdcomp:
        loss_fun = SLSIoULossComp(cov_weight=args.cov_weight, min_radius=args.min_radius,
                                  bg_weight=args.bg_weight,
                                  gll_weight=args.gll_weight).to(device)
    elif args.acsl:
        loss_fun = ACSLSIoULoss(w_ldmc=args.ldmc_w, w_abs=args.abs_w, w_fafl=args.fafl_w,
                                margin=args.acsl_margin, ohnm_frac=args.ohnm_frac,
                                abs_alpha=args.abs_alpha, fafl_tau=args.fafl_tau).to(device)
    elif args.sdloss:
        loss_fun = SLSIoULossSD().to(device)
    else:
        loss_fun = SLSIoULoss().to(device)
    down = nn.MaxPool2d(2, 2)

    start_epoch, best_iou = 0, 0
    if args.resume and osp.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['net'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1
        best_iou = ck.get('iou', 0)
        print(f'resumed from {args.resume} at epoch {start_epoch}, best_iou={best_iou}', flush=True)

    metrics_path = osp.join(save_folder, 'metrics.jsonl')
    sources = load_eval_lists(args)
    evaluator = Evaluator(args.base_size)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        losses, tag_multi = [], epoch > args.warm_epoch
        tbar = tqdm(train_loader, desc=f'[{tag}] ep{epoch}', ncols=100)
        for i, (data, mask) in enumerate(tbar):
            data, labels = data.to(device), mask.to(device)
            masks, pred = model(data, tag_multi)
            loss = loss_fun(pred, labels, args.warm_epoch, epoch)
            for j in range(len(masks)):
                if j > 0:
                    labels = down(labels)
                loss = loss + loss_fun(masks[j], labels, args.warm_epoch, epoch)
            loss = loss / (len(masks) + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tbar.set_postfix(loss=f'{sum(losses)/len(losses):.4f}')

        rec = {'tag': tag, 'epoch': epoch, 'loss': sum(losses) / len(losses),
               'sec': round(time.time() - t0, 1)}

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            miou, pd, fa = evaluator.run(model, val_loader, device, tag_multi)
            rec.update(iou=round(miou, 2), pd=round(pd, 2), fa=round(fa, 2))
            if miou > best_iou:
                best_iou = miou
                torch.save(model.state_dict(), osp.join(save_folder, 'weight.pkl'))
                rec['best'] = True
            print(f'[{tag}] ep{epoch} IoU {miou:.2f} Pd {pd:.2f} Fa {fa:.2f} '
                  f'best {best_iou:.2f} ({rec["sec"]}s/ep)', flush=True)

        with open(metrics_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

        ck = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
              'epoch': epoch, 'iou': best_iou}
        torch.save(ck, osp.join(save_folder, 'checkpoint.pkl'))

    with open(osp.join(save_folder, 'summary.json'), 'w') as f:
        json.dump({'tag': tag, 'best_iou': best_iou, 'epochs': args.epochs,
                   'params_m': round(n_params / 1e6, 2),
                   'wkv': args.wkv, 'pconv': args.pconv, 'mspconv': args.mspconv,
                   'conv_type': model.conv_type, 'ca': args.ca,
                   'sdloss': args.sdloss, 'sdcomp': args.sdcomp, 'acsl': args.acsl,
                   'acsl_w': [args.ldmc_w, args.abs_w, args.fafl_w] if args.acsl else None,
                   'cov_weight': args.cov_weight,
                   'bg_weight': args.bg_weight, 'gll_weight': args.gll_weight,
                   'seed': args.seed if args.seed >= 0 else None}, f, indent=2)
    print(f'[{tag}] DONE best_iou={best_iou:.2f}', flush=True)


if __name__ == '__main__':
    main()
