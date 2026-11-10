# -*- coding: utf-8 -*-
"""
SIRST3 统一评测：合并口径 + 三个子集口径，一次遍历全部算完。

用法:
    python eval_sirst3.py                     # 自动扫 runs/*/weight.pkl
    python eval_sirst3.py --tags base pconv msp

输出:
    runs/_eval_sirst3.md     可直接贴进论文的表格（含按目标面积/多目标的分层统计）
    runs/_eval_sirst3.json   明细

为什么不用 main_train.py 里的 Evaluator：
  - 那里只有合并口径，看不到子集差异（IRSTD-1K 与 NUAA 的难度差 3 倍以上）；
  - PD_FA 的 reset() 漏掉 self.target（上游 bug），这里每个口径都新建实例规避。
"""
import json
import os
import os.path as osp
import re
import sys
from argparse import ArgumentParser

import torch
import torch.utils.data as Data

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np
from scipy import ndimage

from utils.data import IRSTD_Dataset          # noqa: E402
from utils.metric import PD_FA, mIoU          # noqa: E402
from model.MSHNet import MSHNet               # noqa: E402

_S8 = np.ones((3, 3), dtype=np.int32)  # 8 连通


def parse_args():
    p = ArgumentParser()
    p.add_argument('--dataset-dir', type=str, default='datasets/SIRST3')
    p.add_argument('--save-root', type=str, default='runs')
    p.add_argument('--log-dir', type=str, default='runs',
                   help='folder holding <tag>.log when summary.json is unavailable')
    p.add_argument('--source-map', type=str, default='datasets/SIRST3_source_map.txt',
                   help='optional TSV mapping image name -> source subset (for the per-subset columns)')
    p.add_argument('--base-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tags', nargs='*', default=None)
    p.add_argument('--out', type=str, default='')
    return p.parse_args()


def tag_of(dirname):
    """'base-2026-10-26-19-39-12' -> 'base'"""
    return re.sub(r'-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$', '', dirname)


def build_model_from_dir(run_dir, log_dir):
    """从 summary.json 或训练日志的启动行恢复结构开关。"""
    cfg = dict(wkv=False, conv_type='std', ca=False, msp_ks=(3, 5), msp_gate=False)
    sj = osp.join(run_dir, 'summary.json')
    if osp.exists(sj):
        d = json.load(open(sj))
        cfg['wkv'] = bool(d.get('wkv', False))
        cfg['ca'] = bool(d.get('ca', False))
        ct = d.get('conv_type')
        if ct is None:
            ct = 'pconv' if d.get('pconv') else ('mspconv' if d.get('mspconv') else 'std')
        cfg['conv_type'] = ct
        return cfg
    tag = tag_of(osp.basename(run_dir))
    lg = osp.join(log_dir, f'{tag}.log')
    if osp.exists(lg):
        head = open(lg, 'r', errors='replace').read(2000)
        m = re.search(r'params=[\d.]+M wkv=(\w+) conv=(\w+)', head)
        if m:
            cfg['wkv'] = m.group(1) == 'True'
            cfg['conv_type'] = m.group(2)
            m2 = re.search(r'gate=(\w+) ca=(\w+)', head)
            if m2:
                cfg['msp_gate'] = m2.group(1) == 'True'
                cfg['ca'] = m2.group(2) == 'True'
    return cfg


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 数据集 ----
    class A:
        pass
    a = A()
    a.dataset_dir = args.dataset_dir
    a.crop_size = args.base_size
    a.base_size = args.base_size
    valset = IRSTD_Dataset(a, mode='val')
    names = list(valset.names)
    loader = Data.DataLoader(valset, 1, shuffle=False, drop_last=False,
                             num_workers=args.workers, pin_memory=True)

    # ---- 子集划分 ----
    src = {}
    if osp.exists(args.source_map):
        for line in open(args.source_map):
            line = line.rstrip('\n')
            if not line:
                continue
            n, s = line.rsplit('\t', 1)
            src[n] = s
    sources = sorted(set(src.values())) if src else []
    keys = ['ALL'] + sources

    # ---- 待评测臂 ----
    runs = []
    for d in sorted(os.listdir(args.save_root)):
        full = osp.join(args.save_root, d)
        if not osp.isdir(full) or d.startswith('_'):
            continue
        if not osp.exists(osp.join(full, 'weight.pkl')):
            continue
        tag = tag_of(d)
        if args.tags and tag not in args.tags:
            continue
        runs.append((tag, full))
    if not runs:
        print('no weight.pkl found')
        return

    results = {}
    for tag, run_dir in runs:
        cfg = build_model_from_dir(run_dir, args.log_dir)
        model = MSHNet(3, use_wkv=cfg['wkv'], use_ca=cfg['ca'],
                       use_mspconv=(cfg['conv_type'] == 'mspconv'),
                       use_pconv=(cfg['conv_type'] == 'pconv'),
                       msp_ks=cfg['msp_ks'], msp_gate=cfg['msp_gate']).to(device)
        sd = torch.load(osp.join(run_dir, 'weight.pkl'), map_location='cpu', weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        miou = {k: mIoU(1) for k in keys}
        pdfa = {k: PD_FA(1, 10, args.base_size) for k in keys}
        cnt = {k: 0 for k in keys}
        # 额外分析维度：目标面积分箱 / 单目标 vs 多目标（逐图 IoU，非像素池化）
        bins = {'area<=81': [], '81<area<=289': [], 'area>289': []}
        shots = {'single': [], 'multi': []}
        img_iou_all = []
        nparam = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            for idx, (data, mask) in enumerate(loader):
                data, mask = data.to(device), mask.to(device)
                _, pred = model(data, False)
                name = names[idx]
                k_list = ['ALL'] + ([src[name]] if name in src else [])
                for k in k_list:
                    miou[k].update(pred, mask)
                    pdfa[k].update(pred, mask)
                    cnt[k] += 1

                # ---- 逐图分析：不受大目标图主导的口径 ----
                pb = (torch.sigmoid(pred) > 0.5).float()
                tb = (mask > 0.5).float()
                area = float(tb.sum().item())
                inter = float((pb * tb).sum().item())
                union = float(((pb + tb) > 0).float().sum().item())
                i1 = inter / (union + 1e-8)
                img_iou_all.append(i1)
                key = ('area<=81' if area <= 81 else
                       '81<area<=289' if area <= 289 else 'area>289')
                bins[key].append(i1)
                ncomp = int(ndimage.label(tb[0, 0].cpu().numpy() > 0.5,
                                          structure=_S8)[1])
                shots['multi' if ncomp > 1 else 'single'].append(i1)

        def mean(x):
            return round(sum(x) / len(x) * 100, 2) if x else None

        extra = dict(
            iou_imgmean=round(sum(img_iou_all) / len(img_iou_all) * 100, 2),
            bins={k: dict(iou=mean(v), n=len(v)) for k, v in bins.items()},
            shots={k: dict(iou=mean(v), n=len(v)) for k, v in shots.items()},
        )

        row = {}
        for k in keys:
            if cnt[k] == 0:
                continue
            _, mi = miou[k].get()
            fa, pd = pdfa[k].get(cnt[k])
            row[k] = dict(iou=round(float(mi) * 100, 2),
                          pd=round(float(pd[0]) * 100, 2),
                          fa=round(float(fa[0]) * 1e6, 2),
                          n=cnt[k])
        results[tag] = dict(params_m=round(nparam / 1e6, 3), cfg=cfg,
                            metrics=row, extra=extra)
        print(f"[{tag}] conv={cfg['conv_type']} wkv={cfg['wkv']} "
              f"ALL IoU={row['ALL']['iou']:.2f} Pd={row['ALL']['pd']:.2f} Fa={row['ALL']['fa']:.2f}", flush=True)

    # ---- 输出 ----
    order = ['ALL'] + sources
    lines = []
    lines.append('# SIRST3 评测（合并 + 三子集）')
    lines.append(f'> IoU 为该口径下**像素池化**值（Σinter/Σunion，大目标图权重更高）；'
                 f'IoU(img) 为**逐图求 IoU 再平均**，不受目标面积影响。')
    lines.append('')
    hdr = '| 实验臂 | 参数量(M) | ' + ' | '.join(
        f'{k} IoU' for k in order) + ' | IoU(img) | ' + ' | '.join(
        f'{k} Pd | {k} Fa' for k in order) + ' |'
    lines.append(hdr)
    lines.append('|' + '---|' * (3 + len(order) + 2 * len(order)))
    for tag in sorted(results, key=lambda t: -results[t]['metrics'].get('ALL', {}).get('iou', 0)):
        r = results[tag]
        m = r['metrics']
        cells = [f"{m[k]['iou']:.2f}" if k in m else '-' for k in order]
        pdfas = []
        for k in order:
            pdfas += [f"{m[k]['pd']:.2f}" if k in m else '-',
                      f"{m[k]['fa']:.2f}" if k in m else '-']
        lines.append(f"| {tag} | {r['params_m']:.2f} | " + ' | '.join(cells) + ' | '
                     + f"{r['extra']['iou_imgmean']:.2f} | " + ' | '.join(pdfas) + ' |')

    # ---- 表2：按目标面积分箱（改进A的机制证据：多尺度风车应当在大目标上收益更明显）----
    bkeys = ['area<=81', '81<area<=289', 'area>289']
    lines.append('')
    lines.append('## 表2 按 GT 目标面积分箱（逐图 IoU 均值, %）')
    lines.append('| 实验臂 | ' + ' | '.join(f'{k} (n)' for k in bkeys) + ' |')
    lines.append('|' + '---|' * (1 + len(bkeys)))
    for tag in sorted(results, key=lambda t: -results[t]['metrics'].get('ALL', {}).get('iou', 0)):
        b = results[tag]['extra']['bins']
        row = [f"{b[k]['iou']:.2f} ({b[k]['n']})" if b[k]['iou'] is not None else '-'
               for k in bkeys]
        lines.append(f"| {tag} | " + ' | '.join(row) + ' |')

    # ---- 表3：单目标 vs 多目标（改进B的机制证据）----
    lines.append('')
    lines.append('## 表3 单目标 / 多目标图像（逐图 IoU 均值, %）')
    lines.append('| 实验臂 | single (n) | multi (n) | multi 相对收益 |')
    lines.append('|' + '---|' * 4)
    for tag in sorted(results, key=lambda t: -results[t]['metrics'].get('ALL', {}).get('iou', 0)):
        s = results[tag]['extra']['shots']
        gain = (s['multi']['iou'] - s['single']['iou']) if (
            s['multi']['iou'] is not None and s['single']['iou'] is not None) else None
        lines.append(f"| {tag} | {s['single']['iou']:.2f} ({s['single']['n']}) | "
                     f"{s['multi']['iou']:.2f} ({s['multi']['n']}) | "
                     f"{gain:+.2f} |" if gain is not None else f"| {tag} | - | - | - |")

    md = '\n'.join(lines)
    out_md = args.out or osp.join(args.save_root, '_eval_sirst3.md')
    with open(out_md, 'w') as f:
        f.write(md + '\n')
    with open(osp.join(args.save_root, '_eval_sirst3.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\n' + md)
    print(f'\nwrote {out_md}')


if __name__ == '__main__':
    main()
