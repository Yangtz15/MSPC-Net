# -*- coding: utf-8 -*-
"""参数量 / FLOPs / 单张推理延迟（batch=1, 256x256）。

用法:
    python eval_complexity.py                       # 评测 runs/ 下 base / pconv / msp 三个臂
    python eval_complexity.py --tags base msp --iters 100

输出:
    runs/_eval_complexity.json
"""
import json
import os
import os.path as osp
import sys
import time
from argparse import ArgumentParser

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from eval_sirst3 import build_model_from_dir  # noqa: E402
from model.MSHNet import MSHNet               # noqa: E402


def parse_args():
    p = ArgumentParser()
    p.add_argument('--save-root', type=str, default='runs')
    p.add_argument('--log-dir', type=str, default='runs')
    p.add_argument('--tags', nargs='*', default=['base', 'pconv', 'msp'])
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--out', type=str, default='')
    return p.parse_args()


def latest_run(save_root, tag):
    dirs = sorted(d for d in os.listdir(save_root)
                  if d.startswith(tag + '-') and osp.isdir(osp.join(save_root, d)))
    if not dirs:
        raise FileNotFoundError(f'no run directory found for tag "{tag}" under {save_root}')
    return osp.join(save_root, dirs[-1])


def load_model(run_dir, log_dir, device):
    cfg = build_model_from_dir(run_dir, log_dir)
    model = MSHNet(3, use_wkv=cfg['wkv'], use_ca=cfg['ca'],
                   use_mspconv=(cfg['conv_type'] == 'mspconv'),
                   use_pconv=(cfg['conv_type'] == 'pconv'),
                   msp_ks=cfg['msp_ks'], msp_gate=cfg['msp_gate']).to(device)
    state = torch.load(osp.join(run_dir, 'weight.pkl'), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def count_flops(model, x):
    """Forward FLOPs via torch.profiler (counts conv/matmul only)."""
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if x.is_cuda:
        acts.append(ProfilerActivity.CUDA)
    with torch.no_grad():
        with profile(activities=acts, with_flops=True) as prof:
            model(x, True)
    return sum(e.flops for e in prof.key_averages())


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, args.size, args.size, device=device)

    out = {}
    for tag in args.tags:
        run_dir = latest_run(args.save_root, tag)
        model = load_model(run_dir, args.log_dir, device)

        n_params = sum(p.numel() for p in model.parameters())
        flops = count_flops(model, x)

        with torch.no_grad():
            for _ in range(args.warmup):
                model(x, True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                model(x, True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latency = (time.time() - t0) / args.iters * 1000

        out[tag] = dict(run_dir=osp.basename(run_dir),
                        params_m=round(n_params / 1e6, 2),
                        gflops=round(flops / 1e9, 2) if flops else None,
                        latency_ms=round(latency, 2))
        print(f'[{tag}] {out[tag]["run_dir"]}: params {n_params / 1e6:.2f}M  '
              f'FLOPs {out[tag]["gflops"]}G  latency {latency:.2f} ms', flush=True)

    out_path = args.out or osp.join(args.save_root, '_eval_complexity.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved ->', out_path)


if __name__ == '__main__':
    main()
