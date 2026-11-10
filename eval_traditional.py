#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-Hat 与 IPI 传统方法在 SIRST3/NUAA 测试集上的评测（免训练）。
统一协议：256x256 resize（与深度模型一致）；指标 = 像素 IoU + 官方目标级 Pd/Fa。
阈值策略：对显著图做 mean+k*std 扫描（k=1.5~3.5），报告 best-IoU 工作点。
用法: python eval_traditional.py --dataset both --workers 8
"""
import argparse, json, os, warnings
import numpy as np
import cv2
from multiprocessing import Pool

warnings.filterwarnings("ignore")

DATASETS = {
    "SIRST3": dict(root="datasets/SIRST3"),
    "NUAA":   dict(root="datasets/NUAA-SIRST"),
}
SIZE = 256
PATCH, STRIDE = 50, 10
KS = [1.5, 2.0, 2.5, 3.0, 3.5]


def load_list(root):
    ids = [l.strip() for l in open(os.path.join(root, "test.txt")) if l.strip()]
    return ids


def load_pair(root, name):
    img = cv2.imread(os.path.join(root, "images", name + ".png"), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(root, "masks", name + ".png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(os.path.join(root, "images", name + ".bmp"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(root, "masks", name + ".bmp"), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE)).astype(np.float32)
    if msk is not None:
        msk = cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        msk = np.zeros((SIZE, SIZE), np.uint8)
    return img, (msk > 127).astype(np.uint8)


def disk_kernel(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.uint8)


def tophat_map(img):
    k = disk_kernel(7)  # 15x15 圆盘，经典设置
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k).astype(np.float32)


def soft_threshold(M, t):
    return np.sign(M) * np.maximum(np.abs(M) - t, 0)


def rpca_ialm(M, lmbda=0.02, max_iter=60, tol=1e-5):
    """RPCA via IALM（IPOL 标准实现），返回稀疏分量 S"""
    m, n = M.shape
    normM = np.linalg.norm(M, "fro") + 1e-12
    Y = M / max(normM * 1e-3, 1e-12)
    mu = 1.25 / (np.linalg.norm(M, 2) + 1e-12)
    mu_bar = mu * 1e6
    S = np.zeros_like(M)
    for _ in range(max_iter):
        A = M - S + Y / mu
        # economy SVD（2500x441 规模直接算足够快）
        try:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        s_t = np.maximum(s - 1.0 / mu, 0)
        L = (U * s_t) @ Vt
        S = soft_threshold(M - L, lmbda / mu)
        Y = Y + mu * (M - L - S)
        mu = min(mu * 1.5, mu_bar)
        if np.linalg.norm(M - L - S, "fro") / normM < tol:
            break
    return np.abs(S)


def ipi_map(img):
    """IPI: 滑窗切块 -> RPCA -> 稀疏分量 max 投影"""
    h, w = SIZE, SIZE
    patch_list, positions = [], []
    for i in range(0, h - PATCH + 1, STRIDE):
        for j in range(0, w - PATCH + 1, STRIDE):
            p = img[i:i + PATCH, j:j + PATCH].flatten()
            patch_list.append(p - p.mean())
            positions.append((i, j))
    Mmat = np.stack(patch_list, axis=1)  # (2500, N)
    S = rpca_ialm(Mmat)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32) + 1e-6
    for idx, (i, j) in enumerate(positions):
        acc[i:i + PATCH, j:j + PATCH] += S[:, idx].reshape(PATCH, PATCH)
        cnt[i:i + PATCH, j:j + PATCH] += 1
    return acc / cnt


def components(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        comps.append(dict(area=len(ys), cy=float(ys.mean()), cx=float(xs.mean())))
    return comps


def metrics(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = inter / union if union else 1.0
    gt_c = components(gt)
    pr_c = components(pred)
    matched_gt = set()
    fa_area = 0
    for pc in pr_c:
        hit = False
        for gi, gc in enumerate(gt_c):
            if gi in matched_gt:
                continue
            if (pc["cy"] - gc["cy"]) ** 2 + (pc["cx"] - gc["cx"]) ** 2 <= 9.0:  # 3px
                matched_gt.add(gi); hit = True; break
        if not hit:
            fa_area += pc["area"]
    pd = len(matched_gt) / len(gt_c) if gt_c else 1.0
    fa = fa_area / (SIZE * SIZE)  # 每图比例，报告时 x1e6
    return iou, pd * 100, fa * 1e6


def eval_one(args):
    root, name, method = args
    img, gt = load_pair(root, name)
    sal = tophat_map(img) if method == "tophat" else ipi_map(img)
    mu, sd = float(sal.mean()), float(sal.std())
    out = {}
    for k in KS:
        pred = (sal > mu + k * sd).astype(np.uint8)
        out[k] = metrics(pred, gt)
    return name, out


def run_dataset(name, method, workers):
    root = DATASETS[name]["root"]
    ids = load_list(root)
    tasks = [(root, i, method) for i in ids]
    with Pool(workers) as pool:
        results = pool.map(eval_one, tasks)
    # 聚合：每个 k 求全测试集均值
    agg = {}
    for k in KS:
        ious = [r[1][k][0] for r in results]
        pds = [r[1][k][1] for r in results]
        fas = [r[1][k][2] for r in results]
        agg[k] = (float(np.mean(ious)), float(np.mean(pds)), float(np.mean(fas)))
    best_k = max(KS, key=lambda k: agg[k][0])
    return dict(dataset=name, method=method, n_test=len(ids), best_k=best_k,
                iou=agg[best_k][0], pd=agg[best_k][1], fa=agg[best_k][2],
                sweep={str(k): agg[k] for k in KS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["SIRST3", "NUAA", "both"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", default="tophat,ipi")
    ap.add_argument("--out", default="runs/traditional_results.json")
    a = ap.parse_args()
    names = ["SIRST3", "NUAA"] if a.dataset == "both" else [a.dataset]
    out = []
    for ds in names:
        for m in a.methods.split(","):
            r = run_dataset(ds, m, a.workers)
            out.append(r)
            print(f"[{ds}] {m.upper()}: IoU={r['iou']:.2f} Pd={r['pd']:.2f} "
                  f"Fa={r['fa']:.2f} (best k={r['best_k']}, n_test={r['n_test']})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {a.out}")
