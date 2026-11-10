# Component-Windowed Supervision for Infrared Small Target Detection

Official PyTorch implementation of **MSPC-Net**, the network proposed in the
manuscript *"Component-Windowed Supervision for Infrared Small Target
Detection"*, built upon the MSHNet backbone (CVPR 2024). Conventional pinwheel
convolution uses a single arm length within each layer, while image-level
centroid supervision makes localization depend on predictions from other targets
and background regions. We address both limitations with a structure-side and a
loss-side improvement that can be enabled **independently**:

- **Improvement A — MS-PConv (structure side)**: parallel pinwheel branches with
  arm lengths **k=3 and k=5** provide two spatial extents with the *same
  strip-convolution weight budget* as the original arm-length-4 module — the
  parameter count is strictly equal (2.77M vs 4.07M baseline).
- **Improvement B — CW-SDM (loss side)**: a **component-windowed** loss computes
  the location and coverage terms around each ground-truth connected component
  (square window of side `2*r_k`, `r_k = max(12, 4*sqrt(a_k))`), limiting the
  spatial scope of those terms, and adds an explicit **background penalty** `L_bg`
  (BCE on non-target pixels) across the image.

Key results on merged SIRST3 (best epoch, 400-epoch protocol), matching the
paper: IoU 69.15% -> **76.27%** and false-alarm rate 26.63 -> **13.83**×10⁻⁶
relative to MSHNet, using **2.77M** parameters (~5.5% of UIU-Net's 50.5M). The
loss alone achieves higher IoU (**77.74%**); the complete model has fewer
parameters and fewer false alarms. On IRSTD-1k, IoU falls and false alarms rise —
reported as-is, consistent with the paper.

## 1. Environment

```
pip install -r requirements.txt
```

Python >= 3.8, PyTorch >= 1.13 (CUDA), torchvision, numpy, scipy, scikit-image, tqdm, Pillow, opencv-python.

## 2. Dataset layout

Each dataset directory follows the MSHNet convention:

```
datasets/SIRST3/
├── trainval.txt      # training image name list (1676)
├── test.txt          # test image name list (1079)
├── images/           # infrared images
└── masks/            # binary GT masks
```

The train/test splits used in the paper are provided in `datasets/` for **SIRST3 (merged)**, **NUAA-SIRST**, **NUDT-SIRST** and **IRSTD-1k**; download the images/masks of NUAA-SIRST / NUDT-SIRST / IRSTD-1k from their official sources and place them under the corresponding folder. SIRST3 is the merged training protocol described in the paper.

`datasets/SIRST3_source_map.txt` maps every merged-SIRST3 image name to its source dataset, which is what `eval_sirst3.py` uses to report the per-subset columns of the paper.

Total counts of the merged split: 1676 training / 1079 test images, made of NUAA-SIRST (213 / 214), NUDT-SIRST (663 / 664) and IRSTD-1k (800 / 201).

## 3. Training

All arms share the same entry point `main_train.py` (400 epochs, eval every 10 epochs, best-epoch checkpoint reported in the paper):

```bash
# MSHNet baseline
python main_train.py --tag base --dataset-dir datasets/SIRST3 --save-root runs --epochs 400 --seed 42

# Improvement A only (MS-PConv)
python main_train.py --tag msp --dataset-dir datasets/SIRST3 --save-root runs --epochs 400 --seed 42 \
    --mspconv --msp-ks 3,5

# Improvement B only (CW-SDM, bg weight 16)
python main_train.py --tag sdcomp_bg16 --dataset-dir datasets/SIRST3 --save-root runs --epochs 400 --seed 42 \
    --sdcomp --bg-weight 16

# Full model (A + B) = MSPC-Net
python main_train.py --tag msp_sdcomp_bg16 --dataset-dir datasets/SIRST3 --save-root runs --epochs 400 --seed 42 \
    --mspconv --msp-ks 3,5 --sdcomp --bg-weight 16

# Family control: AAAI'25 original configuration (PConv + SDM)
python main_train.py --tag pconv_sd --dataset-dir datasets/SIRST3 --save-root runs --epochs 400 --seed 42 \
    --pconv --sdloss
```

Important switches:

| Flag | Meaning |
| --- | --- |
| `--mspconv --msp-ks 3,5` | Improvement A: dual-scale pinwheel convolution (k=3 + k=5, 8 arms, equal params to PConv) |
| `--sdcomp` | Improvement B: component-windowed SDM, CW-SDM (`r_k = max(min-radius, 4*sqrt(area))`) |
| `--bg-weight 16` | Improvement B: explicit background BCE weight (paper uses 16; 8 works similarly — insensitive) |
| `--cov-weight 0.5` | Local coverage term weight of CW-SDM |
| `--sdloss` | Reproduce the original image-level SDM (AAAI'25) |
| `--acsl` | Reproduce AC-SLSIoU (comparison method) |
| `--seed` | -1 = framework default initialization; set for reproducibility |

Checkpoints and logs are written to `runs/<tag>-<timestamp>/`: `weight.pkl` (best-IoU epoch, used for all evaluation), `checkpoint.pkl` (last epoch, for `--resume`), `metrics.jsonl` (per-epoch metrics) and `summary.json` (full configuration, read back by the evaluation scripts).

## 4. Evaluation

`eval_sirst3.py` re-evaluates trained arms with the unified single-pass protocol (merged SIRST3 + three subset protocols in one traversal; it also fixes the `PD_FA.reset()` upstream bug by instantiating a fresh metric per protocol):

```bash
python eval_sirst3.py --dataset-dir datasets/SIRST3 --save-root runs
python eval_sirst3.py --tags base msp sdcomp_bg16 msp_sdcomp_bg16   # selected arms only
```

Outputs `runs/_eval_sirst3.md` (paper-ready table) and `runs/_eval_sirst3.json` (per-image details). Besides the merged-protocol metrics, the markdown also reports the **per-image IoU** (mean over images instead of pixel pooling) and the two stratifications used in the paper: target-area bins (`<=81`, `81-289`, `>289` px² on the total target area of each image) and single- versus multi-target images.

Model complexity (parameters / FLOPs / latency) of the selected arms:

```bash
python eval_complexity.py --tags base pconv msp msp_sdcomp_bg16
```

Traditional baselines (Top-Hat / IPI, training-free) can be reproduced with:

```bash
python eval_traditional.py --dataset <SIRST3|NUAA> --methods tophat,ipi --workers 8
```

Note: multi-process evaluation should be launched with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` to avoid BLAS thread over-subscription.

## 5. Main results (merged SIRST3, best-epoch)

| Config | Params(M) | IoU(%) | Pd(%) | Fa(x1e-6) |
| --- | --- | --- | --- | --- |
| MSHNet baseline | 4.07 | 69.15 | 90.48 | 26.63 |
| + A (MS-PConv) | 2.77 | 72.82 | 95.81 | 26.33 |
| + B (CW-SDM) | 4.07 | 77.74 | 96.87 | 18.34 |
| **A + B (MSPC-Net)** | **2.77** | **76.27** | 96.67 | **13.83** |
| PConv+SDM (AAAI'25 cfg) | 2.77 | 70.05 | 93.54 | 40.13 |

## 6. Code structure

```
MSPC-Net-main/
├── main_train.py            # training entry (all arms / all switches)
├── eval_sirst3.py           # unified evaluation (merged + subsets + area/multi-target strata)
├── eval_complexity.py       # params / FLOPs / latency of trained arms
├── eval_traditional.py      # training-free baselines: Top-Hat / IPI
├── model/
│   ├── MSHNet.py            # backbone with pluggable conv & loss (CBAM ResBlocks, deep supervision)
│   ├── mspconv.py           # Improvement A: multi-scale pinwheel convolution (k=3/k=5)
│   ├── pconv.py             # original pinwheel convolution (baseline / family control)
│   ├── loss_sd_comp.py      # Improvement B: CW-SDM (L_loc / L_cov / L_bg)
│   ├── loss_sd.py           # original SDM loss (AAAI'25 reproduction)
│   ├── loss.py              # SLSIoU losses of MSHNet
│   ├── loss_acsl.py         # AC-SLSIoU reproduction (comparison method)
│   ├── coordatt.py          # coordinate attention used by ResBlocks
│   └── wkv.py               # optional WKV variant (unused in the paper)
├── utils/
│   ├── data.py              # IRSTD_Dataset (256x256, ImageNet-normalized)
│   └── metric.py            # IoU / Pd / Fa metrics
├── datasets/                # train/test splits for the four benchmarks + SIRST3 source map
├── Fig/                     # paper figures (architecture, MS-PConv, CW-SDM)
└── requirements.txt
```

Paper tables map to the scripts as follows:

| Paper table | Script | Notes |
| --- | --- | --- |
| Table 3 (ablation, merged SIRST3) | `main_train.py` + `eval_sirst3.py` | best-epoch checkpoint of each arm |
| Table 4 (per-subset results) | `main_train.py` + `eval_sirst3.py` | subset columns driven by `SIRST3_source_map.txt` |
| Table 5 (area / multi-target strata) | `eval_sirst3.py` | `bins` and `shots` blocks of the output |
| Params / FLOPs columns (Tables 2 and 3) | `eval_complexity.py` | params, FLOPs and latency |

## 7. Acknowledgements

This code is built on [MSHNet](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Pinwheel-shaped_Convolution_and_Scale-based_Dynamic_Loss_for_Infrared_Small_Target_ICVPR_2024_paper.html) (CVPR 2024) and the SDM loss from the AAAI 2025 follow-up. We thank the authors for releasing their code.
