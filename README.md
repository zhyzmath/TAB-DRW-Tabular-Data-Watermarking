# TAB-DRW

TAB-DRW is a DFT-based robust watermark for generative tabular data. This repository provides an end-to-end pipeline built on TabSyn, including training, watermark embedding, detection, attacks, and evaluation.

Implementation is based on [TabWak](https://github.com/chaoyitud/TabWak).

## Highlights
- TabSyn VAE + diffusion training for tabular generation
- Watermark embedding during sampling and post-editing on generated tables
- Detection with optional attack simulation
- Evaluation scripts for density, detection, MLE, DCR, and quality metrics

## Table of Contents
- [TAB-DRW](#tab-drw)
  - [Highlights](#highlights)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [1. Create environment](#1-create-environment)
    - [2. Install PyTorch](#2-install-pytorch)
    - [3. Install dependencies](#3-install-dependencies)
    - [Optional: quality metrics (synthcity)](#optional-quality-metrics-synthcity)
  - [Datasets](#datasets)
    - [Add a custom dataset](#add-a-custom-dataset)
  - [Quickstart](#quickstart)
  - [Training](#training)
  - [Sampling and Watermarking](#sampling-and-watermarking)
  - [Watermark Detection](#watermark-detection)
  - [Attacks](#attacks)
  - [Evaluation](#evaluation)
  - [Outputs](#outputs)

## Installation

**Python**: 3.10

### 1. Create environment
```bash
conda create -n tabsyn python=3.10
conda activate tabsyn
```

### 2. Install PyTorch
Using `pip`:
```bash
pip install torch torchvision torchaudio
```

Or via `conda` (CUDA 11.7):
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install wandb
```

`wandb` is imported in training and sampling scripts; install it or remove the imports if you do not plan to log.

### Optional: quality metrics (synthcity)
`synthcity` is only needed for `eval/eval_quality.py`. We recommend a separate environment.
```bash
conda create -n synthcity python=3.10
conda activate synthcity
pip install synthcity category_encoders
```

## Datasets

Supported datasets from the paper:
`adult`, `default`, `magic`, `shoppers`, `beijing`, `news`, `drybean`.

If the processed datasets are already present under `data/<name>/` (e.g., `info.json`, `train.csv`, `test.csv`, and `*.npy` files), you can skip the download and preprocessing steps below.

Download and preprocess:
```bash
python download_dataset.py
python process_dataset.py
```

Process a single dataset:
```bash
python process_dataset.py --dataname adult
```

### Add a custom dataset
1. Place the raw file under `data/<name>/`.
2. Create `data/Info/<name>.json` (see existing files) with required fields such as `task_type`, `file_type`, `data_path`, `header`, `num_col_idx`, `cat_col_idx`, `target_col_idx`, and `dis_col_idx`.
3. Run `python process_dataset.py --dataname <name>` to generate `data/<name>/info.json`.
4. For `TAB-DRW`, `GLW`, `tabmark`, and `muse`, ensure `data/<name>/info.json` includes `key_col_idx`, `value_col_idx`, and `continuous_col_idx` (used for watermark column selection). You can add them manually or via `watermark_tool/tool.py:add_fields_to_info_json`.

## Quickstart
```bash
python download_dataset.py
python process_dataset.py

python main.py --dataname adult --method vae --mode train
python main.py --dataname adult --method tabsyn --mode train

python main.py --dataname adult --method tabsyn --mode sample --steps 1000 --with_w TAB-DRW --num_samples 5000
python main.py --dataname adult --method tabsyn --mode detect --steps 1000 --with_w TAB-DRW --num_samples 5000
```

## Training

1. Train the VAE:
```bash
python main.py --dataname [DATASET] --method vae --mode train
```

2. Train the TabSyn diffusion model:
```bash
python main.py --dataname [DATASET] --method tabsyn --mode train
```

Notes:
- Supported `--method` values: `vae`, `tabsyn`.
- `--gpu` selects the CUDA device index when CUDA is available.
- `--save_path` overrides the default output directory `synthetic/<dataname>`.

## Sampling and Watermarking

Generate synthetic data without watermark:
```bash
python main.py --dataname [DATASET] --method tabsyn --mode sample --steps 1000 --num_samples [NUM_SAMPLES]
```

Watermark during sampling (latent-space):
```bash
python main.py --dataname [DATASET] --method tabsyn --mode sample --steps 1000 --with_w treering --num_samples [NUM_SAMPLES]
```

Post-editing watermark on generated tables:
```bash
python main.py --dataname [DATASET] --method tabsyn --mode sample --steps 1000 --with_w TAB-DRW --num_samples [NUM_SAMPLES]
```

Watermark options:
- Latent-space: `treering`, `GS`, `TabWak`, `TabWak_star`
- Post-editing/sampling: `TAB-DRW`, `GLW`, `tabmark`, `muse`
- No watermark: `--with_w none` (default)

Notes:
- `--num_samples -1` uses the training set size.
- `--steps` controls DDIM sampling steps (default 50).
- For post-editing methods, rerunning with the same `--with_w` reuses saved diffusion outputs if available.
- Sampling runs three seeds (0, 1, 2) by default and stores outputs under those subfolders.

### muse prerequisite: `real_raw.csv`
The `muse` watermark requires `synthetic/<dataname>/real_raw.csv`. Generate it from `real.csv` after dataset preprocessing:
```bash
python synthetic/convert_real_to_raw.py adult
```

You can pass multiple dataset names, or omit args to use the script defaults.

## Watermark Detection

```bash
python main.py --dataname [DATASET] --method tabsyn --mode detect --steps 1000 --with_w [WATERMARK] --num_samples [NUM_SAMPLES]
```

Use the same `--steps` and `--num_samples` you used during sampling. Detection expects the generated files under `synthetic/<dataname>`.

## Attacks

Run detection with attacks:
```bash
python main.py --dataname [DATASET] --method tabsyn --mode detect --steps 1000 --with_w [WATERMARK] --num_samples [NUM_SAMPLES] --attack [ATTACK] --attack_percentage [0-1]
```

Supported attacks:
`rowdeletion`, `coldeletion`, `celldeletion`, `noise`, `shuffle`, `catsubstitution`, `precision`, `stratified_sampling`, `quantization`, `gaussian_noise_standardized`.

## Evaluation

Density and detection (sdmetrics):
```bash
python eval/eval_density.py --dataname [DATASET] --model tabsyn --path [SYN_CSV]
python eval/eval_detection.py --dataname [DATASET] --model tabsyn --path [SYN_CSV]
```

Downstream utility (MLE):
```bash
python eval/eval_mle.py --dataname [DATASET] --model tabsyn --path [SYN_CSV]
```

Distance to closest record (DCR):
```bash
python eval/eval_dcr.py --dataname [DATASET] --model tabsyn --path [SYN_CSV]
```

Quality metrics (synthcity, optional):
```bash
python eval/eval_quality.py --dataname [DATASET] --model tabsyn --path [SYN_CSV]
```

If `--path` is omitted, the scripts default to `synthetic/<dataname>/<model>.csv`.

## Outputs

Generated artifacts are stored under `synthetic/<dataname>/`. Example structure:

```
synthetic/<dataname>/
  real.csv
  test.csv
  <watermark>/<num_samples>/<seed>/
    no-w-tabsyn.csv
    w-tabsyn.csv
    w-num-tabsyn.csv
    no-w-tabsyn-raw.csv
    w-tabsyn-raw.csv
    w-num-tabsyn-raw.csv
```

Available files depend on the watermarking method. For latent-space watermarks, additional `.npy` files are saved for latents and intermediate tensors.
