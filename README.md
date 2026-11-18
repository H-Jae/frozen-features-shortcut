# Shortcut learning: linear probing vs. full fine-tuning

Code for our XAI course project proposal, "Does Freezing Pretrained Features Mitigate Shortcut Learning?"

Main question: on the same biased training data, does freezing a pretrained backbone (linear probing, LP) reduce reliance on a shortcut patch compared with full fine-tuning (FT)? Training from scratch is included as a reference.

The setup is CIFAR-10 with a 24x24 color patch in the bottom-right corner. Each class has its own color, and a training image gets the color of its true class with probability p (0.10, 0.90 or 0.99). We trained ResNet-18 with three strategies (LP, FT, Scratch) at three values of p with three seeds, 27 models in total.

The numbers are in `results/summary.json` and `results/per_model.csv`, and the figures are in `results/figures/`.

## What is in the repository

Included: the code, `results/` (metrics and figures), and from `runs/` the configs, training logs (json) and evaluation outputs (npz files with test logits and IG attributions). With these you can run `python analyze.py` without a GPU and recompute every metric.

Not included: the CIFAR-10 files (`data/`), the checkpoints (`runs/**/best.pt`, about 1.5 GB in total) and the written reports (`docs/` is git-ignored). You need checkpoints for `ig_cases.py` and for any new evaluation or IG run. Ask Hyeongjae for them or retrain with `driver.py` (training alone took 4.8 GPU hours). A run whose folder still has `done.json` is skipped, so delete that run's folder before retraining it.

## Environment

- We ran on an RTX 5070 12GB, CUDA 13.4 driver, Python 3.12, Windows.
- Training, evaluation and IG need `torch==2.14.1+cu130` and `torchvision==0.29.1+cu130`
  (`pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 torchvision==0.29.1`).
  Everything else came in as their dependencies; exact versions are in `requirements-lock.txt`. Pick the torch build that matches your CUDA version.
- Analysis (`analyze.py`, `report.py`) needs numpy, scipy and matplotlib, and no torch. `report.py` also uses Pillow, which matplotlib installs.
- Data: CIFAR-10 binary version, `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`,
  md5 `c32a1d4ab5d03f1284b67883e8d87530` (matches the official value). Extract it to `data/raw/cifar-10-batches-bin/`.
  We chose the binary version because it does not use pickle.
- Pretrained weights: torchvision `ResNet18_Weights.IMAGENET1K_V1` (`resnet18-f37072fd.pth`, hash-checked on download).

## How to run

Run from the repository folder. The first seven commands need torch. The last two need only numpy, scipy and matplotlib.

```
python driver.py tune       # 6 settings per strategy, p=0.10, seed 100
python driver.py select     # writes results/tuned.json
python driver.py final      # 27 training runs + evaluation + IG + precision check
python driver.py lplong     # follow-up: LP at p=0.99 for 60 epochs with weight decay 0 (evaluation only)
python ig_cases.py          # cases where IG and behavior disagree (GPU, after final)
python export_meta.py       # results/test_meta.npz (labels and draws used by the analysis)
python figure_inputs.py     # results/figures/fig0_inputs.png (input examples, CPU)
python analyze.py           # results/summary.json, per_model.csv, figures/
python report.py            # docs/report.html (the report text is in Korean)
```

Finished runs are skipped, so if a command stops halfway you can run it again and it continues.

## Files

| File | What it does | Proposal section |
|---|---|---|
| `common.py` | Reads CIFAR-10, fixed split, patch sampling (Eq. 1), color palette, GPU batching, models | 3.1 |
| `train.py` | One training run; picks the checkpoint by ID validation accuracy | 3.2 |
| `evaluate.py` | Saves logits for the 24 test variants | 4.1, 4.3 |
| `ig.py` | IG, R_patch, completeness residual, 128-step convergence check, IG by patch size, heatmaps | 4.2 |
| `driver.py` | Runs the stages (tune / select / final / lplong) | 5 |
| `ig_cases.py` | Heatmaps for six images per model where IG and measured sensitivity disagree (selection rule fixed before looking at results) | 4.2 |
| `analyze.py` | Metrics, seed mean and standard deviation, paired image bootstrap, Spearman correlation, figures | 5 |
| `report.py` | Builds the HTML report; numbers, tables and figures come from summary.json | 5 |
| `report_text.py` | Interpretation text for the report (Korean). Each directional claim has a data check (`need`), and `report.py` stops if the data disagrees | 5 |

## Design decisions the proposal left open

- Patch sampling is coupled across p. For each image we draw u ~ U(0,1) and a wrong color w (uniform over the other nine) once, then set `s = y if u < p else w`. Within a seed the patches are the same for every strategy, and the images with the correct color at p=0.90 are a subset of those at p=0.99.
- The wrong color w and the draw u for test images are the same for all models. ID accuracy (training p) and independent OOD accuracy (p=0.10) use the prediction on x+ when u < p and on x- otherwise. Conflicting OOD is x- for every image.
- Hyperparameters: each strategy gets a grid of the same size (3 learning rates x 2 weight decays), trained at p=0.10 with a tuning-only seed (100) and selected by ID validation accuracy. The chosen values are then fixed for every p and for seeds 0, 1, 2. Tuning per p would mix the effect of p (SQ1) with the effect of tuning.
  The first LP grid was set low because in a smoke run the loss jumped to 3.9 at lr 0.1 (ImageNet features have norm about 28). After widening the grid twice under the rule below, 0.1 was best in the final setup (warmup + cosine, 15 epochs), 0.18 points above 0.03.
- Grid-edge rule: if the best learning rate is at the edge of the grid, add one more step in that direction (`driver.LADDER`) and run both weight decays. The rule applies to every strategy, at most twice. We added it during tuning after the best learning rates for LP and FT landed on the upper edge (0.01 and 0.03), so that an undertrained model would not be read as robust to the shortcut. The number of settings can therefore differ between strategies.
- Epochs: LP 15, FT 10, Scratch 40. SGD (momentum 0.9, Nesterov), batch 256, one warmup epoch then cosine decay. Training uses bf16 autocast.
- Precision: test evaluation uses IEEE fp32 including convolutions (TF32, the cuDNN default, is turned off). IG uses fp32 tensors with TF32 convolutions, because IEEE is about 2.5 times slower. We measured the difference on the seed 0 models (6 models, the same 200 images computed both ways, `ig_precision.npz`).
- Completion markers: `done.json` for training, `eval_done.json` for evaluation, `ig_done.json` for IG. Without the marker a step is recomputed even if its npz exists, since an interrupted save can leave a partial file. Calling a finished run again with a different lr or weight decay raises an error.
- Upscaling is bilinear. Random crop uses reflection padding, because black is one of the patch colors and zero padding would add a black border.
- Occlusion control: a gray square of the same area (24x24) placed over x+ at the center or the top-left.
- Object blur: Gaussian sigma = 4, 8, 16 at 224px. The blurred IG baseline uses sigma = 16.
- Follow-up `lplong` (2026-11-02, decided after seeing the main results): LP at p=0.99 stopped at 96.2% training accuracy. To check whether LP used the patch less because the head was undertrained, we retrained the head with the same seeds (0, 1, 2) and the same lr (0.1) for 60 epochs with weight decay 0, and ran evaluation only (`runs/lplong`, no IG). `analyze.py` keeps these three runs apart from the 27 main runs and writes them under `followup` in `summary.json`.
