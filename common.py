# 공통 부품: CIFAR-10 읽기, 고정 분할, 색 패치, GPU 배치 만들기, 모델 만들기.
# 제출 제안서(Shortcut_Learning_Proposal.pdf) 3절의 설계를 그대로 옮긴 것이다.
# - 분할: 학습 50,000장 -> 학습 45,000 / 검증 5,000 (클래스별 4,500 / 500), 모든 실행에서 같다.
# - 패치 색 번호 s: P(s=y)=p, P(s=k!=y)=(1-p)/9. 같은 seed 안에서는 전략과 무관하게 같고,
#   p끼리도 결합(coupling)해 둔다: 이미지마다 u~U(0,1)과 틀린 색 w를 한 번 뽑고 s = y if u<p else w.
# - 순서: (학습 때만) 32x32에서 crop+flip -> 224x224 bilinear 확대 -> 패치 붙이기 -> ImageNet 정규화.
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "cifar-10-batches-bin"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# 클래스 k <-> 색 k. 중간 회색은 IG 기준 이미지와 같아져서 뺐다.
PALETTE = np.array([
    [255, 0, 0],      # 0 red
    [0, 255, 0],      # 1 green
    [0, 0, 255],      # 2 blue
    [255, 255, 0],    # 3 yellow
    [255, 0, 255],    # 4 magenta
    [0, 255, 255],    # 5 cyan
    [255, 255, 255],  # 6 white
    [0, 0, 0],        # 7 black
    [255, 128, 0],    # 8 orange
    [128, 0, 255],    # 9 purple
], dtype=np.uint8)

IMG = 224
PATCH = 24
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SPLIT_SEED = 20261023
TEST_SEED = 20261031
P_INDEP = 0.10

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_cifar10():
    """CIFAR-10 binary version. 한 줄 = 라벨 1바이트 + R/G/B 각 1024바이트."""
    def read(name):
        raw = np.fromfile(DATA_DIR / name, dtype=np.uint8).reshape(-1, 3073)
        return raw[:, 1:].reshape(-1, 3, 32, 32).copy(), raw[:, 0].astype(np.int64)
    parts = [read(f"data_batch_{i}.bin") for i in range(1, 6)]
    x_tr = np.concatenate([a for a, _ in parts])
    y_tr = np.concatenate([b for _, b in parts])
    x_te, y_te = read("test_batch.bin")
    return x_tr, y_tr, x_te, y_te


def split_indices(y_train_full):
    """층화 분할: 클래스마다 500장을 검증으로. 모든 전략·seed·p에서 같다."""
    rng = np.random.default_rng(SPLIT_SEED)
    tr, va = [], []
    for k in range(10):
        idx = np.flatnonzero(y_train_full == k)
        rng.shuffle(idx)
        va.append(idx[:500])
        tr.append(idx[500:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def coupled_draws(y, seed, salt):
    """이미지마다 u~U(0,1)과 틀린 색 w(9개 중 균등)를 한 번 뽑는다."""
    rng = np.random.default_rng([seed, salt])
    u = rng.random(len(y))
    w = (y + rng.integers(1, 10, len(y))) % 10
    return u, w


def assign_patches(y, p, u, w):
    return np.where(u < p, y, w).astype(np.int64)


class Splits:
    """한 번 읽어 GPU에 올려 둔 데이터. 패치는 배치마다 GPU에서 붙인다."""
    def __init__(self, device=DEV):
        x_full, y_full, x_te, y_te = load_cifar10()
        tr, va = split_indices(y_full)
        self.idx_tr, self.idx_va = tr, va
        self.y_tr, self.y_va, self.y_te = y_full[tr], y_full[va], y_te
        self.x_tr = torch.from_numpy(x_full[tr]).to(device)
        self.x_va = torch.from_numpy(x_full[va]).to(device)
        self.x_te = torch.from_numpy(x_te).to(device)
        # 테스트 쪽 u, w는 모든 모델에 공통이다 (짝지은 비교가 같은 입력에서 이뤄지게).
        self.u_te, self.w_te = coupled_draws(y_te, TEST_SEED, 3)

    def train_patches(self, p, seed):
        u, w = coupled_draws(self.y_tr, seed, 1)
        return assign_patches(self.y_tr, p, u, w)

    def val_patches(self, p, seed):
        u, w = coupled_draws(self.y_va, seed, 2)
        return assign_patches(self.y_va, p, u, w)


# ---------------------------------------------------------------- images
_palette_t = {}
_norm_t = {}


def palette(device):
    if device not in _palette_t:
        _palette_t[device] = torch.tensor(PALETTE, dtype=torch.float32, device=device) / 255.0
    return _palette_t[device]


def normalize(x):
    dev = x.device
    if dev not in _norm_t:
        _norm_t[dev] = (torch.tensor(MEAN, device=dev).view(1, 3, 1, 1),
                        torch.tensor(STD, device=dev).view(1, 3, 1, 1))
    m, s = _norm_t[dev]
    return (x - m) / s


def augment32(x, gen):
    """32x32에서 4픽셀 반사 패딩 random crop + 좌우 뒤집기. x: float [0,1], (B,3,32,32).
    0(검정) 패딩을 쓰지 않는 이유: 검정이 클래스 7(horse)의 패치 색이라 가장자리 검은 띠가 패치와 섞인다."""
    b = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar)[:, None, :, None]
    cols = (ox[:, None] + ar)[:, None, None, :]
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(3, device=x.device)[None, :, None, None]
    x = xp[bi, ci, rows, cols]
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


def upsample(x):
    return F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)


def corner_origin(corner, size):
    r0 = 0 if corner in ("tl", "tr") else IMG - size
    c0 = 0 if corner in ("tl", "bl") else IMG - size
    return r0, c0


def patch_mask(corner="br", size=PATCH, device="cpu"):
    m = torch.zeros(IMG, IMG, dtype=torch.bool, device=device)
    r0, c0 = corner_origin(corner, size)
    m[r0:r0 + size, c0:c0 + size] = True
    return m


def paste(x, s, corner="br", size=PATCH):
    """x: (B,3,224,224) [0,1]; s: (B,) 색 번호 (LongTensor). 새 텐서를 돌려준다."""
    x = x.clone()
    r0, c0 = corner_origin(corner, size)
    x[:, :, r0:r0 + size, c0:c0 + size] = palette(x.device)[s][:, :, None, None]
    return x


def gray_square(x, r0, c0, size=PATCH):
    x = x.clone()
    x[:, :, r0:r0 + size, c0:c0 + size] = 0.5
    return x


def blur224(x, sigma):
    k = int(2 * math.ceil(3 * sigma) + 1)
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def make_train_batch(x_u8, s, gen):
    x = augment32(x_u8.float() / 255.0, gen)
    return normalize(paste(upsample(x), s))


def make_eval_batch(x_u8, s):
    return normalize(paste(upsample(x_u8.float() / 255.0), s))


# ---------------------------------------------------------------- models
STRATEGIES = ("lp", "ft", "scratch")


def build_model(strategy, seed):
    """세 전략 모두 같은 ResNet-18 구조 + 새로 초기화한 10-클래스 head."""
    if strategy == "scratch":
        torch.manual_seed(seed)
        m = torchvision.models.resnet18(weights=None)
    else:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    torch.manual_seed(seed)  # 같은 seed면 LP와 FT의 head 초기값이 같다
    m.fc = nn.Linear(512, 10)
    if strategy == "lp":
        for name, prm in m.named_parameters():
            prm.requires_grad = name.startswith("fc.")
    return m


def backbone_features(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return torch.flatten(m.avgpool(x), 1)


def load_model(run_dir, device=DEV):
    cfg = json.loads((run_dir / "config.json").read_text())
    m = build_model(cfg["strategy"], cfg["seed"])
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if cfg["strategy"] == "lp":
        m.fc.load_state_dict(state)
    else:
        m.load_state_dict(state)
    return m.to(device).eval(), cfg


def run_name(strategy, p, seed, lr=None, wd=None):
    s = f"{strategy}_p{p:.2f}_s{seed}"
    if lr is not None:
        s += f"_lr{lr:g}_wd{wd:g}"
    return s


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
