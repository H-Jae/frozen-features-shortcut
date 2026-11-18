# IG(Integrated Gradients) 분석 (제안서 4.2절).
# IG = (x - x') * [x'에서 x까지 직선 경로 위 gradient의 평균]. 대상은 정답 클래스 y의 logit이고 입력은 x+(맞는 색 패치).
# - 주 기준 이미지 x': 픽셀 공간의 중간 회색(0.5)을 같은 정규화에 통과시킨 것. 민감도용: x+를 σ=16(224px)으로 흐린 것.
# - 64단계 (중점 리만 합). 수렴 확인: 200장에서 128단계와 비교 + completeness 잔차 |ΣA - (f(x)-f(x'))| / |f(x)-f(x')|.
# - R_patch = 패치 안 |A| 합 / 전체 |A| 합 (채널까지 합). 넓이 비율로 나눈 값도 저장한다.
# - 고정 부분집합: 클래스마다 100장 = 1,000장 (모든 모델 공통). 크기 stress test(12, 48)는 그중 클래스마다 50장 = 500장.
# - 히트맵: 모델마다 |Δy| 하위·중간·상위 예 3장씩 + 모든 모델 공통 6장의 채널합 |A| 지도를 저장한다.
# - 정밀도: 텐서는 fp32이고 convolution은 TF32다. 몇 모델에서 IEEE fp32와 비교한다 (precision_check).
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (DEV, IMG, PATCH, TEST_SEED, Splits, blur224, load_model, normalize, now,
                    paste, patch_mask, upsample)

CL = torch.channels_last
STEPS = 64
BLUR_SIGMA = 16
MAX_BATCH = 128
EPS = 1e-12
COMMON_CLASSES = (0, 1, 3, 5, 7, 8)


def ig_subsets(y_te):
    """클래스마다 100장을 고정해 뽑고, 그 앞쪽 50장·20장으로 부분집합을 만든다."""
    rng = np.random.default_rng([TEST_SEED, 99])
    per = [rng.choice(np.flatnonzero(y_te == k), 100, replace=False) for k in range(10)]
    s1000 = np.sort(np.concatenate(per))
    s500 = np.sort(np.concatenate([a[:50] for a in per]))
    s200 = np.sort(np.concatenate([a[:20] for a in per]))
    common = np.array([per[k][0] for k in COMMON_CLASSES])
    return s1000, s500, s200, common


def integrated_gradients(model, x, base, target, steps):
    """x, base: 정규화된 (B,3,H,W) fp32. 돌려주는 것: A (B,3,H,W), f(x)_y, f(x')_y."""
    b = x.shape[0]
    diff = x - base
    alphas = (torch.arange(steps, device=DEV, dtype=torch.float32) + 0.5) / steps
    k = max(1, MAX_BATCH // b)
    grad_sum = torch.zeros_like(x)
    for a0 in range(0, steps, k):
        a = alphas[a0:a0 + k]
        xa = (base[None] + a[:, None, None, None, None] * diff[None]).reshape(-1, *x.shape[1:])
        xa = xa.detach().requires_grad_(True)
        out = model(xa.contiguous(memory_format=CL))
        sel = out.gather(1, target.repeat(len(a))[:, None]).sum()
        (g,) = torch.autograd.grad(sel, xa)
        grad_sum += g.view(len(a), b, *x.shape[1:]).sum(0)
    attr = diff * grad_sum / steps
    with torch.no_grad():
        fx = model(x.contiguous(memory_format=CL)).gather(1, target[:, None])[:, 0]
        fb = model(base.contiguous(memory_format=CL)).gather(1, target[:, None])[:, 0]
    return attr, fx, fb


def summarize(attr, fx, fb, mask):
    a = attr.abs().sum(1)  # (B,H,W) 채널합 |A|
    tot = a.flatten(1).sum(1)
    inside = (a * mask).flatten(1).sum(1)
    r = inside / (tot + EPS)
    gap = fx - fb
    resid = (attr.flatten(1).sum(1) - gap).abs() / (gap.abs() + EPS)
    return r, resid, gap, tot, a


def run_ig(model, data, idx, patch_size, baseline, steps, img_batch, keep_maps=()):
    mask = patch_mask("br", patch_size, DEV).float()
    keep = {int(i) for i in keep_maps}
    out = {k: [] for k in ("r", "resid", "gap", "tot", "fx")}
    maps, map_idx, imgs = [], [], []
    y_all = torch.from_numpy(data.y_te).to(DEV)
    for i in range(0, len(idx), img_batch):
        ii = torch.from_numpy(idx[i:i + img_batch]).to(DEV)
        clean = upsample(data.x_te[ii].float() / 255.0)
        y = y_all[ii]
        xpix = paste(clean, y, size=patch_size)
        x = normalize(xpix)
        if baseline == "gray":
            base = normalize(torch.full_like(xpix, 0.5))
        else:
            base = normalize(blur224(xpix, BLUR_SIGMA))
        attr, fx, fb = integrated_gradients(model, x, base, y, steps)
        r, resid, gap, tot, a = summarize(attr, fx, fb, mask)
        for name, v in (("r", r), ("resid", resid), ("gap", gap), ("tot", tot), ("fx", fx)):
            out[name].append(v.detach().cpu().numpy())
        for j, gi in enumerate(idx[i:i + img_batch]):
            if int(gi) in keep:
                maps.append(a[j].detach().cpu().numpy().astype(np.float16))
                map_idx.append(int(gi))
                imgs.append((xpix[j].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    res = {k: np.concatenate(v) for k, v in out.items()}
    return res, maps, map_idx, imgs


def pick_examples(run_dir, data, s1000):
    """|Δy| 순위의 5·10·15% / 45·50·55% / 85·90·95% 지점 이미지 (낮음·중간·높음 민감도)."""
    z = np.load(run_dir / "test_logits.npz")
    y = data.y_te[s1000]
    dy = np.abs(z["match"][s1000, y] - z["wrong"][s1000, y])
    order = s1000[np.argsort(dy, kind="stable")]
    qs = [0.05, 0.10, 0.15, 0.45, 0.50, 0.55, 0.85, 0.90, 0.95]
    return [int(order[int(q * (len(order) - 1))]) for q in qs]


def ig_model(run_dir, data, limit=None, steps=STEPS):
    """끝난 표시는 ig_done.json이다. convolution은 TF32 tensor core로 계산한다 (IEEE fp32는 약 2.5배 느림).
    TF32가 R_patch를 얼마나 바꾸는지는 precision_check()로 따로 잰다."""
    run_dir = Path(run_dir)
    if (run_dir / "ig_done.json").exists():
        return
    if not (run_dir / "eval_done.json").exists():
        raise RuntimeError(f"{run_dir.name}: 평가(eval_done.json)가 먼저 끝나야 한다")
    with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False, allow_tf32=True):
        _ig(run_dir, data, limit, steps)


def _ig(run_dir, data, limit, steps):
    out_path = run_dir / "ig.npz"
    model, cfg = load_model(run_dir)
    model = model.to(memory_format=CL)
    for q in model.parameters():
        q.requires_grad_(False)
    s1000, s500, s200, common = ig_subsets(data.y_te)
    if limit is not None:
        s1000, s500, s200 = s1000[:limit], s500[:max(1, limit // 2)], s200[:max(1, limit // 5)]
    examples = pick_examples(run_dir, data, s1000)
    keep = list(dict.fromkeys(examples + [int(c) for c in common]))
    t0 = time.time()
    res, timings = {}, {}
    g64, maps, map_idx, imgs = run_ig(model, data, s1000, PATCH, "gray", steps, 2, keep)
    timings["gray64"] = round(time.time() - t0, 1)
    t1 = time.time()
    b64, *_ = run_ig(model, data, s1000, PATCH, "blur", steps, 2)
    timings["blur64"] = round(time.time() - t1, 1)
    t1 = time.time()
    g128, *_ = run_ig(model, data, s200, PATCH, "gray", 2 * steps, 1)
    timings["gray128"] = round(time.time() - t1, 1)
    t1 = time.time()
    sizes = {}
    for sz in (12, 48):
        sizes[sz], *_ = run_ig(model, data, s500, sz, "gray", steps, 2)
    timings["sizes"] = round(time.time() - t1, 1)
    for k, v in g64.items():
        res[f"gray_{k}"] = v
    for k, v in b64.items():
        res[f"blur_{k}"] = v
    for k, v in g128.items():
        res[f"gray128_{k}"] = v
    for sz, d in sizes.items():
        for k, v in d.items():
            res[f"s{sz}_{k}"] = v
    np.savez_compressed(out_path, idx1000=s1000, idx500=s500, idx200=s200, examples=np.array(examples),
                        common=common, map_idx=np.array(map_idx), maps=np.stack(maps), imgs=np.stack(imgs),
                        **res)
    (run_dir / "ig_done.json").write_text(json.dumps(
        dict(steps=steps, blur_sigma=BLUR_SIGMA, n1000=len(s1000), sec=timings,
             conv_precision="fp32 tensors, TF32 tensor-core convolutions (cudnn.allow_tf32=True)",
             finished=now()), indent=1))
    del model
    torch.cuda.empty_cache()


def precision_check(run_dir, data, steps=STEPS):
    """200장, 회색 기준 64단계 IG를 TF32와 IEEE fp32 convolution으로 각각 계산해 저장한다 (ig_precision.npz)."""
    run_dir = Path(run_dir)
    if (run_dir / "ig_precision.npz").exists():
        return
    model, _ = load_model(run_dir)
    model = model.to(memory_format=CL)
    for q in model.parameters():
        q.requires_grad_(False)
    s200 = ig_subsets(data.y_te)[2]
    res = {}
    for tag, tf in (("tf32", True), ("ieee", False)):
        with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False, allow_tf32=tf):
            r, *_ = run_ig(model, data, s200, PATCH, "gray", steps, 2)
        for k, v in r.items():
            res[f"{tag}_{k}"] = v
    np.savez_compressed(run_dir / "ig_precision.npz", idx200=s200, **res)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    d = Splits()
    for rd in sys.argv[1:]:
        ig_model(rd, d)
        print("ig done", rd, flush=True)
