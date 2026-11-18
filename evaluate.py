# 학습된 모델 하나를 테스트 10,000장의 변형들에 넣고 logit을 저장한다 (fp32).
# 모든 모델이 같은 테스트 이미지·같은 틀린 색 w를 쓴다 (common.Splits.u_te, w_te).
# 변형 (제안서 4.1절 + 4.3절 stress test 1~5 + 가림 대조군):
#   match / wrong          : 오른쪽 아래 24x24에 맞는 색(x+) / 틀린 색(x-)  -> PSR, Δy, 충돌 OOD
#   none                   : 패치 자리의 원래 픽셀 복원 (removal control)
#   occ_center, occ_tl     : x+에 같은 넓이의 회색 사각형을 패치가 아닌 곳에 덧댐 (perturbation control)
#   {match,wrong}_{tl,tr,bl}: 패치를 다른 모서리로 옮김 (position)
#   {match,wrong}_s{12,48} : 패치 크기 바꿈 (size)
#   {none,match,wrong}_blur{4,8,16}: 원본을 흐린 뒤 온전한 패치를 붙임 (object degradation, σ는 224px 기준)
# ID(p)와 독립 OOD(p=0.10) 정확도는 이미지마다 u<p이면 match, 아니면 wrong의 logit을 골라 계산한다
# (학습 때와 같은 결합 추출이라 따로 추론할 필요가 없다).
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (DEV, IMG, PATCH, Splits, blur224, gray_square, load_model, normalize, now,
                    paste, upsample)

CL = torch.channels_last
CENTER = (IMG - PATCH) // 2

VARIANTS = [
    # name, base image, patch (color source, corner, size) or None, gray occluder (r0, c0) or None
    ("match", "clean", ("y", "br", PATCH), None),
    ("wrong", "clean", ("w", "br", PATCH), None),
    ("none", "clean", None, None),
    ("occ_center", "clean", ("y", "br", PATCH), (CENTER, CENTER)),
    ("occ_tl", "clean", ("y", "br", PATCH), (0, 0)),
]
for _c in ("tl", "tr", "bl"):
    VARIANTS += [(f"match_{_c}", "clean", ("y", _c, PATCH), None),
                 (f"wrong_{_c}", "clean", ("w", _c, PATCH), None)]
for _s in (12, 48):
    VARIANTS += [(f"match_s{_s}", "clean", ("y", "br", _s), None),
                 (f"wrong_s{_s}", "clean", ("w", "br", _s), None)]
for _b in (4, 8, 16):
    VARIANTS += [(f"none_blur{_b}", f"blur{_b}", None, None),
                 (f"match_blur{_b}", f"blur{_b}", ("y", "br", PATCH), None),
                 (f"wrong_blur{_b}", f"blur{_b}", ("w", "br", PATCH), None)]


def build_variant(spec, bases, y, w):
    _, base, patch, occ = spec
    x = bases[base]
    if patch is not None:
        src, corner, size = patch
        x = paste(x, y if src == "y" else w, corner=corner, size=size)
    if occ is not None:
        x = gray_square(x, *occ)
    return x


def eval_model(run_dir, data, batch=500, limit=None):
    """끝난 표시는 eval_done.json이다 (npz는 저장 도중 끊기면 일부만 남을 수 있어서 표시로 쓰지 않는다).
    convolution도 IEEE fp32로 계산한다 (cuDNN 기본값인 TF32를 끈다)."""
    run_dir = Path(run_dir)
    if (run_dir / "eval_done.json").exists():
        return
    with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False, allow_tf32=False):
        _eval(run_dir, data, batch, limit)


@torch.no_grad()
def _eval(run_dir, data, batch, limit):
    out = run_dir / "test_logits.npz"
    model, cfg = load_model(run_dir)
    model = model.to(memory_format=CL)
    n = len(data.y_te) if limit is None else limit
    y_all = torch.from_numpy(data.y_te).to(DEV)
    w_all = torch.from_numpy(data.w_te).to(DEV)
    logits = {v[0]: np.zeros((n, 10), np.float32) for v in VARIANTS}
    t0 = time.time()
    for i in range(0, n, batch):
        j = min(n, i + batch)
        clean = upsample(data.x_te[i:j].float() / 255.0)
        bases = {"clean": clean}
        for b in (4, 8, 16):
            bases[f"blur{b}"] = blur224(clean, b)
        y, w = y_all[i:j], w_all[i:j]
        for spec in VARIANTS:
            x = normalize(build_variant(spec, bases, y, w)).contiguous(memory_format=CL)
            logits[spec[0]][i:j] = model(x).float().cpu().numpy()
    np.savez_compressed(out, **logits)
    (run_dir / "eval_done.json").write_text(json.dumps(
        dict(n=n, variants=[v[0] for v in VARIANTS], conv_precision="IEEE fp32 (cudnn.allow_tf32=False)",
             sec=round(time.time() - t0, 1), finished=now()), indent=1))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    d = Splits()
    for rd in sys.argv[1:]:
        eval_model(rd, d)
        print("evaluated", rd, flush=True)
