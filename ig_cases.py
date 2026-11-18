# IG와 실제 패치 민감도가 어긋나는 사례 (제안서 "representative explanation failures")의 히트맵을 만든다.
# 실행: ..\..\.venv\Scripts\python.exe ig_cases.py   (driver.py final이 끝난 뒤. GPU를 쓴다)
# 고르는 규칙 (결과를 보기 전에 정했다). seed 0 모델 중 p=0.90, 0.99인 6개에서, IG 1,000장의 |Δy|와 R_patch(회색 기준)에
# 백분위 순위(0~1)를 매긴다.
#   - IG가 놓친 사례: |Δy| 순위 상위 25% 안에서 (|Δy| 순위 − R 순위)가 가장 큰 3장
#   - IG가 헛짚은 사례: R 순위 상위 25% 안에서 (R 순위 − |Δy| 순위)가 가장 큰 3장
# 같은 사진의 IG를 ig.py와 같은 설정(회색 기준, 64단계, TF32 convolution)으로 다시 계산해 히트맵을 저장하고,
# 다시 잰 R이 ig.npz의 값과 얼마나 같은지 잰다 (TF32로 다시 돌렸을 때의 일치. convolution 알고리즘은 고정하지 않는다).
# 출력: runs/final/<모델>/ig_cases.npz
import numpy as np
import torch

from common import PATCH, RUNS_DIR, Splits, load_model, run_name
from ig import CL, STEPS, run_ig

MODELS = [(st, p) for st in ("lp", "ft", "scratch") for p in (0.90, 0.99)]
K = 3


def pct_rank(a):
    return a.argsort(kind="stable").argsort(kind="stable") / (len(a) - 1)


def pick(r, absdy):
    rr, rd = pct_rank(r), pct_rank(absdy)
    top_dy = np.where(rd >= 0.75)[0]
    miss = top_dy[np.argsort(-(rd - rr)[top_dy], kind="stable")[:K]]
    top_r = np.where(rr >= 0.75)[0]
    false = top_r[np.argsort(-(rr - rd)[top_r], kind="stable")[:K]]
    return np.concatenate([miss, false]), rr, rd


def main():
    data = Splits()
    y = data.y_te
    for st, p in MODELS:
        rd = RUNS_DIR / "final" / run_name(st, p, 0)
        out = rd / "ig_cases.npz"
        if out.exists() or not (rd / "ig_done.json").exists():
            continue
        g = np.load(rd / "ig.npz")
        idx, r = g["idx1000"], g["gray_r"].astype(np.float64)
        z = np.load(rd / "test_logits.npz")
        absdy = np.abs(z["match"][idx, y[idx]].astype(np.float64) - z["wrong"][idx, y[idx]])
        sel, rr, rdy = pick(r, absdy)
        assert len(set(sel.tolist())) == len(sel), "같은 사진이 두 번 뽑혔다"
        gi = idx[sel]
        model, _ = load_model(rd)
        model = model.to(memory_format=CL)
        for q in model.parameters():
            q.requires_grad_(False)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
            res, maps, map_idx, imgs = run_ig(model, data, gi, PATCH, "gray", STEPS, 2, keep_maps=gi)
        assert map_idx == [int(x) for x in gi]
        np.savez_compressed(out, idx=gi, kind=np.array(["miss"] * K + ["false"] * K), r_saved=r[sel], r_new=res["r"],
                            absdy=absdy[sel], rank_r=rr[sel], rank_dy=rdy[sel], maps=np.stack(maps), imgs=np.stack(imgs))
        rel = np.abs(res["r"] - r[sel]) / r[sel]
        print(f"{rd.name}: 다시 잰 R과 저장된 R의 상대 차이 최대 {rel.max():.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
