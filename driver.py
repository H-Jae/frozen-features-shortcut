# 전체 실험 지휘. 실행: ..\..\.venv\Scripts\python.exe driver.py <단계>
#   smoke : 작은 부분집합으로 학습->평가->IG가 끝까지 도는지 확인 (runs/smoke)
#   bench : 전략별 학습 속도, 평가·IG 속도 측정 -> 전체 소요 시간 추정
#   tune  : 전략마다 같은 크기의 격자(lr 3개 x wd 2개 = 6개)를 p=0.10, 튜닝 전용 seed 100에서 학습 (runs/tune)
#   select: 전략마다 검증 정확도가 가장 높은 설정을 results/tuned.json에 기록 (격자 끝이면 한 칸 넓혀 본다)
#   final : 3전략 x p{0.10,0.90,0.99} x seed{0,1,2} = 27회. 한 회가 끝날 때마다 평가와 IG까지 한다 (runs/final)
#           끝에 seed 0 모델 6개에서 IG의 TF32/IEEE 차이를 잰다 (ig_precision.npz)
#   lplong: 후속 실험. p=0.99 LP의 head를 60에폭, weight decay 0으로 다시 학습하고 평가만 한다 (runs/lplong)
# 끝난 표시(done.json, eval_done.json, ig_done.json)가 있는 단계는 건너뛰므로 중간에 멈춰도 다시 실행하면 이어진다.
import json
import sys
import time
import traceback

import torch

from common import RESULTS_DIR, RUNS_DIR, Splits, now, run_name
from evaluate import eval_model
from ig import ig_model, precision_check
from train import train

TUNE_P, TUNE_SEED = 0.10, 100
GRID = {
    # LP의 처음 격자를 낮게 잡은 이유: smoke 실행에서 lr 0.1일 때 loss 3.9로 흔들렸다(ImageNet 특징의 노름이 약 28).
    # 격자 끝 규칙으로 두 번 넓힌 결과 본 설정(warmup + cosine, 15에폭)에서는 0.1이 가장 좋았다(tuned.json).
    "lp": dict(lrs=[0.001, 0.003, 0.01], wds=[1e-4, 5e-4], epochs=15),
    "ft": dict(lrs=[0.003, 0.01, 0.03], wds=[1e-4, 5e-4], epochs=10),
    "scratch": dict(lrs=[0.05, 0.1, 0.2], wds=[1e-4, 5e-4], epochs=40),
}
# 격자 끝에서 고른 lr을 한 칸 더 넓혀 볼 때 쓰는 사다리 (처음 격자는 가운데 3칸)
LADDER = {"lp": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
          "ft": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
          "scratch": [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]}
MAX_EXT = 2
PRECISION_MODELS = [(st, p) for st in ("lp", "ft", "scratch") for p in (0.10, 0.99)]  # seed 0
FINAL_ORDER = [("lp", p) for p in (0.10, 0.99, 0.90)] + [("ft", p) for p in (0.10, 0.99, 0.90)] \
    + [("scratch", p) for p in (0.10, 0.99, 0.90)]
SEEDS = (0, 1, 2)
LPLONG = dict(p=0.99, epochs=60, wd=0.0)  # 본 LP(15에폭, wd 5e-4)보다 head를 4배 오래, weight decay 없이
LOG = RUNS_DIR / "driver.log"


def say(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def guarded(label, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        say(f"FAILED {label}\n{traceback.format_exc()}")
        torch.cuda.empty_cache()
        return None


def stage_smoke(data):
    d = RUNS_DIR / "smoke"
    for st in ("lp", "ft", "scratch"):
        rd = d / run_name(st, 0.99, 0)
        say(f"smoke train {rd.name}")
        train(st, 0.99, 0, GRID[st]["lrs"][1], 1e-4, 1, rd, data=data, limit=2048)
        eval_model(rd, data, limit=1000)
        ig_model(rd, data, limit=20, steps=8)
        say(f"smoke ok {rd.name}")


def stage_bench(data):
    d = RUNS_DIR / "bench"
    for st in ("lp", "ft", "scratch"):
        rd = d / run_name(st, 0.10, 0)
        torch.cuda.synchronize()
        t0 = time.time()
        train(st, 0.10, 0, GRID[st]["lrs"][1], 1e-4, 2, rd, data=data, limit=10240, quiet=True)
        log = json.loads((rd / "log.json").read_text())
        sec = log[-1]["sec"]  # 두 번째 에폭 (cudnn 탐색 이후)
        say(f"bench {st}: {10240 / sec:.0f} img/s train -> 45k epoch {45000 / (10240 / sec):.0f}s")
    rd = d / run_name("ft", 0.10, 0)
    t0 = time.time()
    eval_model(rd, data, limit=2000)
    say(f"bench eval: {time.time() - t0:.1f}s for 2000 imgs x 24 variants")
    t0 = time.time()
    ig_model(rd, data, limit=20)
    say(f"bench ig (20 imgs gray64+blur64, 4 gray128, 10x2 sizes): {time.time() - t0:.1f}s")


def stage_tune(data):
    for st, g in GRID.items():
        for lr in g["lrs"]:
            for wd in g["wds"]:
                rd = RUNS_DIR / "tune" / run_name(st, TUNE_P, TUNE_SEED, lr, wd)
                say(f"tune {rd.name}")
                r = guarded(rd.name, train, st, TUNE_P, TUNE_SEED, lr, wd, g["epochs"], rd, data=data, quiet=True)
                if r:
                    say(f"  -> best_val_acc={r['best_val_acc']:.4f} at ep{r['best_epoch']} ({r['total_sec']}s)")


def tune_rows(st, lrs, wds):
    rows = []
    for lr in lrs:
        for wd in wds:
            rd = RUNS_DIR / "tune" / run_name(st, TUNE_P, TUNE_SEED, lr, wd)
            if not (rd / "done.json").exists():
                raise SystemExit(f"{st}: 튜닝이 덜 끝났다 ({rd.name}). 전략마다 같은 격자를 다 돌린 뒤에 고른다.")
            r = json.loads((rd / "done.json").read_text())
            rows.append(dict(lr=lr, wd=wd, best_val_acc=r["best_val_acc"], best_epoch=r["best_epoch"]))
    return rows


def stage_select():
    """전략마다 검증 정확도 최고 설정을 고른다. 최고 lr이 격자 끝이면 그 방향으로 LADDER의 다음 값을
    (wd 2개 모두) 더 돌려 본다. 이 규칙은 모든 전략에 똑같이, 최대 MAX_EXT번 적용한다."""
    chosen, data = {}, None
    for st, g in GRID.items():
        lrs, ext = list(g["lrs"]), []
        for _ in range(MAX_EXT):
            best = max(tune_rows(st, lrs, g["wds"]), key=lambda r: r["best_val_acc"])
            i = LADDER[st].index(best["lr"])
            if best["lr"] == max(lrs) and i + 1 < len(LADDER[st]):
                new = LADDER[st][i + 1]
            elif best["lr"] == min(lrs) and i > 0:
                new = LADDER[st][i - 1]
            else:
                break
            say(f"select {st}: best lr={best['lr']} is at the grid edge -> also try lr={new}")
            data = data or Splits()
            for wd in g["wds"]:
                rd = RUNS_DIR / "tune" / run_name(st, TUNE_P, TUNE_SEED, new, wd)
                r = train(st, TUNE_P, TUNE_SEED, new, wd, g["epochs"], rd, data=data, quiet=True)
                say(f"  {rd.name} -> best_val_acc={r['best_val_acc']:.4f} at ep{r['best_epoch']} ({r['total_sec']}s)")
            lrs = sorted(lrs + [new])
            ext.append(new)
        rows = tune_rows(st, lrs, g["wds"])
        best = max(rows, key=lambda r: r["best_val_acc"])
        chosen[st] = dict(lr=best["lr"], wd=best["wd"], epochs=g["epochs"], grid=rows, extended_lrs=ext)
        say(f"select {st}: lr={best['lr']} wd={best['wd']} val={best['best_val_acc']:.4f} (of {len(rows)} configs)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tuned.json").write_text(json.dumps(chosen, indent=1))


def stage_final(data):
    tuned = json.loads((RESULTS_DIR / "tuned.json").read_text())
    for st, p in FINAL_ORDER:
        c = tuned[st]
        for seed in SEEDS:
            rd = RUNS_DIR / "final" / run_name(st, p, seed)
            if not (rd / "done.json").exists():
                say(f"final train {rd.name} (lr={c['lr']} wd={c['wd']} ep={c['epochs']})")
            r = guarded(rd.name, train, st, p, seed, c["lr"], c["wd"], c["epochs"], rd, data=data, quiet=True)
            if r is None:
                continue
            if not (rd / "eval_done.json").exists():
                say(f"  trained: best_val_acc={r['best_val_acc']:.4f} ep{r['best_epoch']} ({r['total_sec']}s); eval")
            guarded(rd.name + " eval", eval_model, rd, data)
            if not (rd / "ig_done.json").exists():
                say("  ig")
            guarded(rd.name + " ig", ig_model, rd, data)
    for st, p in PRECISION_MODELS:
        rd = RUNS_DIR / "final" / run_name(st, p, 0)
        if (rd / "done.json").exists() and not (rd / "ig_precision.npz").exists():
            say(f"precision check (TF32 vs IEEE IG) {rd.name}")
            guarded(rd.name + " precision", precision_check, rd, data)


def stage_lplong(data):
    """보고서 '학습이 충분했나' 절의 열린 질문: p=0.99의 LP가 패치를 덜 쓴 것이 head를 덜 학습한 탓인가.
    본 실험과 같은 seed(같은 패치 배정)·같은 학습률로 head만 더 오래, weight decay 없이 학습하고 평가한다 (IG는 하지 않는다)."""
    lr = json.loads((RESULTS_DIR / "tuned.json").read_text())["lp"]["lr"]
    for seed in SEEDS:
        rd = RUNS_DIR / "lplong" / run_name("lp", LPLONG["p"], seed)
        if not (rd / "done.json").exists():
            say(f"lplong train {rd.name} (lr={lr} wd={LPLONG['wd']} ep={LPLONG['epochs']})")
        r = guarded(rd.name, train, "lp", LPLONG["p"], seed, lr, LPLONG["wd"], LPLONG["epochs"], rd,
                    data=data, quiet=True)
        if r is None:
            continue
        if not (rd / "eval_done.json").exists():
            say(f"  trained: best_val_acc={r['best_val_acc']:.4f} ep{r['best_epoch']} ({r['total_sec']}s); eval")
        guarded(rd.name + " eval", eval_model, rd, data)


if __name__ == "__main__":
    if sys.platform == "win32":
        import ctypes  # 이 프로세스가 도는 동안만 절전 진입을 막는다 (설정은 바꾸지 않음, 종료하면 풀림)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    stage = sys.argv[1]
    say(f"=== stage {stage} start")
    if stage == "select":
        stage_select()
    else:
        dat = Splits()
        {"smoke": stage_smoke, "bench": stage_bench, "tune": stage_tune, "final": stage_final,
         "lplong": stage_lplong}[stage](dat)
    say(f"=== stage {stage} end")
