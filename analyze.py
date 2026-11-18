# 결과 분석. 실행: python analyze.py   (전역 python의 numpy·scipy·matplotlib을 쓴다. torch는 필요 없다)
# 입력: results/test_meta.npz, results/tuned.json, runs/tune/*, runs/final/*, runs/lplong/*(후속 실험, 있으면)
# 출력: results/summary.json (보고서의 모든 수치), results/per_model.csv, results/figures/*.png
# 통계 (제안서 5절):
#   - 시드 변동: (전략, p)마다 시드 3개 값의 평균과 표준편차(ddof=1).
#   - 이미지 표집 불확실성: 테스트 원본 이미지 10,000장을 복원 추출(B=2000)하고, 한 이미지의 변형(x+, x-, ...)은 함께 따라간다.
#     모든 모델이 같은 재추출 표를 쓰므로 모델 간 차이도 이미지 단위로 짝지어진다. 시드 평균 벡터의 95% 백분위 구간.
#     이 구간은 사진 표집만 반영한다. 학습 seed의 흔들림은 per_seed(같은 seed끼리의 차이)와 seed_std로 따로 적는다.
#     차이(전략 간, p 간, 차이의 차이)는 비교하는 칸에 모두 있는 seed로만 계산한다.
#   - IG: 모델마다 Spearman(R_patch on x+, |Δy|)과 그 1,000장 bootstrap 95% 구간(B=1000, 모델마다 고정한 난수).
#     시드 평균 ρ와 전략 간 ρ 차이는 1,000장을 클래스 안에서 함께 재추출하는 표 하나를 모든 모델이 써서 짝지은 구간을 낸다.
#     클래스 안 ρ(클래스마다 구한 ρ의 평균)와 패치 안 |A| 합의 ρ를 민감도 분석으로 함께 낸다.
# 모델 간 비교는 정확도·PSR·확률 변화로만 한다. logit 크기는 모델마다 달라서 |Δy|는 모델 안에서만 쓴다.
# 패치 따라가기는 우연 수준(패치가 없을 때 그 클래스를 고른 비율)을 사진마다 뺀 follow_excess로 모델끼리 비교한다.
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNS, RES = ROOT / "runs", ROOT / "results"
FIG = RES / "figures"
STRATS = ("lp", "ft", "scratch")
NAME = {"lp": "LP (frozen)", "ft": "FT (full)", "scratch": "Scratch"}
COLOR = {"lp": "#2a78d6", "ft": "#eb6834", "scratch": "#1baf7a"}  # dataviz 참조 팔레트 범주 1~3번
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", [SURF, "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
PS = (0.10, 0.90, 0.99)
SEEDS = (0, 1, 2)
P_INDEP = 0.10
IMG = 224
AREA = {s: s * s / (IMG * IMG) for s in (12, 24, 48)}
B_IMG, B_IG = 2000, 1000
CORNERS = ("br", "tl", "tr", "bl")
SIZES = (12, 24, 48)
BLURS = (0, 4, 8, 16)
PAIRS = (("lp", "ft"), ("lp", "scratch"), ("ft", "scratch"))
CONTRAST_KEYS = ("psr", "psr_to_w", "psr_other", "follow", "follow_excess", "acc_conf", "acc_indep", "acc_id", "acc_match",
                 "acc_none", "dprob", "gap_id_indep", "gap_id_conf", "psr_none", "psr_occ_center", "psr_occ_tl")
PE_KEYS = ("psr", "follow", "follow_excess", "acc_conf", "acc_id", "acc_indep", "acc_none", "dprob")
DID_KEYS = ("psr", "psr_none", "psr_other", "follow", "follow_excess", "acc_conf", "acc_id", "acc_none", "dprob")
IG_KEYS = ("rho_gray", "rho_gray_within", "rho_gray_abs", "rho_gray_abs_within")  # 시드 평균 구간과 전략 간 차이를 내는 IG 상관
SEED = 20261031  # 분석 난수의 뿌리. 쓰임새마다 [SEED, 번호, ...]로 따로 뽑는다
CIFAR = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
LPLONG_P = 0.99  # 후속 실험(driver.py lplong): p = 0.99 LP의 head를 60에폭, weight decay 0으로 다시 학습 (runs/lplong)
LPLONG_COLOR = "#0d366b"  # 같은 LP를 더 오래 학습한 것이라 LP 파랑의 진한 쪽 (SEQ 팔레트 끝)

meta = np.load(RES / "test_meta.npz")
Y, U, W = meta["y_te"], meta["u_te"], meta["w_te"]
N = len(Y)
AR = np.arange(N)


def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def per_image(z, p):
    """모델 하나의 이미지별 지표 (길이 10,000 벡터들). 평균이 곧 그 모델의 지표다."""
    pred = {k: z[k].argmax(1) for k in z.files}
    zy = {k: z[k][AR, Y].astype(np.float64) for k in z.files}
    py = {k: softmax(z[k])[AR, Y] for k in z.files}
    v = {}
    v["acc_id"] = np.where(U < p, pred["match"], pred["wrong"]) == Y          # 학습과 같은 p
    v["acc_indep"] = np.where(U < P_INDEP, pred["match"], pred["wrong"]) == Y  # p=0.10: 패치와 라벨 독립
    v["acc_conf"] = pred["wrong"] == Y                                          # 9개 틀린 색 중 하나
    v["acc_match"] = pred["match"] == Y
    v["gap_id_indep"] = v["acc_id"].astype(float) - v["acc_indep"]               # ID-OOD 정확도 차이
    v["gap_id_conf"] = v["acc_id"].astype(float) - v["acc_conf"]
    v["psr"] = pred["match"] != pred["wrong"]
    v["follow"] = pred["wrong"] == W        # x-에서 패치 색의 클래스를 고른 비율
    v["psr_to_w"] = v["psr"] & v["follow"]  # 예측이 바뀌었고, 패치 색의 클래스로 갔다
    v["psr_other"] = v["psr"] & ~v["follow"]  # 예측이 바뀌었는데, 패치 색이 아닌 다른 클래스로 갔다
    v["follow_base"] = pred["none"] == W    # 패치가 없을 때 우연히 그 클래스를 고른 비율 (follow의 우연 수준)
    v["follow_excess"] = v["follow"].astype(float) - v["follow_base"]  # 우연 수준을 뺀 따라가기 (사진마다 -1, 0, 1)
    v["dy"] = zy["match"] - zy["wrong"]
    v["abs_dy"] = np.abs(v["dy"])
    v["dprob"] = py["match"] - py["wrong"]
    v["acc_none"] = pred["none"] == Y
    v["psr_none"] = pred["match"] != pred["none"]
    v["dprob_none"] = py["match"] - py["none"]
    for o in ("occ_center", "occ_tl"):
        v[f"psr_{o}"] = pred["match"] != pred[o]
        v[f"dprob_{o}"] = py["match"] - py[o]
        v[f"abs_dy_{o}"] = np.abs(zy["match"] - zy[o])
    for c in CORNERS:
        m, w = ("match", "wrong") if c == "br" else (f"match_{c}", f"wrong_{c}")
        v[f"acc_match_{c}"] = pred[m] == Y
        v[f"acc_wrong_{c}"] = pred[w] == Y
        v[f"psr_{c}"] = pred[m] != pred[w]
        v[f"follow_{c}"] = pred[w] == W
        v[f"follow_excess_{c}"] = v[f"follow_{c}"].astype(float) - v["follow_base"]  # 같은 사진, 패치 없음이 우연 수준
    for s in SIZES:
        m, w = ("match", "wrong") if s == 24 else (f"match_s{s}", f"wrong_s{s}")
        v[f"acc_match_s{s}"] = pred[m] == Y
        v[f"acc_wrong_s{s}"] = pred[w] == Y
        v[f"psr_s{s}"] = pred[m] != pred[w]
        v[f"follow_s{s}"] = pred[w] == W
    for b in BLURS:
        n_, m, w = ("none", "match", "wrong") if b == 0 else (f"none_blur{b}", f"match_blur{b}", f"wrong_blur{b}")
        v[f"acc_none_b{b}"] = pred[n_] == Y
        v[f"acc_match_b{b}"] = pred[m] == Y
        v[f"acc_wrong_b{b}"] = pred[w] == Y
        v[f"psr_b{b}"] = pred[m] != pred[w]
        v[f"follow_b{b}"] = pred[w] == W
        v[f"follow_base_b{b}"] = pred[n_] == W  # 같은 흐림, 패치 없음에서 우연히 그 클래스를 고른 비율
        v[f"follow_excess_b{b}"] = v[f"follow_b{b}"].astype(float) - v[f"follow_base_b{b}"]
    return {k: np.asarray(x, dtype=np.float64) for k, x in v.items()}


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def boot_spearman(a, b, rng):
    n = len(a)
    out = np.empty(B_IG)
    for i in range(B_IG):
        ii = rng.integers(0, n, n)
        out[i] = stats.spearmanr(a[ii], b[ii]).statistic
    return pct(out)


def pct(x):
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


def within_rho(a, b, cls):
    """클래스마다 Spearman을 따로 구해 평균한다 (클래스끼리의 차이를 뺀, 같은 클래스 사진끼리의 관계)."""
    return float(np.mean([spearman(a[cls == c], b[cls == c]) for c in range(10)]))


def rho_rows(a, b):
    """B x n 행렬 두 개의 행마다 Spearman ρ (bootstrap 반복을 한 번에 계산)."""
    ra, rb = stats.rankdata(a, axis=1), stats.rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))


def positions(big, small):
    pos = np.searchsorted(big, small)
    assert np.array_equal(big[pos], small), "IG 부분집합이 1,000장 안에 없다"
    return pos


def ig_stats(d, vec, rng):
    g = np.load(d / "ig.npz")
    idx = g["idx1000"]
    assert np.array_equal(idx, meta["s1000"]), "IG 부분집합이 모델마다 다르다"
    absdy, dprob, sw = vec["abs_dy"][idx], vec["dprob"][idx], vec["psr"][idx]
    r, rb = g["gray_r"].astype(np.float64), g["blur_r"].astype(np.float64)
    inside = r * g["gray_tot"].astype(np.float64)  # 패치 안 |A| 합 (전체 합·넓이로 나누지 않은 절대량)
    cls = Y[idx]
    p2, p5 = positions(idx, g["idx200"]), positions(idx, g["idx500"])
    r64, r128 = r[p2], g["gray128_r"].astype(np.float64)
    out = dict(
        n=len(idx),
        r_mean=float(r.mean()), r_median=float(np.median(r)),
        r_norm_mean=float(r.mean() / AREA[24]), rb_norm_mean=float(rb.mean() / AREA[24]),
        rho_gray=spearman(r, absdy), rho_gray_ci=boot_spearman(r, absdy, rng),
        rho_blur=spearman(rb, absdy), rho_blur_ci=boot_spearman(rb, absdy, rng),
        rho_gray_within=within_rho(r, absdy, cls), rho_blur_within=within_rho(rb, absdy, cls),
        rho_gray_abs=spearman(inside, absdy), rho_gray_abs_within=within_rho(inside, absdy, cls),
        rho_gray_absdprob=spearman(r, np.abs(dprob)),
        rho_gray_vs_blur=spearman(r, rb),
        r_norm_switched=float(r[sw == 1].mean() / AREA[24]) if sw.sum() else None,
        r_norm_unswitched=float(r[sw == 0].mean() / AREA[24]) if (sw == 0).sum() else None,
        n_switched=int(sw.sum()),
        resid_gray_median=float(np.median(g["gray_resid"])), resid_gray_p90=float(np.percentile(g["gray_resid"], 90)),
        resid_gray_gt5=float(np.mean(g["gray_resid"] > 0.05)),
        resid_blur_median=float(np.median(g["blur_resid"])),
        resid_gray128_median=float(np.median(g["gray128_resid"])),
        resid_gray64_on200_median=float(np.median(g["gray_resid"][p2])),
        conv_rho_64_128=spearman(r64, r128),
        conv_rel_diff_median=float(np.median(np.abs(r64 - r128) / (r128 + 1e-12))),
    )
    for s, arr in ((12, g["s12_r"]), (24, r[p5]), (48, g["s48_r"])):
        out[f"r_s{s}_mean"] = float(np.mean(arr))
        out[f"r_norm_s{s}_mean"] = float(np.mean(arr) / AREA[s])
    # 클래스(= 패치 색)마다: 패치 몫/넓이 평균과 |Δy| 중간값 (클래스끼리의 차이가 ρ에 얼마나 들어가는지 보는 값)
    out["r_norm_by_class"] = [float(r[cls == c].mean() / AREA[24]) for c in range(10)]
    out["absdy_med_by_class"] = [float(np.median(absdy[cls == c])) for c in range(10)]
    raw = dict(idx=idx, r=r, rb=rb, absdy=absdy, inside=inside, examples=g["examples"], common=g["common"],
               map_idx=g["map_idx"], maps=g["maps"], imgs=g["imgs"])
    return out, raw


def precision_stats(d, vec):
    """같은 200장의 IG를 TF32와 IEEE fp32 convolution으로 계산한 차이."""
    z = np.load(d / "ig_precision.npz")
    idx = z["idx200"]
    absdy = vec["abs_dy"][idx]
    rt, ri = z["tf32_r"].astype(np.float64), z["ieee_r"].astype(np.float64)
    rel = np.abs(rt - ri) / (ri + 1e-12)
    return dict(n=int(len(idx)), rho_tf32_vs_ieee=spearman(rt, ri),
                rel_diff_median=float(np.median(rel)), rel_diff_p95=float(np.percentile(rel, 95)),
                rho_absdy_tf32=spearman(rt, absdy), rho_absdy_ieee=spearman(ri, absdy),
                resid_tf32_median=float(np.median(z["tf32_resid"])), resid_ieee_median=float(np.median(z["ieee_resid"])))


def run_dir(st, p, s):
    return RUNS / "final" / f"{st}_p{p:.2f}_s{s}"


def load_models():
    M = {}
    for st in STRATS:
        for p in PS:
            for s in SEEDS:
                d = run_dir(st, p, s)
                if not (d / "eval_done.json").exists():  # npz만 있고 표시가 없으면 저장이 끊긴 것
                    continue
                # IG bootstrap 난수는 모델마다 고정한다 (다른 모델이 더 생겨도 이 모델의 구간은 그대로)
                rng = np.random.default_rng([SEED, 2, STRATS.index(st), PS.index(p), s])
                vec = per_image(np.load(d / "test_logits.npz"), p)
                m = dict(cfg=json.loads((d / "config.json").read_text()),
                         done=json.loads((d / "done.json").read_text()),
                         log=json.loads((d / "log.json").read_text()), vec=vec,
                         scal={k: float(x.mean()) for k, x in vec.items()})
                if (d / "ig_done.json").exists():
                    m["ig"], m["ig_raw"] = ig_stats(d, vec, rng)
                if (d / "ig_precision.npz").exists():
                    m["precision"] = precision_stats(d, vec)
                if (d / "ig_cases.npz").exists():  # 다시 계산한 R과 ig.npz 값의 일치 (TF32로 다시 돌림, 알고리즘 고정 안 함)
                    c = np.load(d / "ig_cases.npz")
                    m["cases"] = dict(n=int(len(c["idx"])),
                                      max_rel_diff=float(np.max(np.abs(c["r_new"] - c["r_saved"]) / c["r_saved"])),
                                      kind=[str(k) for k in c["kind"]], classes=[int(Y[i]) for i in c["idx"]])
                M[(st, p, s)] = m
    return M


def load_lplong():
    """후속 실험의 모델 (seed마다). 본 실험과 같은 seed라 학습 패치 배정과 head 초기값이 같고, 테스트 사진·틀린 색 w도 같다.
    IG는 하지 않았다."""
    F = {}
    for s in SEEDS:
        d = RUNS / "lplong" / f"lp_p{LPLONG_P:.2f}_s{s}"
        if not (d / "eval_done.json").exists():
            continue
        vec = per_image(np.load(d / "test_logits.npz"), LPLONG_P)
        F[s] = dict(cfg=json.loads((d / "config.json").read_text()), done=json.loads((d / "done.json").read_text()),
                    ev=json.loads((d / "eval_done.json").read_text()), log=json.loads((d / "log.json").read_text()), vec=vec,
                    scal={k: float(x.mean()) for k, x in vec.items()})
    return F


def agg(vals):
    a = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if not len(a):
        return None
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a) > 1 else 0.0, n=int(len(a)),
                values=[float(x) for x in a])


def ig_boot_reps(M):
    """모델마다 IG_KEYS 상관의 bootstrap 반복값 (B_IG개). IG 1,000장(클래스마다 100장)을 클래스 안에서 복원 추출하는
    표 하나를 모든 모델이 같이 쓰므로, 시드 평균과 모델 간 차이의 구간이 사진끼리 짝지어진다."""
    rng = np.random.default_rng([SEED, 3])
    cls = Y[meta["s1000"]]
    tabs = [pos[rng.integers(0, len(pos), (B_IG, len(pos)))] for pos in (np.where(cls == c)[0] for c in range(10))]
    allt = np.concatenate(tabs, axis=1)
    reps = {}
    for key, m in M.items():
        if "ig_raw" not in m:
            continue
        r, dy, ins = m["ig_raw"]["r"], m["ig_raw"]["absdy"], m["ig_raw"]["inside"]
        reps[key] = dict(rho_gray=rho_rows(r[allt], dy[allt]),
                         rho_gray_within=np.mean([rho_rows(r[t], dy[t]) for t in tabs], axis=0),
                         rho_gray_abs=rho_rows(ins[allt], dy[allt]),
                         rho_gray_abs_within=np.mean([rho_rows(ins[t], dy[t]) for t in tabs], axis=0))
    return reps


def blur_patch_retained():
    """흐린 기준 이미지(x+를 σ=16으로 흐림)의 패치 자리에 패치가 얼마나 남나: 패치 마스크를 같은 커널
    (torchvision gaussian_blur: 크기 2*ceil(3σ)+1 = 97, reflect 패딩 = scipy mirror)로 흐린 값의 패치 안 평균."""
    from scipy import ndimage
    mask = np.zeros((IMG, IMG))
    mask[-24:, -24:] = 1.0
    return float(ndimage.gaussian_filter(mask, 16, mode="mirror", truncate=3.0)[-24:, -24:].mean())


def main():
    M = load_models()
    if not M:
        raise SystemExit("평가가 끝난 실행이 없다")
    # 이미지 bootstrap 표는 전용 난수로 뽑는다 (IG가 있는 모델 수와 무관하게 같은 표)
    counts = np.random.default_rng([SEED, 1]).multinomial(N, np.full(N, 1.0 / N), size=B_IMG).astype(np.float32)

    def boot(vec):
        bs = counts @ vec.astype(np.float32) / N
        return [float(x) for x in np.percentile(bs, [2.5, 97.5])]

    def shared(*cells):
        """비교하는 칸 (전략, p)에 모두 있는 seed. 차이는 이 seed들로만 계산한다."""
        return [s for s in SEEDS if all((st, p, s) in M for st, p in cells)]

    def vec_mean(st, p, key, seeds):
        return np.mean([M[(st, p, s)]["vec"][key] for s in seeds], axis=0)

    def sc(st, p, s, key):
        return M[(st, p, s)]["scal"][key]

    def diff_entry(vec, per_seed):
        """vec = 사진별 시드 평균 차이. diff는 per_seed의 평균과 같다."""
        return dict(diff=float(vec.mean()), ci=boot(vec), per_seed=[float(x) for x in per_seed],
                    seed_std=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None)

    keys = list(next(iter(M.values()))["scal"].keys())
    groups = {}
    for st in STRATS:
        for p in PS:
            ss = shared((st, p))
            if not ss:
                continue
            ms = [M[(st, p, s)] for s in ss]
            g = {}
            for k in keys:
                g[k] = agg([m["scal"][k] for m in ms])
                if not k.startswith("abs_dy") and k != "dy":
                    g[k]["ci"] = boot(vec_mean(st, p, k, ss))
            g["dprob_by_class"] = [float(np.mean([m["vec"]["dprob"][Y == c].mean() for m in ms])) for c in range(10)]
            g["best_val_acc"] = agg([m["done"]["best_val_acc"] for m in ms])
            g["best_epoch"] = agg([m["done"]["best_epoch"] for m in ms])
            g["train_sec"] = agg([m["done"]["total_sec"] for m in ms])
            g["final_train_acc"] = agg([m["log"][-1]["train_acc"] for m in ms])   # 덜 학습됐는지 보는 값
            g["final_train_loss"] = agg([m["log"][-1]["train_loss"] for m in ms])
            # 고른 에폭의 검증 정확도 − 마지막 에폭의 검증 정확도 (작으면 끝 무렵이 평평했다는 뜻)
            g["val_drop_last"] = agg([m["done"]["best_val_acc"] - m["log"][-1]["val_acc"] for m in ms])
            igm = [m for m in ms if "ig" in m]
            if igm:
                igs = [m["ig"] for m in igm]
                g["ig"] = {k: agg([i[k] for i in igs]) for k in igs[0] if not isinstance(igs[0][k], list)}
                g["ig"]["rho_gray_ci_per_seed"] = [i["rho_gray_ci"] for i in igs]
                g["ig"]["rho_blur_ci_per_seed"] = [i["rho_blur_ci"] for i in igs]
                for k in ("r_norm_by_class", "absdy_med_by_class"):  # 클래스마다 seed 평균
                    g["ig"][k] = [float(x) for x in np.mean([i[k] for i in igs], axis=0)]
                if len(igm) > 1:  # 같은 사진의 |Δy|·R 순위가 seed끼리 얼마나 같은지 (잡음인지 보는 값)
                    pairs = [(i, j) for i in range(len(igm)) for j in range(i + 1, len(igm))]
                    g["ig"]["absdy_seed_rho"] = agg([spearman(igm[i]["ig_raw"]["absdy"], igm[j]["ig_raw"]["absdy"])
                                                     for i, j in pairs])
                    g["ig"]["r_seed_rho"] = agg([spearman(igm[i]["ig_raw"]["r"], igm[j]["ig_raw"]["r"]) for i, j in pairs])
            groups[f"{st}|{p:.2f}"] = g

    contrasts = {}
    for a, b in PAIRS:
        for p in PS:
            ss = shared((a, p), (b, p))
            if ss:
                contrasts[f"{a}-{b}|{p:.2f}"] = {
                    k: diff_entry(vec_mean(a, p, k, ss) - vec_mean(b, p, k, ss), [sc(a, p, s, k) - sc(b, p, s, k) for s in ss])
                    for k in CONTRAST_KEYS}

    # H2·SQ1: 같은 전략 안에서 p를 올린 효과 (같은 seed끼리 짝: seed가 학습 패치 추출·초기화·셔플을 정한다)
    p_effects = {}
    for st in STRATS:
        for p in (0.90, 0.99):
            ss = shared((st, p), (st, P_INDEP))
            if ss:
                p_effects[f"{st}|{p:.2f}-0.10"] = {
                    k: diff_entry(vec_mean(st, p, k, ss) - vec_mean(st, P_INDEP, k, ss),
                                  [sc(st, p, s, k) - sc(st, P_INDEP, s, k) for s in ss]) for k in PE_KEYS}

    # SQ1: 두 전략의 차이가 p=0.10에서 p=0.90·0.99로 갈 때 얼마나 바뀌나 (차이의 차이, 이미지 짝 bootstrap)
    did = {}
    for a, b in PAIRS:
        for p in (0.90, 0.99):
            ss = shared((a, p), (b, p), (a, P_INDEP), (b, P_INDEP))
            if ss:
                did[f"{a}-{b}|{p:.2f}-0.10"] = {
                    k: diff_entry((vec_mean(a, p, k, ss) - vec_mean(b, p, k, ss))
                                  - (vec_mean(a, P_INDEP, k, ss) - vec_mean(b, P_INDEP, k, ss)),
                                  [(sc(a, p, s, k) - sc(b, p, s, k)) - (sc(a, P_INDEP, s, k) - sc(b, P_INDEP, s, k))
                                   for s in ss]) for k in DID_KEYS}

    # 후속 실험: p = 0.99 LP를 head만 더 오래(60에폭), weight decay 없이 학습. 본 LP·FT(p = 0.99)와 같은 seed끼리 짝짓는다
    # (같은 seed = 같은 학습 패치 배정). 'LP가 덜 학습돼서 패치를 덜 썼나'를 본다.
    FU = load_lplong()
    followup = None
    if FU:
        fs = sorted(FU)
        g = {}
        for k in keys:
            g[k] = agg([FU[s]["scal"][k] for s in fs])
            if not k.startswith("abs_dy") and k != "dy":
                g[k]["ci"] = boot(np.mean([FU[s]["vec"][k] for s in fs], axis=0))
        g["best_val_acc"] = agg([FU[s]["done"]["best_val_acc"] for s in fs])
        g["best_epoch"] = agg([FU[s]["done"]["best_epoch"] for s in fs])
        g["train_sec"] = agg([FU[s]["done"]["total_sec"] for s in fs])
        g["eval_sec"] = agg([FU[s]["ev"]["sec"] for s in fs])
        g["final_train_acc"] = agg([FU[s]["log"][-1]["train_acc"] for s in fs])
        g["final_train_loss"] = agg([FU[s]["log"][-1]["train_loss"] for s in fs])
        g["val_drop_last"] = agg([FU[s]["done"]["best_val_acc"] - FU[s]["log"][-1]["val_acc"] for s in fs])
        # 마지막 20에폭 동안 오른 학습 정확도 (끝까지 오르고 있었는지)
        g["train_acc_gain_last20"] = agg([FU[s]["log"][-1]["train_acc"] - FU[s]["log"][-21]["train_acc"] for s in fs])
        cfg = FU[fs[0]]["cfg"]
        followup = dict(config={k: cfg[k] for k in ("p", "lr", "wd", "epochs", "batch")}, seeds=fs, group=g, contrasts={},
                        # 돌린 때는 실행 폴더의 시각으로 적는다 (driver.log의 표시는 같은 명령을 다시 부르면 늘어난다)
                        started=min(FU[s]["cfg"]["started"] for s in fs), finished=max(FU[s]["ev"]["finished"] for s in fs))
        for b in ("lp", "ft"):
            ss = [s for s in fs if (b, LPLONG_P, s) in M]
            if ss:
                followup["contrasts"][f"lplong-{b}"] = {
                    k: diff_entry(np.mean([FU[s]["vec"][k] for s in ss], axis=0) - vec_mean(b, LPLONG_P, k, ss),
                                  [FU[s]["scal"][k] - sc(b, LPLONG_P, s, k) for s in ss]) for k in CONTRAST_KEYS}
        # FT와의 PSR 차이 중 후속에서도 남은 몫 = (후속 LP − FT) / (본 LP − FT). 구간은 같은 bootstrap 표로 낸다
        if all((b, LPLONG_P, s) in M for b in ("lp", "ft") for s in fs):
            d_fu = np.mean([FU[s]["vec"]["psr"] for s in fs], axis=0) - vec_mean("ft", LPLONG_P, "psr", fs)
            d_lp = vec_mean("lp", LPLONG_P, "psr", fs) - vec_mean("ft", LPLONG_P, "psr", fs)
            bs = (counts @ d_fu.astype(np.float32)) / (counts @ d_lp.astype(np.float32))
            followup["psr_gap_kept"] = dict(
                value=float(d_fu.mean() / d_lp.mean()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])],
                per_seed=[(FU[s]["scal"]["psr"] - sc("ft", LPLONG_P, s, "psr"))
                          / (sc("lp", LPLONG_P, s, "psr") - sc("ft", LPLONG_P, s, "psr")) for s in fs])

    # IG 상관: 시드 평균의 짝지은 구간과 전략 간 차이
    reps = ig_boot_reps(M)
    ig_contrasts = {}
    for st in STRATS:
        for p in PS:
            ss = [s for s in SEEDS if (st, p, s) in reps]
            g = groups.get(f"{st}|{p:.2f}")
            if ss and g and "ig" in g:
                for k in IG_KEYS:
                    g["ig"][k]["ci"] = pct(np.mean([reps[(st, p, s)][k] for s in ss], axis=0))
    for a, b in PAIRS:
        for p in PS:
            ss = [s for s in SEEDS if (a, p, s) in reps and (b, p, s) in reps]
            if ss:
                ig_contrasts[f"{a}-{b}|{p:.2f}"] = {}
                for k in IG_KEYS:
                    per_seed = [M[(a, p, s)]["ig"][k] - M[(b, p, s)]["ig"][k] for s in ss]
                    ig_contrasts[f"{a}-{b}|{p:.2f}"][k] = dict(
                        diff=float(np.mean(per_seed)),
                        ci=pct(np.mean([reps[(a, p, s)][k] - reps[(b, p, s)][k] for s in ss], axis=0)),
                        per_seed=per_seed, seed_std=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None)

    # 모델 단위 관계: 모델마다 평균 R_patch(넓이 정규화)와 행동 지표의 순위 상관.
    # 같은 조건의 seed 3개는 거의 같은 점이라 모델 전체의 ρ는 주로 p의 효과다. 그래서 조건 평균과 같은 p 안의 ρ도 낸다.
    def ml_rhos(items):
        rn = [m["ig"]["r_norm_mean"] for m in items]
        return dict(n=len(items), rnorm_psr=spearman(rn, [m["scal"]["psr"] for m in items]),
                    rnorm_follow_excess=spearman(rn, [m["scal"]["follow_excess"] for m in items]),
                    rblur_psr=spearman([m["ig"]["rb_norm_mean"] for m in items], [m["scal"]["psr"] for m in items]))

    mods = [(k, m) for k, m in M.items() if "ig" in m]
    model_level = None
    if len(mods) >= 4:
        conds = sorted({k[:2] for k, _ in mods}, key=lambda c: (PS.index(c[1]), STRATS.index(c[0])))
        cmeans = [dict(ig={f: np.mean([m["ig"][f] for k, m in mods if k[:2] == c]) for f in ("r_norm_mean", "rb_norm_mean")},
                       scal={f: np.mean([m["scal"][f] for k, m in mods if k[:2] == c]) for f in ("psr", "follow_excess")})
                  for c in conds]
        model_level = dict(n=len(mods), all=ml_rhos([m for _, m in mods]),
                           cond=ml_rhos(cmeans) if len(cmeans) >= 4 else None,
                           within_p={f"{p:.2f}": ml_rhos([m for k, m in mods if k[1] == p])
                                     for p in PS if sum(k[1] == p for k, _ in mods) >= 4})

    tuning = collect_tuning()
    summary = dict(
        n_models=len(M), models=sorted(f"{st}|{p:.2f}|{s}" for st, p, s in M),
        tuned=json.loads((RES / "tuned.json").read_text()) if (RES / "tuned.json").exists() else None,
        tuning=tuning, groups=groups, contrasts=contrasts, p_effects=p_effects, did=did, ig_contrasts=ig_contrasts,
        model_level=model_level, followup=followup,
        precision={f"{st}|{p:.2f}|{s}": m["precision"] for (st, p, s), m in M.items() if "precision" in m},
        ig_cases={f"{st}|{p:.2f}|{s}": m["cases"] for (st, p, s), m in M.items() if "cases" in m},
        per_model={f"{st}|{p:.2f}|{s}": dict(scal=m["scal"], ig=m.get("ig"), done=m["done"])
                   for (st, p, s), m in M.items()},
        notes=dict(bootstrap_images=B_IMG, bootstrap_ig=B_IG, test_n=N, ig_n=len(meta["s1000"]),
                   area=AREA, p_indep=P_INDEP, blur_patch_retained=blur_patch_retained()))
    RES.mkdir(exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    write_csv(M)
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    figures(M, groups)
    if FU:
        lplong_figure(M, FU)
    print("models:", len(M), "| with IG:", len(mods), "| follow-up (lplong):", len(FU))


def collect_tuning():
    rows = []
    for d in sorted((RUNS / "tune").glob("*")):
        if (d / "done.json").exists():
            c = json.loads((d / "config.json").read_text())
            r = json.loads((d / "done.json").read_text())
            log = json.loads((d / "log.json").read_text())
            rows.append(dict(strategy=c["strategy"], lr=c["lr"], wd=c["wd"], epochs=c["epochs"],
                             best_val_acc=r["best_val_acc"], best_epoch=r["best_epoch"],
                             final_val_acc=log[-1]["val_acc"], final_train_loss=log[-1]["train_loss"],
                             sec=r["total_sec"]))
    return rows


def write_csv(M):
    cols = ["strategy", "p", "seed", "best_val_acc", "best_epoch", "acc_id", "acc_indep", "acc_conf", "acc_match",
            "acc_none", "psr", "follow", "dprob", "abs_dy", "psr_none", "psr_occ_center", "psr_occ_tl",
            "ig_rho_gray", "ig_rho_blur", "ig_r_norm_mean", "ig_resid_gray_median"]
    with open(RES / "per_model.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for (st, p, s), m in sorted(M.items()):
            ig = m.get("ig") or {}
            wr.writerow([st, p, s, m["done"]["best_val_acc"], m["done"]["best_epoch"]]
                        + [round(m["scal"][k], 5) for k in cols[5:17]]
                        + [ig.get("rho_gray"), ig.get("rho_blur"), ig.get("r_norm_mean"), ig.get("resid_gray_median")])


# ---------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False,
        "lines.linewidth": 1.8, "lines.markersize": 5.5, "savefig.dpi": 160})


def gval(groups, st, p, key, field="mean"):
    g = groups.get(f"{st}|{p:.2f}")
    if not g or key not in g or g[key] is None:
        return np.nan
    return g[key][field]


def line_panel(ax, groups, key, title, ylabel=None, xs=PS, getter=None, label_end=True):
    x = np.arange(len(xs))
    for st in STRATS:
        if getter:
            mu = np.array([getter(st, p)[0] for p in xs])
            sd = np.array([getter(st, p)[1] for p in xs])
        else:
            mu = np.array([gval(groups, st, p, key) for p in xs])
            sd = np.array([gval(groups, st, p, key, "std") for p in xs])
        if np.all(np.isnan(mu)):
            continue
        ax.errorbar(x, mu, yerr=sd, color=COLOR[st], marker="o", capsize=3, label=NAME[st])
        if label_end and not np.isnan(mu[-1]):  # 선 끝 이름표는 finish()가 겹치지 않게 놓는다
            ax._end_labels = getattr(ax, "_end_labels", []) + [(x[-1], mu[-1], NAME[st].split(" ")[0])]
    ax.set_xticks(x, [f"{p:.2f}" for p in xs])
    ax.set_xlim(-0.3, len(xs) - 0.4)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def place_end_labels(fig, size=8.5):
    """선 끝 이름표를 값 순서대로 세우고, 가까우면 글자 높이만큼 위아래로 벌린다 (전체 위치는 평균이 그대로이게).
    이름표가 칸 밖으로 나가면 글자 반 높이만큼 안쪽으로 되돌리고, 제 선 끝에서 2pt 넘게 밀린 이름표는 가는 선으로 잇는다."""
    fig.canvas.draw()
    gap = size * 1.3 * fig.dpi / 72  # 픽셀
    for ax in fig.axes:
        labs = getattr(ax, "_end_labels", None)
        if not labs:
            continue
        ys = np.array([ax.transData.transform((x, y))[1] for x, y, _ in labs])
        order = np.argsort(ys, kind="stable")
        new = ys[order].copy()
        for i in range(1, len(new)):
            new[i] = max(new[i], new[i - 1] + gap)
        new -= (new - ys[order]).mean()
        box = ax.get_window_extent()
        lo, hi = box.y0 + gap / 2, box.y1 - gap / 2
        if new[0] < lo:
            new += lo - new[0]
        if new[-1] > hi:
            new -= new[-1] - hi
        for i, ny in zip(order, new):
            x, y, t = labs[i]
            dy = (ny - ys[i]) * 72 / fig.dpi  # 포인트
            arrow = dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=3) if abs(dy) > 2 else None
            ax.annotate(t, (x, y), xytext=(8 if arrow else 6, dy), textcoords="offset points", va="center",
                        fontsize=size, color=INK2, annotation_clip=False, arrowprops=arrow)


def finish(fig, name, legend_from=None):
    """범례는 그림 맨 위 한 줄에 둔다 (데이터를 가리지 않게). 모든 칸의 항목을 모아 전략 순서로. 그다음 끝 이름표를 놓고 저장한다."""
    top = 1.0
    if legend_from is not None:
        seen = {}
        for ax in fig.axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(lab, h)
        labs = [NAME[st] for st in STRATS if NAME[st] in seen]
        if labs:
            fig.legend([seen[lab] for lab in labs], labs, loc="upper center", ncol=len(labs), frameon=False, fontsize=9)
            top = 1 - 0.34 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    place_end_labels(fig)
    fig.savefig(FIG / name)
    plt.close(fig)


def figures(M, groups):
    # 1. 정확도 4조건
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, (k, t) in zip(axs, [("acc_id", "ID (training p)"), ("acc_indep", "Independent OOD (p=0.10)"),
                                ("acc_conf", "Conflicting OOD (wrong patch)"), ("acc_none", "No patch (removal)")]):
        line_panel(ax, groups, k, t)
        ax.set_xlabel("training correlation p")
    axs[0].set_ylabel("test accuracy")
    finish(fig, "fig1_accuracy.png", axs[0])

    # 2. 패치 의존 지표
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, (k, t) in zip(axs, [("psr", "PSR: x+ vs x- (patch swap)"), ("follow_excess", "Follows patch on x- (minus chance)"),
                                ("psr_none", "Switch: x+ vs no patch"), ("psr_occ_center", "Switch: x+ vs gray occluder")]):
        line_panel(ax, groups, k, t)
        ax.set_xlabel("training correlation p")
    axs[0].set_ylabel("rate over 10,000 test images")
    finish(fig, "fig2_patch_reliance.png", axs[0])

    # 3. 위치 stress test (결과가 있는 전략만 막대로)
    sts = [st for st in STRATS if any(f"{st}|{p:.2f}" in groups for p in PS)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, p in zip(axs, PS):
        x = np.arange(len(CORNERS))
        wbar = 0.26
        here = [st for st in sts if f"{st}|{p:.2f}" in groups]  # 이 p에 결과가 있는 전략만, 가운데 맞춤
        for j, st in enumerate(here):
            mu = [gval(groups, st, p, f"psr_{c}") for c in CORNERS]
            sd = [gval(groups, st, p, f"psr_{c}", "std") for c in CORNERS]
            ax.bar(x + (j - (len(here) - 1) / 2) * (wbar + 0.01), mu, wbar, yerr=sd, color=COLOR[st], label=NAME[st],
                   capsize=2, error_kw=dict(ecolor=INK2, lw=0.8))
        ax.set_xticks(x, ["bottom-right\n(train)", "top-left", "top-right", "bottom-left"])
        ax.set_title(f"patch position, p={p:.2f}")
    axs[0].set_ylabel("PSR (x+ vs x-)")
    finish(fig, "fig3_position.png", axs[0])

    # 4. 크기 stress test (+ 넓이 정규화 IG)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, p in zip(axs[:2], (0.90, 0.99)):
        line_panel(ax, groups, None, f"PSR by patch size, p={p:.2f}", xs=SIZES,
                   getter=lambda st, s, p=p: (gval(groups, st, p, f"psr_s{s}"), gval(groups, st, p, f"psr_s{s}", "std")))
        ax.set_xticks(range(3), ["12", "24 (train)", "48"])
        ax.set_xlabel("patch side (px at 224)")
    axs[0].set_ylabel("PSR")
    for ax, p in zip(axs[2:], (0.90, 0.99)):
        def get(st, s, p=p):
            g = groups.get(f"{st}|{p:.2f}", {}).get("ig", {}).get(f"r_norm_s{s}_mean")
            return (g["mean"], g["std"]) if g else (np.nan, np.nan)
        line_panel(ax, groups, None, f"IG patch share / area, p={p:.2f}", xs=SIZES, getter=get)
        ax.set_xticks(range(3), ["12", "24 (train)", "48"])
        ax.set_xlabel("patch side (px at 224)")
    axs[2].set_ylabel("R_patch / area fraction")
    finish(fig, "fig4_size.png", axs[0])

    # 5. 물체 정보 줄이기 (blur) stress test
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharey="row")
    for r, p in enumerate((0.90, 0.99)):
        for ax, (k, t) in zip(axs[r], [("acc_none_b", "no patch accuracy"), ("acc_wrong_b", "conflicting accuracy"),
                                       ("follow_excess_b", "follows patch on x- (minus chance)")]):
            line_panel(ax, groups, None, f"{t}, p={p:.2f}", xs=BLURS,
                       getter=lambda st, b, k=k, p=p: (gval(groups, st, p, f"{k}{b}"), gval(groups, st, p, f"{k}{b}", "std")))
            ax.set_xticks(range(4), ["none", "σ=4", "σ=8", "σ=16"])
        axs[r][0].set_ylabel("rate over 10,000 test images")
    for ax in axs[1]:
        ax.set_xlabel("blur σ of the object image (px at 224)")
    finish(fig, "fig5_blur.png", axs[0][0])

    # 6. 학습 곡선 (결과가 있는 전략만 행으로)
    fig, axs = plt.subplots(len(sts), 3, figsize=(12, 2.7 * len(sts)), sharey=True, squeeze=False)
    for i, st in enumerate(sts):
        for j, p in enumerate(PS):
            ax = axs[i][j]
            if not any((st, p, s) in M for s in SEEDS):
                ax.text(0.5, 0.5, "no runs yet", transform=ax.transAxes, ha="center", va="center", color=INK2)
                ax.tick_params(axis="x", labelbottom=False)
            for s in SEEDS:
                if (st, p, s) not in M:
                    continue
                log = M[(st, p, s)]["log"]
                ep = [r["epoch"] for r in log]
                ax.plot(ep, [r["val_acc"] for r in log], color=COLOR[st], lw=1.2, alpha=0.9)
                ax.plot(ep, [r["train_acc"] for r in log], color=INK2, lw=1.0, ls="--", alpha=0.7)
                be = M[(st, p, s)]["done"]["best_epoch"]
                ax.plot([be], [log[be - 1]["val_acc"]], marker="o", color=COLOR[st], ms=5)
            ax.set_title(f"{NAME[st]}, p={p:.2f}")
            if j == 0:
                ax.set_ylabel("accuracy")
            if i == len(sts) - 1:
                ax.set_xlabel("epoch")
    axs[0][0].plot([], [], color=INK2, ls="--", label="train (augmented)")
    axs[0][0].plot([], [], color=COLOR["lp"], label="ID validation")
    axs[0][0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_curves.png")
    plt.close(fig)

    ig_figures(M, groups)


def lplong_figure(M, FU):
    """후속 실험의 학습 곡선: p = 0.99 LP, 본 실험(15에폭)과 후속(60에폭). seed마다 선 하나. 점선 = 패치만 봐도 맞히는 비율."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    runs = [(M[("lp", LPLONG_P, s)], COLOR["lp"], "LP, 15 epochs, wd 5e-4 (main)") for s in SEEDS if ("lp", LPLONG_P, s) in M]
    runs += [(FU[s], LPLONG_COLOR, "LP, 60 epochs, wd 0 (follow-up)") for s in sorted(FU)]
    for m, c, lab in runs:
        ep = [r["epoch"] for r in m["log"]]
        axs[0].plot(ep, [r["train_acc"] for r in m["log"]], color=c, lw=1.2, label=lab)
        axs[1].plot(ep, [r["val_acc"] for r in m["log"]], color=c, lw=1.2, label=lab)
        be = m["done"]["best_epoch"]
        axs[1].plot([be], [m["log"][be - 1]["val_acc"]], marker="o", color=c, ms=5)
    for ax, t in zip(axs, ("training accuracy (augmented batches)", f"ID validation accuracy (p={LPLONG_P:.2f})")):
        ax.axhline(LPLONG_P, color=INK2, lw=0.9, ls=":")
        ax.set_title(t)
        ax.set_xlabel("epoch")
    axs[0].annotate("patch alone", (1, LPLONG_P), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    axs[0].set_ylabel("accuracy")
    seen = {}
    for h, lab in zip(*axs[0].get_legend_handles_labels()):
        seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="upper center", ncol=len(seen), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.34 / fig.get_figheight()))
    fig.savefig(FIG / "fig13_lplong.png")
    plt.close(fig)


def ig_figures(M, groups):
    have = {k: m for k, m in M.items() if "ig" in m}
    if not have:
        return
    # 7. 산점도 (seed 0, IG가 있는 전략만 행으로)
    sts = [st for st in STRATS if any((st, p, 0) in have for p in PS)]
    fig, axs = plt.subplots(len(sts), 3, figsize=(11, 3 * len(sts)), squeeze=False)
    for i, st in enumerate(sts):
        for j, p in enumerate(PS):
            ax = axs[i][j]
            m = have.get((st, p, 0))
            if m is None:
                ax.set_axis_off()
                continue
            raw, ig = m["ig_raw"], m["ig"]
            ax.scatter(raw["absdy"], raw["r"] / AREA[24], s=7, color=COLOR[st], alpha=0.45, linewidths=0)
            ax.set_yscale("log")
            ax.set_title(f"{NAME[st]}, p={p:.2f}, seed 0")
            lo, hi = ig["rho_gray_ci"]
            ax.text(0.97, 0.05, f"ρ = {ig['rho_gray']:.2f} [{lo:.2f}, {hi:.2f}]", transform=ax.transAxes,
                    ha="right", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("IG R_patch / area (log)")
            if not any((s2, p, 0) in have for s2 in sts[i + 1:]):  # 이 열에서 그림이 있는 마지막 칸
                ax.set_xlabel("|Δy| = |z_y(x+) − z_y(x−)|")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_ig_scatter.png")
    plt.close(fig)

    # 8. ρ 요약
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, (k, t) in zip(axs, [("rho_gray", "Spearman ρ, gray baseline"), ("rho_blur", "Spearman ρ, blur baseline"),
                                ("rho_gray_vs_blur", "ρ between the two baselines' R_patch")]):
        def get(st, p, k=k):
            g = groups.get(f"{st}|{p:.2f}", {}).get("ig", {}).get(k)
            return (g["mean"], g["std"]) if g else (np.nan, np.nan)
        line_panel(ax, groups, None, t, getter=get)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("training correlation p")
    axs[0].set_ylabel("Spearman ρ (within model)")
    finish(fig, "fig8_ig_rho.png", axs[0])

    # 9. 모델 단위: PSR vs 평균 R_patch
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    marker = {0.10: "o", 0.90: "s", 0.99: "^"}
    for (st, p, s), m in have.items():
        ax.scatter(m["scal"]["psr"], m["ig"]["r_norm_mean"], color=COLOR[st], marker=marker[p], s=46,
                   edgecolors=SURF, linewidths=1.5)
    for st in sts:
        ax.scatter([], [], color=COLOR[st], label=NAME[st], s=46)
    for p in PS:
        ax.scatter([], [], color=INK2, marker=marker[p], label=f"p={p:.2f}", s=46)
    ax.set_xlabel("PSR (behavioral patch reliance)")
    ax.set_ylabel("mean IG R_patch / area")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8.5)
    ax.set_title("one point per trained model")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_model_level.png")
    plt.close(fig)

    # 10. 히트맵: 모델마다 |Δy| 하위·중간·상위 예 (seed 0)
    for st in STRATS:
        rows = [(p, have.get((st, p, 0))) for p in PS if have.get((st, p, 0)) is not None]
        if not rows:
            continue
        fig, axs = plt.subplots(2 * len(rows), 9, figsize=(13.5, 3.1 * len(rows)))
        axs = np.atleast_2d(axs)
        for r, (p, m) in enumerate(rows):
            raw = m["ig_raw"]
            pos = {int(g): i for i, g in enumerate(raw["map_idx"])}
            dyv = m["vec"]["abs_dy"]
            for c, gi in enumerate(raw["examples"]):
                k = pos[int(gi)]
                a_img, a_map = axs[2 * r][c], axs[2 * r + 1][c]
                a_img.imshow(raw["imgs"][k])
                hm = raw["maps"][k].astype(np.float32)
                a_map.imshow(hm, cmap=SEQ, vmin=0, vmax=np.percentile(hm, 99.5) + 1e-12)
                a_map.add_patch(Rectangle((IMG - 24 - 0.5, IMG - 24 - 0.5), 24, 24, fill=False, ec=INK, lw=0.8))
                rr = raw["r"][np.searchsorted(raw["idx"], gi)] / AREA[24]
                grp = ["low", "low", "low", "mid", "mid", "mid", "high", "high", "high"][c]
                a_img.set_title(f"{grp} |Δy|={dyv[gi]:.2f}", fontsize=8)
                a_map.set_title(f"R/area={rr:.1f}", fontsize=8)
                for a in (a_img, a_map):
                    a.set_xticks([])
                    a.set_yticks([])
                    a.grid(False)
            axs[2 * r][0].set_ylabel(f"p={p:.2f}\nimage", fontsize=9)
            axs[2 * r + 1][0].set_ylabel("IG |A|", fontsize=9)
        fig.suptitle(f"{NAME[st]} (seed 0): IG on x+ for low / middle / high patch sensitivity", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(FIG / f"fig10_heatmaps_{st}.png")
        plt.close(fig)

    # 11. 모든 모델 공통 이미지 6장
    order = [(st, p) for st in STRATS for p in PS if (st, p, 0) in have]
    if order:
        m0 = have[(order[0][0], order[0][1], 0)]
        common = [int(c) for c in m0["ig_raw"]["common"]]
        fig, axs = plt.subplots(len(order) + 1, len(common), figsize=(1.6 * len(common) + 1, 1.6 * (len(order) + 1)))
        for c, gi in enumerate(common):
            raw0 = m0["ig_raw"]
            k0 = {int(g): i for i, g in enumerate(raw0["map_idx"])}[gi]
            axs[0][c].imshow(raw0["imgs"][k0])
            axs[0][c].set_title(CIFAR[Y[gi]], fontsize=8)
            for r, (st, p) in enumerate(order, start=1):
                raw = have[(st, p, 0)]["ig_raw"]
                k = {int(g): i for i, g in enumerate(raw["map_idx"])}[gi]
                hm = raw["maps"][k].astype(np.float32)
                axs[r][c].imshow(hm, cmap=SEQ, vmin=0, vmax=np.percentile(hm, 99.5) + 1e-12)
                axs[r][c].add_patch(Rectangle((IMG - 24.5, IMG - 24.5), 24, 24, fill=False, ec=INK, lw=0.6))
                if c == 0:
                    axs[r][c].set_ylabel(f"{NAME[st]}\np={p:.2f}", fontsize=8)
        for a in axs.flat:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
        axs[0][0].set_ylabel("x+ image", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "fig11_common_heatmaps.png")
        plt.close(fig)

    # 12. IG와 패치 민감도가 어긋난 사례 (ig_cases.py가 정해 둔 규칙으로 고른 것, seed 0)
    for p in (0.90, 0.99):
        rows = [(st, np.load(run_dir(st, p, 0) / "ig_cases.npz")) for st in STRATS
                if (run_dir(st, p, 0) / "ig_cases.npz").exists()]
        if not rows:
            continue
        ncol = max(len(c["idx"]) for _, c in rows)
        fig, axs = plt.subplots(2 * len(rows), ncol, figsize=(1.75 * ncol + 0.6, 3.7 * len(rows)))
        axs = np.atleast_2d(axs)
        for i, (st, c) in enumerate(rows):
            for j in range(len(c["idx"])):
                a_img, a_map = axs[2 * i][j], axs[2 * i + 1][j]
                a_img.imshow(c["imgs"][j])
                hm = c["maps"][j].astype(np.float32)
                a_map.imshow(hm, cmap=SEQ, vmin=0, vmax=np.percentile(hm, 99.5) + 1e-12)
                a_map.add_patch(Rectangle((IMG - 24.5, IMG - 24.5), 24, 24, fill=False, ec=INK, lw=0.8))
                kind = "IG misses" if str(c["kind"][j]) == "miss" else "IG false alarm"
                a_img.set_title(f"{kind}\n|Δy| rank {100 * c['rank_dy'][j]:.0f}%", fontsize=8)
                a_map.set_title(f"R rank {100 * c['rank_r'][j]:.0f}%, R/area {c['r_new'][j] / AREA[24]:.1f}", fontsize=8)
                for a in (a_img, a_map):
                    a.set_xticks([])
                    a.set_yticks([])
                    a.grid(False)
            axs[2 * i][0].set_ylabel(f"{NAME[st]}\nimage", fontsize=9)
            axs[2 * i + 1][0].set_ylabel("IG |A|", fontsize=9)
        fig.suptitle(f"p={p:.2f}, seed 0: images where IG and the measured patch effect disagree", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(FIG / f"fig12_cases_p{p:.2f}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
