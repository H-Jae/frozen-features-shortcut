# 결과 보고서 HTML 만들기. 실행: python report.py [--artifact]   (analyze.py 다음에 돌린다)
# 입력: results/summary.json, results/test_meta.npz, results/figures/*.png, report_text.py(결과 해석 글)
# 출력: docs/report.html (그림을 data URI로 넣은 파일 하나)
#       --artifact면 docs/report_artifact.html도 만든다 (Artifact 게시용: doctype·html·head·body 태그 없이 title·style·본문만)
# 보고서의 수치는 모두 summary.json에서 코드로 넣는다. 손으로 옮겨 적은 숫자는 없다.
# 결과 해석 글(절 제목·질문·굵은 답·본문)은 report_text.py의 texts(R, H, S)가 준다. 없으면 자리 표시만 넣는다.
# 사진 같은 그림(입력 예시·히트맵)은 JPEG(품질 85, 4:4:4)로 다시 넣는다. PNG 그대로면 27개 모델에서 약 29 MB라
# Artifact 한도(16 MB)를 넘는다. 4:2:0은 패치 경계 색이 번지고, 256색 PNG는 패치 색 자체가 바뀌어서 쓰지 않는다.
import base64
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import PIL
import scipy
from PIL import Image

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
DOCS = ROOT / "docs"  # 보고서 같은 글은 여기에 쓴다 (git에는 올리지 않는다)
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
META = np.load(RES / "test_meta.npz")
STRATS = ("lp", "ft", "scratch")
PS = (0.10, 0.90, 0.99)
NAME = {"lp": "LP", "ft": "FT", "scratch": "Scratch"}
CLASSES = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]
COLOR_NAMES = ["빨강", "초록", "파랑", "노랑", "자홍", "청록", "흰색", "검정", "주황", "보라"]
N_PARAMS = {"lp": 5130, "ft": 11181642, "scratch": 11181642}  # torchvision resnet18(num_classes=10)에서 센 값


# ---------------------------------------------------------------- 수치 평탄화와 서식
def p2(p):
    return f"{round(float(p) * 100):02d}"


def flatten():
    """summary.json을 'psr.lp.99', 'c.psr.lp-ft.99', 'pe.psr.lp.99', 'did.psr.lp-ft.99', 'ig.rho_gray.lp.99',
    'igc.rho_gray.lp-ft.99', 'ml.all.rnorm_psr', 'ml.p99.rnorm_psr' 꼴의 키로 편다. 시드 수는 '.n'."""
    R = {}

    def put(base, x):
        R[base], R[base + ".sd"], R[base + ".n"] = x["mean"], x["std"], x["n"]
        R[base + ".values"] = x["values"]
        if "ci" in x:
            R[base + ".lo"], R[base + ".hi"] = x["ci"]

    for gk, g in S["groups"].items():
        st, p = gk.split("|")
        for k, x in g.items():
            if k == "ig":
                for k2, y in x.items():
                    if isinstance(y, dict):
                        put(f"ig.{k2}.{st}.{p2(p)}", y)
                    elif k2 in ("r_norm_by_class", "absdy_med_by_class"):
                        R[f"ig.{k2}.{st}.{p2(p)}"] = y
            elif isinstance(x, dict):
                put(f"{k}.{st}.{p2(p)}", x)
            elif k == "dprob_by_class":
                R[f"{k}.{st}.{p2(p)}"] = x
    for tag, key in (("c", "contrasts"), ("pe", "p_effects"), ("did", "did"), ("igc", "ig_contrasts")):
        for ck, c in (S.get(key) or {}).items():
            who, p = ck.split("|")
            for k, x in c.items():
                base = f"{tag}.{k}.{who}.{p2(p.split('-')[0])}"
                R[base] = x["diff"]
                R[base + ".lo"], R[base + ".hi"] = x["ci"]
                if "per_seed" in x:
                    R[base + ".seeds"] = x["per_seed"]
    fu = S.get("followup")
    if fu:  # 후속 실험(p = 0.99 LP, head를 더 오래): 'psr.lplong.99', 'c.psr.lplong-lp.99', 'c.psr.lplong-ft.99'
        q = p2(fu["config"]["p"])
        for k, x in fu["group"].items():
            if isinstance(x, dict):
                put(f"{k}.lplong.{q}", x)
        for who, c in fu["contrasts"].items():
            for k, x in c.items():
                base = f"c.{k}.{who}.{q}"
                R[base] = x["diff"]
                R[base + ".lo"], R[base + ".hi"] = x["ci"]
                R[base + ".seeds"] = x["per_seed"]
        kg = fu.get("psr_gap_kept")
        if kg:  # FT와의 PSR 차이 중 후속에서도 남은 몫: 'psr_gap_kept.lplong.99'
            base = f"psr_gap_kept.lplong.{q}"
            R[base], R[base + ".seeds"] = kg["value"], kg["per_seed"]
            R[base + ".lo"], R[base + ".hi"] = kg["ci"]
    ml = S.get("model_level") or {}
    for part in ("all", "cond"):
        for k, x in (ml.get(part) or {}).items():
            R[f"ml.{part}.{k}"] = x
    for p, d in (ml.get("within_p") or {}).items():
        for k, x in d.items():
            R[f"ml.p{p2(p)}.{k}"] = x
    R["blur_patch_retained"] = (S.get("notes") or {}).get("blur_patch_retained")
    return R


R = flatten()


def has(k):
    return R.get(k) is not None


MINUS = "−"  # 본문의 'LP − FT'와 같은 빼기 기호


def fx(x, d=1, sign=False):
    """숫자 서식. 반올림해서 0이 되면 부호를 뗀다 ('-0.0' 방지). 음수는 빼기 기호(−)로 쓴다."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if s[0] in "+-" and float(s) == 0:
        s = s[1:]
    return s.replace("-", MINUS)


def pc(k, d=1):
    """비율 -> '12.3%'"""
    return f"{fx(100 * R[k], d)}%" if has(k) else "–"


def one_seed(k):
    """seed가 하나뿐이면 표준편차를 쓸 수 없다."""
    return R.get(k + ".n", 3) < 2


def pcsd(k, d=1):
    """비율 -> '12.3% (seed 표준편차 0.4)'"""
    if not has(k):
        return "–"
    return f"{fx(100 * R[k], d)}%" + (" (seed 1개)" if one_seed(k) else f" (seed 표준편차 {fx(100 * R[k + '.sd'], d)})")


def ppx(k, d=1):
    """차이 -> '+4.5%p'"""
    return f"{fx(100 * R[k], d, True)}%p" if has(k) else "–"


def pp(k, d=1):
    """차이 -> '+4.5%p [+3.9, +5.1]' (괄호 = 테스트 사진 bootstrap 95% 구간, 표와 같은 꼴)"""
    if not has(k):
        return "–"
    return (f"{fx(100 * R[k], d, True)}%p [{bound(100 * R[k + '.lo'], d, True)}, "
            f"{bound(100 * R[k + '.hi'], d, True)}]")


def num(k, d=2, sd=False):
    if not has(k):
        return "–"
    return fx(R[k], d) + (f" ± {fx(R[k + '.sd'], d)}" if sd and not one_seed(k) else "")


TOL = 1e-9  # 계산 잡음 (float32 합) 때문에 0에 붙은 구간 끝을 0과 다르다고 보지 않게


def excl0(k):
    """bootstrap 95% 구간이 0을 포함하지 않으면 +1(양수) 또는 -1(음수), 포함하면 0"""
    if not has(k):
        return None
    lo, hi = R[k + ".lo"], R[k + ".hi"]
    return 1 if lo > TOL else (-1 if hi < -TOL else 0)


def chip(kind, label):
    sym = {"yes": "✓", "no": "✗", "mixed": "±", "wait": "…"}[kind]
    return f'<span class="chip {kind}"><span aria-hidden="true">{sym}</span>{label}</span>'


H = SimpleNamespace(pc=pc, pcsd=pcsd, ppx=ppx, pp=pp, num=num, excl0=excl0, chip=chip, has=has, fx=fx)


# ---------------------------------------------------------------- 표
def lab(st, p=None):
    pv = "" if p is None else f' <span class="pv">p={p:.2f}</span>'
    return f'<td class="lab"><span class="dot {st}"></span>{NAME[st]}{pv}</td>'


def td_ms(k, scale=100, d=1, ci=False, extra=""):
    """평균 ± seed 표준편차. ci면 seed 평균의 95% 구간을 작은 줄로. seed가 3개보다 적으면 그 수를 적는다
    (1개면 표준편차를 쓰지 않는다)."""
    if not has(k):
        return '<td class="num">–</td>'
    s = fx(scale * R[k], d) + ("" if one_seed(k) else f'<span class="sd">±{fx(scale * R[k + ".sd"], d)}</span>')
    if ci and has(k + ".lo"):
        s += f'<small>[{bound(scale * R[k + ".lo"], d)}, {bound(scale * R[k + ".hi"], d)}]</small>'
    if R.get(k + ".n", 3) < 3:
        s += f'<small>seed {R[k + ".n"]}개</small>'
    return f'<td class="num">{s}{extra}</td>'


def bound(x, d, sign=False):
    """구간 끝. 0이 아닌데 반올림하면 0이 되는 값은 0이 아니게 보일 때까지 자리를 늘린다(최대 3자리 더).
    그래야 0을 포함하지 않는 구간이 0으로 보이지 않고, 음수 끝의 부호도 사라지지 않는다."""
    dd = d
    while x != 0 and abs(x) < 0.5 * 10 ** -dd and dd < d + 3:
        dd += 1
    return fx(x, dd, sign)


def td_diff(k, scale=100, d=1, bold=True):
    """차이. 굵은 글씨 = 사진 bootstrap 구간이 0을 포함하지 않고, seed별 차이가 2개 이상이며 모두 같은 부호."""
    if not has(k):
        return '<td class="num">–</td>'
    lo, hi = R[k + ".lo"], R[k + ".hi"]
    seeds = R.get(k + ".seeds") or []
    steady = len(seeds) >= 2 and all(v * R[k] > 0 for v in seeds)
    cls = "num hit" if bold and (lo > TOL or hi < -TOL) and steady else "num"
    out = f'{fx(scale * R[k], d, True)}<small>[{bound(scale * lo, d, True)}, {bound(scale * hi, d, True)}]</small>'
    if seeds:  # 둘째 작은 줄: seed 0·1·2 각각의 차이 (표 폭을 줄이려고 이름표 없이)
        out += f'<small>{" ".join(fx(scale * v, d, True) for v in seeds)}</small>'
        if len(seeds) < 3:
            out += f'<small>seed {len(seeds)}개</small>'
    return f'<td class="{cls}">{out}</td>'


def table(head, rows, cap=None, cls="", empty="아직 결과 없음"):
    th = "".join(f"<th>{h}</th>" for h in head)
    if not rows:
        rows = [(False, [f'<td class="lab pending" colspan="{len(head)}">{empty}</td>'])]
    body = "".join(f'<tr{" class=\"grp\"" if g else ""}>{"".join(r)}</tr>' for g, r in rows)
    c = f'<p class="tcap">{cap}</p>' if cap else ""
    return f'{c}<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def pending_strat(st):
    """결과가 아직 없는 전략의 자리 표시 문구."""
    return f"{NAME[st]}: 아직 결과 없음 (본 학습 진행 중)"


def model_rows(cells, ps=PS):
    """전략 x p 행. 전략이 바뀌는 첫 행에 구분선. 결과가 하나도 없는 전략은 자리 표시 한 줄,
    일부 p만 없으면 그 p마다 자리 표시 한 줄."""
    rows = []
    ncell = len(cells(STRATS[0], p2(ps[0])))
    for st in STRATS:
        have = [p for p in ps if f"{st}|{p:.2f}" in S["groups"]]
        if not have:
            rows.append((st != STRATS[0], [lab(st), f'<td class="lab pending" colspan="{ncell}">{pending_strat(st)}</td>']))
            continue
        for i, p in enumerate(ps):
            if p in have:
                rows.append((i == 0 and st != STRATS[0], [lab(st, p)] + cells(st, p2(p))))
            else:
                rows.append((i == 0 and st != STRATS[0],
                             [lab(st, p), f'<td class="lab pending" colspan="{ncell}">아직 결과 없음 (본 학습 진행 중)</td>']))
    return rows


def psr_spread():
    """(같은 조건 seed 사이 PSR 표준편차의 최댓값, 조건 평균 PSR의 최솟값, 최댓값)"""
    gs = [(st, p2(p)) for st in STRATS for p in PS if f"{st}|{p:.2f}" in S["groups"]]
    sds = [R[f"psr.{st}.{q}.sd"] for st, q in gs if not one_seed(f"psr.{st}.{q}")]
    means = [R[f"psr.{st}.{q}"] for st, q in gs]
    return (max(sds) if sds else 0.0), min(means), max(means)


def pending_pair(*sts):
    """비교 표가 비었을 때: 결과가 없는 전략을 밝힌다."""
    miss = [NAME[st] for st in sts if not any(f"{st}|{p:.2f}" in S["groups"] for p in PS)]
    return f"아직 결과 없음 ({', '.join(miss)} 본 학습 진행 중)" if miss else "아직 결과 없음"


def missing_row(label_cells, ncell):
    """비교 표에서 아직 계산할 수 없는 행의 자리 표시."""
    return label_cells + [f'<td class="lab pending" colspan="{ncell}">아직 결과 없음 (본 학습 진행 중)</td>']


CAP_MS = ("값: seed 3개의 평균, 작은 글씨는 seed 표준편차(seed가 3개보다 적으면 그 수를 적는다). "
          "단위 %(차이 열은 %p = 퍼센트포인트).")
CAP_DIFF = ("단위 %p(퍼센트포인트). 큰 글씨: seed 평균끼리의 차이(비교하는 칸에 모두 있는 seed만 쓴다). "
            "괄호: 테스트 사진 bootstrap 95% 구간(사진 표집만 반영한다). "
            "그 아래 줄: seed 0·1·2 각각에서 잰 차이(학습 seed에 따른 흔들림은 이 줄로 본다). "
            "굵은 글씨: 구간이 0을 포함하지 않고, seed별 차이도 모두 같은 부호인 차이.")
CAP_FOLLOW = (" 패치 따라가기(우연 보정) = x−에서 패치 색의 클래스를 고른 비율 − 패치가 없을 때 그 클래스를 고른 비율(사진마다 뺀다). "
              "보정 전 값은 굵게 쓰지 않는다. 정확도가 낮은 모델일수록 우연 수준이 높아서 보정 전 차이는 패치 의존을 부풀린다.")


def build_tables():
    T = {}
    T["acc"] = table(["모델", "ID<br>(학습 p)", "독립 OOD<br>(p=0.10)", "충돌 OOD<br>(틀린 색 x−)", "패치 없음",
                      "ID − 독립 OOD<br>(%p)", "ID − 충돌<br>(%p)"],
                     model_rows(lambda st, q: [td_ms(f"{k}.{st}.{q}") for k in
                                              ("acc_id", "acc_indep", "acc_conf", "acc_none", "gap_id_indep", "gap_id_conf")]),
                     "테스트 정확도. " + CAP_MS)
    T["rel"] = table(["모델", "PSR<br>(색만 바꿈)", "패치 따라가기<br>(보정 전)", "패치 따라가기<br>우연 수준",
                      "패치 따라가기<br>(우연 보정, %p)", "정답 확률 변화<br>x+ − x− (%p)", "패치 제거<br>전환율",
                      "회색 가림<br>전환율 가운데", "회색 가림<br>전환율 왼쪽 위"],
                     model_rows(lambda st, q: [td_ms(f"{k}.{st}.{q}") for k in
                                              ("psr", "follow", "follow_base", "follow_excess", "dprob", "psr_none",
                                               "psr_occ_center", "psr_occ_tl")]),
                     "패치 의존 지표. 패치 따라가기 우연 수준 = 패치가 없을 때 우연히 그 틀린 색의 클래스를 고른 비율. "
                     "우연 보정 = 보정 전 − 우연 수준. 전환율 = x+와 해당 변형의 예측이 다른 사진의 비율. " + CAP_MS)
    ckeys = (("psr", "PSR", True), ("follow_excess", "패치 따라가기<br>(우연 보정)", True),
             ("follow", "패치 따라가기<br>(보정 전)", False), ("dprob", "정답 확률 변화", True),
             ("psr_none", "패치 제거<br>전환율", True), ("psr_occ_center", "회색 가림 전환율<br>(가운데)", True),
             ("acc_conf", "충돌 OOD 정확도", True), ("acc_id", "ID 정확도", True))
    def started(*sts):  # 이 전략들 모두 결과가 하나라도 있나 (있으면 빠진 행에 자리 표시를 단다)
        return all(any(f"{st}|{p:.2f}" in S["groups"] for p in PS) for st in sts)

    for a, b in (("lp", "ft"), ("lp", "scratch"), ("ft", "scratch")):
        rows = []
        for p in PS:
            lab_p = [f'<td class="lab">p={p:.2f}</td>']
            if has(f"c.psr.{a}-{b}.{p2(p)}"):
                rows.append((False, lab_p + [td_diff(f"c.{k}.{a}-{b}.{p2(p)}", bold=bd) for k, _, bd in ckeys]))
            elif started(a, b):
                rows.append((False, missing_row(lab_p, len(ckeys))))
        T[f"c_{a}_{b}"] = table(["학습 p"] + [h for _, h, _ in ckeys], rows,
                                f"{NAME[a]} − {NAME[b]}. " + CAP_DIFF + CAP_FOLLOW, empty=pending_pair(a, b))
    rows = []
    pkeys = ("psr", "follow_excess", "dprob", "acc_conf", "acc_id", "acc_indep")
    for st in STRATS:
        for i, p in enumerate((0.90, 0.99)):
            lab_p = [lab(st), f'<td class="lab">p=0.10 → {p:.2f}</td>']
            if has(f"pe.psr.{st}.{p2(p)}"):
                rows.append((i == 0 and st != STRATS[0], lab_p + [td_diff(f"pe.{k}.{st}.{p2(p)}") for k in pkeys]))
            elif started(st):
                rows.append((i == 0 and st != STRATS[0], missing_row(lab_p, len(pkeys))))
    T["peff"] = table(["전략", "학습 p 변화", "PSR", "패치 따라가기<br>(우연 보정)", "정답 확률 변화", "충돌 OOD 정확도",
                       "ID 정확도", "독립 OOD 정확도"], rows,
                      "같은 전략 안에서 p를 올렸을 때의 변화 (p=0.90 또는 0.99 모델 − p=0.10 모델, 같은 seed끼리 짝). "
                      + CAP_DIFF, empty=pending_pair(*STRATS))
    rows = []
    dkeys = ("psr", "follow_excess", "dprob", "psr_none", "acc_conf", "acc_id")
    for a, b in (("lp", "ft"), ("lp", "scratch"), ("ft", "scratch")):
        for i, p in enumerate((0.90, 0.99)):
            lab_p = [f'<td class="lab">{NAME[a]} − {NAME[b]}</td>', f'<td class="lab">p=0.10 → {p:.2f}</td>']
            if has(f"did.psr.{a}-{b}.{p2(p)}"):
                rows.append((i == 0 and a + b != "lpft", lab_p + [td_diff(f"did.{k}.{a}-{b}.{p2(p)}") for k in dkeys]))
            elif started(a, b):
                rows.append((i == 0 and a + b != "lpft", missing_row(lab_p, len(dkeys))))
    T["did"] = table(["비교 (a − b)", "학습 p 변화", "PSR", "패치 따라가기<br>(우연 보정)", "정답 확률 변화",
                      "패치 제거<br>전환율", "충돌 OOD 정확도", "ID 정확도"], rows,
                     "차이의 차이 = (해당 p에서 a − b) − (p=0.10에서 a − b). +면 p를 올렸을 때 a의 값이 b보다 상대적으로 "
                     "올라갔고, −면 내려갔다. "
                     + CAP_DIFF, empty=pending_pair(*STRATS))
    corners = (("br", "오른쪽 아래<br>(학습 자리)"), ("tl", "왼쪽 위"), ("tr", "오른쪽 위"), ("bl", "왼쪽 아래"))
    T["pos"] = table(["모델"] + [f"PSR<br>{c}" for _, c in corners] + [f"패치 따라가기<br>(우연 보정) {c}" for _, c in corners],
                     model_rows(lambda st, q: [td_ms(f"psr_{c}.{st}.{q}") for c, _ in corners]
                                + [td_ms(f"follow_excess_{c}.{st}.{q}") for c, _ in corners]),
                     "패치를 다른 구석으로 옮겼을 때 (재학습 없음). 값: seed 3개의 평균, 작은 글씨는 seed 표준편차. "
                     "PSR 단위 %, 패치 따라가기(우연 보정) 단위 %p.")
    T["size"] = table(["모델", "PSR<br>12px", "PSR<br>24px (학습)", "PSR<br>48px", "IG 패치 몫/넓이<br>12px",
                       "IG 패치 몫/넓이<br>24px", "IG 패치 몫/넓이<br>48px"],
                      model_rows(lambda st, q: [td_ms(f"psr_s{s}.{st}.{q}") for s in (12, 24, 48)]
                                 + [td_ms(f"ig.r_norm_s{s}_mean.{st}.{q}", 1, 2) for s in (12, 24, 48)]),
                      "패치 크기를 바꿨을 때 (224px 기준 한 변, 재학습 없음). PSR 단위 %. "
                      "IG 패치 몫/넓이 = IG 패치 몫(R_patch) ÷ 패치 넓이 비율 (1이면 넓이만큼, 단위 배). IG는 사진 500장.")
    T["ctrl"] = table(["모델", "PSR<br>(x+ vs x−)", "패치 제거 전환율<br>(x+ vs 없음)", "x+ 정확도", "패치 없음 정확도",
                       "회색 가림 전환율<br>가운데", "회색 가림 전환율<br>왼쪽 위"],
                      model_rows(lambda st, q: [td_ms(f"{k}.{st}.{q}") for k in
                                               ("psr", "psr_none", "acc_match", "acc_none", "psr_occ_center",
                                                "psr_occ_tl")]),
                      "패치 제거와 같은 넓이 회색 가림. 값: seed 3개의 평균, 작은 글씨는 seed 표준편차. 단위 %.")
    blurs = (("b0", "원본"), ("b4", "σ=4"), ("b8", "σ=8"), ("b16", "σ=16"))
    T["blur"] = table(["모델"] + [f"패치 없음 정확도<br>{b}" for _, b in blurs]
                      + [f"패치 따라가기<br>(우연 보정) {b}" for _, b in blurs],
                      model_rows(lambda st, q: [td_ms(f"acc_none_{b}.{st}.{q}") for b, _ in blurs]
                                 + [td_ms(f"follow_excess_{b}.{st}.{q}") for b, _ in blurs]),
                      "물체 사진만 흐리게 하고 패치는 선명하게 붙였을 때 (σ: 224px 기준 가우시안 흐림 폭). "
                      "패치 따라가기(우연 보정)의 우연 수준은 같은 흐림의 패치 없는 사진에서 그 클래스를 고른 비율이다. "
                      "값: seed 3개의 평균, 작은 글씨는 seed 표준편차. 정확도 단위 %, 패치 따라가기 단위 %p.")

    def ig_cells(st, q):
        g = S["groups"].get(f"{st}|0.{q}", {}).get("ig") or {}
        cis = g.get("rho_gray_ci_per_seed") or []
        per_seed = "<small>seed별 " + " ".join(f"[{fx(lo, 2)}, {fx(hi, 2)}]" for lo, hi in cis) + "</small>" if cis else ""
        return [td_ms(f"ig.rho_gray.{st}.{q}", 1, 2, ci=True, extra=per_seed),
                td_ms(f"ig.rho_gray_within.{st}.{q}", 1, 2, ci=True), td_ms(f"ig.rho_gray_abs.{st}.{q}", 1, 2, ci=True),
                td_ms(f"ig.rho_gray_abs_within.{st}.{q}", 1, 2, ci=True),
                td_ms(f"ig.rho_blur.{st}.{q}", 1, 2), td_ms(f"ig.rho_gray_vs_blur.{st}.{q}", 1, 2)]
    T["ig"] = table(["모델", "ρ(R_patch, |Δy|)<br>회색 기준", "ρ 클래스 안<br>(회색)", "ρ 패치 안 |A| 합<br>(회색)",
                     "ρ 패치 안 |A| 합<br>클래스 안", "ρ(R_patch, |Δy|)<br>흐림 기준", "두 기준 R_patch의<br>ρ"],
                    model_rows(ig_cells),
                    "IG 패치 몫과 패치 민감도(|Δy|)의 순위 상관 (테스트 1,000장, 클래스마다 100장). ρ = Spearman 순위 상관. "
                    "값: seed 3개의 평균 ± seed 표준편차. 작은 글씨 [ ]: seed 평균의 95% 구간(1,000장을 클래스 안에서 함께 "
                    "1,000번 재추출, 사진 표집만 반영). seed별: 모델마다 1,000장 bootstrap 95% 구간. "
                    "클래스 안 = 클래스 10개 각각에서 구한 ρ의 평균(클래스끼리의 차이에서 생긴 ρ를 뺀다). "
                    "패치 안 |A| 합 = 전체 합과 넓이로 나누지 않은 패치 안 기여 크기(측정 방식을 바꿔 본 확인).",
                    empty=pending_pair(*STRATS))

    def igq_cells(st, q):
        return [td_ms(f"ig.r_norm_mean.{st}.{q}", 1, 2), td_ms(f"ig.r_norm_switched.{st}.{q}", 1, 2),
                td_ms(f"ig.r_norm_unswitched.{st}.{q}", 1, 2), td_ms(f"ig.n_switched.{st}.{q}", 1, 0),
                td_ms(f"ig.resid_gray_median.{st}.{q}", 100, 2), td_ms(f"ig.resid_gray_p90.{st}.{q}", 100, 2),
                td_ms(f"ig.resid_gray_gt5.{st}.{q}", 100, 1), td_ms(f"ig.conv_rho_64_128.{st}.{q}", 1, 4)]
    T["igq"] = table(["모델", "IG 패치 몫/넓이<br>평균", "IG 패치 몫/넓이<br>전환된 사진", "IG 패치 몫/넓이<br>전환 안 된 사진",
                      "전환된 사진 수<br>(1,000장 중)", "완전성 잔차<br>중간값 (%)", "완전성 잔차<br>90백분위 (%)",
                      "잔차 5% 넘는<br>사진 (%)", "64 vs 128단계<br>ρ (200장)"],
                     model_rows(igq_cells),
                     "IG 패치 몫과 계산 품질 (회색 기준). 패치 몫/넓이 단위 배. 전환된 사진 = x+와 x−의 예측이 다른 사진. "
                     "값: seed 3개의 평균 ± seed 표준편차. 'seed n개' = 값이 있는 seed 수(전환된 사진이 없는 seed는 빠진다).",
                     empty=pending_pair(*STRATS))
    rows = []
    gkeys = ("rho_gray", "rho_gray_within", "rho_gray_abs", "rho_gray_abs_within")
    for a, b in (("lp", "ft"), ("lp", "scratch"), ("ft", "scratch")):
        for i, p in enumerate(PS):
            lab_p = [f'<td class="lab">{NAME[a]} − {NAME[b]}</td>', f'<td class="lab">p={p:.2f}</td>']
            if has(f"igc.rho_gray.{a}-{b}.{p2(p)}"):
                rows.append((i == 0 and a + b != "lpft", lab_p + [td_diff(f"igc.{k}.{a}-{b}.{p2(p)}", 1, 2) for k in gkeys]))
            elif started(a, b):
                rows.append((i == 0 and a + b != "lpft", missing_row(lab_p, len(gkeys))))
    T["igc"] = table(["비교 (a − b)", "학습 p", "ρ 회색 기준", "ρ 클래스 안", "ρ 패치 안 |A| 합", "ρ 패치 안 |A| 합<br>클래스 안"],
                     rows,
                     "전략 사이 IG 상관(ρ)의 차이. 큰 글씨: seed 평균끼리의 차이(비교하는 칸에 모두 있는 seed만 쓴다). "
                     "괄호: 1,000장을 클래스 안에서 함께 재추출한 95% 구간(사진 표집만 반영). 그 아래 줄: seed 0·1·2 각각의 차이. "
                     "굵은 글씨: 구간이 0을 포함하지 않고, seed별 차이도 모두 같은 부호인 차이.",
                     empty=pending_pair(*STRATS))
    ml = S.get("model_level") or {}
    rows = []

    def ml_cells(d):
        return [f'<td class="num">{fx(d[k], 2) if d and d.get(k) is not None and np.isfinite(d[k]) else "–"}</td>'
                for k in ("rnorm_psr", "rnorm_follow_excess", "rblur_psr")]
    if ml:
        rows.append((False, [f'<td class="lab">모델 전체 ({ml["n"]}개)</td>'] + ml_cells(ml["all"])))
        if ml.get("cond"):
            rows.append((False, [f'<td class="lab">조건 평균 ({ml["cond"]["n"]}개: 전략 × p)</td>'] + ml_cells(ml["cond"])))
        for i, (p, d) in enumerate((ml.get("within_p") or {}).items()):
            rows.append((i == 0, [f'<td class="lab">p = {p} 안 (모델 {d["n"]}개)</td>'] + ml_cells(d)))
    sd_max, lo_psr, hi_psr = psr_spread()
    T["ml"] = table(["묶음", "ρ(IG 패치 몫/넓이, PSR)", "ρ(IG 패치 몫/넓이,<br>패치 따라가기(우연 보정))",
                     "ρ(흐림 기준 IG 패치 몫/넓이,<br>PSR)"], rows,
                    f"모델 단위 순위 상관(IG는 회색 기준, 마지막 열만 흐림 기준). 같은 조건 seed 3개의 PSR 표준편차는 최대 "
                    f"{fx(100 * sd_max, 1)}%p인데, 조건(전략 × p) 사이의 PSR은 {fx(100 * lo_psr, 1)}~{fx(100 * hi_psr, 1)}%로 "
                    "벌어진다. 그래서 모델 전체의 ρ는 조건 사이의 차이를 반영한다(p를 올리면 PSR과 IG 패치 몫이 함께 오른다). "
                    "조건 평균과 같은 p 안의 ρ를 함께 적었다. 점이 9~27개뿐이라 구간과 통계 검정 결과는 내지 않는다.")

    def train_cells(st, q):
        k = f"{st}.{q}"
        eps = R.get(f"best_epoch.{k}.values") or []
        ep_total = S["tuned"][st]["epochs"] if S.get("tuned") else "?"
        ep = f'<td class="num">{" · ".join(str(int(e)) for e in eps)} <span class="sd">/ {ep_total}</span></td>'
        sec = f'<td class="num">{R[f"train_sec.{k}"] / 60:.1f}</td>' if has(f"train_sec.{k}") else '<td class="num">–</td>'
        return [td_ms(f"best_val_acc.{k}"), ep, td_ms(f"final_train_acc.{k}"), td_ms(f"final_train_loss.{k}", 1, 3), sec]
    n_runs = S["n_models"]
    T["train"] = table(["모델", "검증 정확도<br>(고른 에폭, %)", "고른 에폭<br>(seed별 / 전체)", "마지막 에폭<br>학습 정확도 (%)",
                        "마지막 에폭<br>학습 손실", "학습 시간<br>(분/회)"], model_rows(train_cells),
                       f"본 학습 {'27회' if n_runs >= 27 else f'27회 중 끝난 {n_runs}회'}. 검증 = 학습과 같은 p로 패치를 붙인 "
                       "검증 5,000장. 학습 정확도는 무작위 자르기·뒤집기가 들어간 학습 배치에서 잰 값이다.")
    rows = []
    for st in STRATS:
        tu = (S.get("tuned") or {}).get(st)
        if not tu:
            continue
        for i, r in enumerate(sorted(tu["grid"], key=lambda r: (r["lr"], r["wd"]))):
            pick = r["lr"] == tu["lr"] and r["wd"] == tu["wd"]
            note = "선택" if pick else ("넓혀 본 값" if r["lr"] in tu["extended_lrs"] else "")
            rows.append((i == 0 and st != STRATS[0],
                         [lab(st) if i == 0 else '<td class="lab"></td>', f'<td class="num">{r["lr"]:g}</td>',
                          f'<td class="num">{r["wd"]:g}</td>',
                          f'<td class="num{" hit" if pick else ""}">{100 * r["best_val_acc"]:.2f}</td>',
                          f'<td class="num">{r["best_epoch"]}</td>', f'<td class="lab">{note}</td>']))
    T["tune"] = table(["전략", "학습률", "weight decay", "검증 정확도 (%)", "고른 에폭", ""], rows,
                      "학습률 튜닝 (p=0.10, 튜닝 전용 seed 100, 모델 하나씩). 넓혀 본 값 = 격자 끝 규칙으로 추가한 학습률.")
    rows = []
    for key in sorted(S.get("precision") or {}, key=lambda k: (STRATS.index(k.split("|")[0]), k)):
        x = S["precision"][key]
        st, p, s = key.split("|")
        rows.append((False, [lab(st, float(p)), f'<td class="num">{fx(x["rho_tf32_vs_ieee"], 4)}</td>',
                             f'<td class="num">{fx(100 * x["rel_diff_median"], 2)}</td>',
                             f'<td class="num">{fx(100 * x["rel_diff_p95"], 2)}</td>',
                             f'<td class="num">{fx(x["rho_absdy_tf32"], 3)}</td>', f'<td class="num">{fx(x["rho_absdy_ieee"], 3)}</td>',
                             f'<td class="num">{fx(100 * x["resid_tf32_median"], 2)}</td>',
                             f'<td class="num">{fx(100 * x["resid_ieee_median"], 2)}</td>']))
    T["prec"] = table(["모델 (seed 0)", "R_patch 순위 상관<br>TF32 vs IEEE", "R_patch 상대 차이<br>중간값 (%)",
                       "상대 차이<br>95백분위 (%)", "ρ(R_patch, |Δy|)<br>TF32", "ρ(R_patch, |Δy|)<br>IEEE", "완전성 잔차<br>TF32 (%)",
                       "완전성 잔차<br>IEEE (%)"], rows,
                      "같은 사진 200장의 IG(회색 기준, 64단계)를 두 정밀도로 계산한 차이.",
                      empty="아직 결과 없음 (본 학습 27회가 모두 끝난 뒤 driver.py가 잰다)")
    fu = S.get("followup")
    T["lplong"] = ""
    if fu:
        cfg, tu, q = fu["config"], (S.get("tuned") or {}).get("lp") or {}, p2(fu["config"]["p"])
        rows = []
        for k, label, diff in (("final_train_acc", "마지막 에폭 학습 정확도", False),
                               ("best_val_acc", "검증 정확도 (고른 에폭)", False), ("acc_id", "ID 정확도", True),
                               ("psr", "PSR (주 지표)", True), ("follow_excess", "패치 따라가기 (우연 보정, %p)", True),
                               ("dprob", "정답 확률 변화 (%p)", True), ("psr_none", "패치 제거 전환율", True),
                               ("acc_conf", "충돌 OOD 정확도", True)):
            cells = [f'<td class="lab">{label}</td>', td_ms(f"{k}.lp.{q}"), td_ms(f"{k}.lplong.{q}"), td_ms(f"{k}.ft.{q}")]
            cells += ([td_diff(f"c.{k}.lplong-lp.{q}"), td_diff(f"c.{k}.lplong-ft.{q}")] if diff
                      else ['<td class="num">–</td>'] * 2)
            rows.append((k == "acc_id", cells))
        T["lplong"] = table(
            ["지표", f"LP 본 실험<br>({tu.get('epochs')}에폭, weight decay {tu.get('wd'):g})",
             f"LP 후속<br>({cfg['epochs']}에폭, weight decay {cfg['wd']:g})", "FT 본 실험", "후속 LP<br>− 본 LP", "후속 LP<br>− FT"], rows,
            f"후속 실험, p = {cfg['p']:.2f}, seed 0·1·2. 후속 LP는 본 LP와 seed(학습 패치 배정)·학습률({cfg['lr']:g})·batch가 같고 "
            f"에폭({tu.get('epochs')} → {cfg['epochs']})과 weight decay({tu.get('wd'):g} → {cfg['wd']:g})만 다르다. "
            "값 열: seed 3개의 평균, 작은 글씨는 seed 표준편차(단위 %, 패치 따라가기(우연 보정)·정답 확률 변화는 %p). 차이 열: "
            + CAP_DIFF
            + " 학습·검증 정확도는 사진마다 잰 값이 아니라 차이 열을 비웠다.")
    return T


# ---------------------------------------------------------------- 그림
CAPTIONS = {
    "fig0_inputs": "입력 예시. (a) 클래스 k와 패치 색 k의 짝. (b) 테스트 사진 두 장의 변형: 정답 색 x+, 틀린 색 x−, 패치 없음, "
                   "회색 가림, 다른 위치·크기, 흐림. (c) p=0.99 학습 배치 일부 (자르기·뒤집기 뒤에 패치).",
    "fig1_accuracy": "테스트 정확도 4조건. 가로축 = 학습 때의 p, 점 = seed 3개 평균, 막대 = seed 표준편차.",
    "fig2_patch_reliance": "패치 의존 지표. 왼쪽부터 PSR(x+와 x−의 예측이 다른 비율), 패치 따라가기(우연 보정), "
                           "패치 제거 전환율(x+와 패치 없음의 예측이 다른 비율), 회색 가림 전환율(가운데).",
    "fig3_position": "패치 위치를 바꿨을 때의 PSR. 학습 때 패치는 항상 오른쪽 아래였다.",
    "fig4_size": "패치 크기를 바꿨을 때. 왼쪽 둘: PSR. 오른쪽 둘: IG 패치 몫/넓이(IG 패치 몫 ÷ 패치 넓이 비율, 사진 500장).",
    "fig5_blur": "물체 사진을 흐리게 하고 패치는 선명하게 붙였을 때. 위: p=0.90, 아래: p=0.99. "
                 "왼쪽부터 패치 없음 정확도, 충돌(x−) 정확도, 패치 따라가기(우연 보정: 같은 흐림의 패치 없는 사진이 우연 수준).",
    "fig6_curves": "본 학습 곡선. 색 실선 = 검증 정확도(학습과 같은 p), 회색 점선 = 학습 정확도(자르기·뒤집기 포함), "
                   "점 = 고른 체크포인트. 한 칸에 seed 3개.",
    "fig13_lplong": "후속 실험의 학습 곡선 (p = {FU_P}, LP, seed마다 선 하나). 파랑 = 본 실험({LP_EP}에폭, weight decay {LP_WD}), "
                    "진한 남색 = 후속 실험({FU_EP}에폭, weight decay {FU_WD}). 왼쪽 = 학습 정확도(자르기·뒤집기 포함), 오른쪽 = 검증 정확도"
                    "(학습과 같은 p), 점 = 고른 체크포인트. 점선 = 패치만 보고 답해도 맞히는 비율({FU_P100}%).",
    "fig7_ig_scatter": "seed 0 모델마다 테스트 1,000장. 가로 = |Δy|(패치 색을 바꿨을 때 정답 클래스 점수가 바뀐 크기), "
                       "세로 = IG 패치 몫/넓이(로그 눈금). ρ = Spearman 순위 상관 [bootstrap 95% 구간].",
    "fig8_ig_rho": "모델 안 순위 상관 요약 (seed 3개 평균 ± 표준편차). 오른쪽: 회색 기준과 흐림 기준이 매긴 R_patch 순위가 "
                   "얼마나 같은지.",
    "fig9_model_level": "학습한 모델 하나가 점 하나. 가로 = PSR(행동으로 잰 패치 의존), 세로 = 평균 IG 패치 몫/넓이(로그 눈금). "
                        "모양 = 학습 p. 같은 조건의 seed 3개는 서로 가깝다(PSR 표준편차 최대 {PSR_SD_MAX}%p).",
    "fig10_heatmaps_lp": "LP(seed 0)의 IG 히트맵(heatmap: 픽셀마다 기여 크기를 색 진하기로 칠한 그림). "
                         "열: |Δy| 하위(5·10·15백분위), 중간(45·50·55), 상위(85·90·95). "
                         "윗줄 = 입력 x+, 아랫줄 = 픽셀별 기여 크기 |A| (진할수록 큼). 검은 네모 = 패치 자리. "
                         "색 범위는 칸마다 따로(그 칸의 99.5백분위까지)라 칸끼리 진하기를 비교하지 않는다.",
    "fig10_heatmaps_ft": "FT(seed 0)의 IG 히트맵. 읽는 법과 색 범위는 LP 그림과 같다.",
    "fig10_heatmaps_scratch": "Scratch(seed 0)의 IG 히트맵. 읽는 법과 색 범위는 LP 그림과 같다.",
    "fig11_common_heatmaps": "모든 모델(seed 0)에 같은 테스트 사진 6장을 넣은 IG 히트맵. 맨 윗줄 = 입력 x+(열 제목 = 클래스), "
                             "아래 줄마다 모델 하나. 색 범위는 칸마다 따로라 패치 안이 주변보다 진한지만 본다.",
    "fig12_cases_p0.99": "p=0.99 모델(seed 0)에서 IG와 패치 민감도가 어긋난 사진. 왼쪽 3장 = IG가 놓친 사례(|Δy| 순위는 높은데 "
                         "R_patch 순위가 낮음), 오른쪽 3장 = IG가 헛짚은 사례(R_patch 순위는 높은데 |Δy| 순위가 낮음). "
                         "순위는 그 모델의 1,000장 안에서의 백분위. 고르는 규칙은 결과를 보기 전에 정했다(ig_cases.py).",
    "fig12_cases_p0.90": "p=0.90 모델(seed 0)에서 같은 규칙으로 고른 사례.",
}


PHOTO = ("fig0_", "fig10_", "fig11_", "fig12_")  # 사진·히트맵 그림 (JPEG로 넣는다)
CASES_STEP = "본 학습이 모두 끝난 뒤 ig_cases.py → analyze.py → report.py 순으로 돌리면 채워진다."
PENDING_FIG = {  # 그림이 아직 없을 때: (짧은 제목, 채워지는 단계)
    "fig0_inputs": ("입력 예시", "figure_inputs.py를 돌린 뒤 report.py를 다시 돌리면 채워진다."),
    "fig10_heatmaps_scratch": ("Scratch의 IG 히트맵", "Scratch의 본 학습과 IG가 끝난 뒤 analyze.py → report.py를 돌리면 채워진다."),
    "fig12_cases_p0.99": ("p=0.99 모델에서 IG가 어긋난 사례", CASES_STEP),
    "fig12_cases_p0.90": ("p=0.90 모델에서 IG가 어긋난 사례", CASES_STEP),
}


def img_data(path):
    """선 그림은 PNG 그대로, 사진 같은 그림은 JPEG(품질 85, 4:4:4: 색 해상도를 줄이지 않음)로."""
    if not path.name.startswith(PHOTO):
        return "image/png", path.read_bytes()
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=85, subsampling=0, optimize=True)
    return "image/jpeg", buf.getvalue()


def build_figures():
    F = {}
    fill = {"{PSR_SD_MAX}": fx(100 * psr_spread()[0], 1)}
    fu, tu = S.get("followup"), (S.get("tuned") or {}).get("lp")
    if fu and tu:  # 후속 실험 그림의 설정값
        fill.update({"{FU_P}": f"{fu['config']['p']:.2f}", "{FU_P100}": f"{100 * fu['config']['p']:g}",
                     "{FU_EP}": str(fu["config"]["epochs"]), "{FU_WD}": f"{fu['config']['wd']:g}",
                     "{LP_EP}": str(tu["epochs"]), "{LP_WD}": f"{tu['wd']:g}"})
    for name, cap in CAPTIONS.items():
        for a, b in fill.items():
            cap = cap.replace(a, b)
        path = FIG / f"{name}.png"
        if not path.exists():
            title, step = PENDING_FIG.get(name, (cap.split(". ")[0].rstrip("."), "analyze.py → report.py를 다시 돌리면 채워진다."))
            F[name] = f'<p class="pending prose">그림 준비 중: {title}. {step}</p>'
            continue
        mime, data = img_data(path)
        b64 = base64.b64encode(data).decode()
        F[name] = (f'<figure class="plate"><img src="data:{mime};base64,{b64}" alt="{cap}" loading="lazy">'
                   f'<figcaption>{cap}</figcaption></figure>')
    return F


# ---------------------------------------------------------------- 고정 절 (결과와 무관한 내용)
def run_info():
    cfgs = sorted((ROOT / "runs" / "final").glob("*/config.json"))
    cfg = json.loads(cfgs[0].read_text()) if cfgs else {}
    log = (ROOT / "runs" / "driver.log").read_text(encoding="utf-8") if (ROOT / "runs" / "driver.log").exists() else ""
    t0 = re.search(r"\[([\d\- :]+)\] === stage tune start", log)
    t1 = re.findall(r"\[([\d\- :]+)\] === stage final end", log)
    train_sec = sum(r["sec"] for r in S.get("tuning") or []) + sum(m["done"]["total_sec"] for m in S["per_model"].values())
    # 후속 실험: 날짜는 실행 폴더의 시작·끝 시각, 시간은 seed마다 학습·평가에 쓴 초의 합 (다시 불러도 바뀌지 않는다)
    fu = S.get("followup") or {}
    fg = fu.get("group") or {}
    fu_train = sum(fg["train_sec"]["values"]) if fg.get("train_sec") else None
    fu_eval = sum(fg["eval_sec"]["values"]) if fg.get("eval_sec") else None
    return dict(torch=cfg.get("torch", "?"), gpu=cfg.get("gpu", "?"), start=t0.group(1) if t0 else "?",
                end=t1[-1] if t1 else "진행 중", train_hours=train_sec / 3600,
                fu_dates=(fu["started"][:10], fu["finished"][:10]) if fu.get("started") and fu.get("finished") else None,
                fu_train_min=fu_train / 60 if fu_train is not None else None,
                fu_total_min=(fu_train + fu_eval) / 60 if fu_train is not None and fu_eval is not None else None)


def design_section(F):
    tu = S.get("tuned") or {}

    def cfg(st, k, fmt="{:g}"):
        return fmt.format(tu[st][k]) if st in tu else "–"
    rows = [(False, [lab("lp"), '<td class="lab">ImageNet 사전학습</td>', '<td class="lab">마지막 층(head)만</td>',
                     f'<td class="num">{N_PARAMS["lp"]:,}</td>', f'<td class="num">{cfg("lp", "lr")}</td>',
                     f'<td class="num">{cfg("lp", "wd")}</td>', f'<td class="num">{cfg("lp", "epochs")}</td>']),
            (False, [lab("ft"), '<td class="lab">ImageNet 사전학습 (LP와 같은 가중치)</td>', '<td class="lab">모든 층</td>',
                     f'<td class="num">{N_PARAMS["ft"]:,}</td>', f'<td class="num">{cfg("ft", "lr")}</td>',
                     f'<td class="num">{cfg("ft", "wd")}</td>', f'<td class="num">{cfg("ft", "epochs")}</td>']),
            (False, [lab("scratch"), '<td class="lab">무작위</td>', '<td class="lab">모든 층</td>',
                     f'<td class="num">{N_PARAMS["scratch"]:,}</td>', f'<td class="num">{cfg("scratch", "lr")}</td>',
                     f'<td class="num">{cfg("scratch", "wd")}</td>', f'<td class="num">{cfg("scratch", "epochs")}</td>'])]
    strat_table = table(["전략", "시작 가중치", "학습하는 층", "학습하는 가중치 수", "학습률", "weight decay", "에폭"], rows,
                        "세 전략. 모두 같은 ResNet-18 구조에 클래스 10개짜리 새 head를 단다. 학습률(한 번에 가중치를 움직이는 크기)과 "
                        "weight decay(가중치 감쇠: 가중치를 0 쪽으로 당기는 세기)는 튜닝으로 고른 값.")
    gloss = [
        ("가중치 (weight)", f"모델 안에서 학습으로 정해지는 숫자. 학습은 이 숫자를 조금씩 고쳐 정답을 더 잘 맞히게 하는 과정이다. "
                          f"이 실험의 ResNet-18은 새 head까지 {N_PARAMS['ft']:,}개."),
        ("층 (layer)", "입력값에 가중치를 곱하고 더하는 계산 한 단계. 신경망은 층을 차례로 거쳐 사진을 클래스 점수로 바꾼다."),
        ("손실 (loss)", "모델의 답이 정답에서 얼마나 먼지를 잰 값. 학습은 이 값을 줄이는 쪽으로 가중치를 고친다. 정답 확률이 1이면 0이다."),
        ("convolution (합성곱)", "작은 필터를 사진 위에서 옮겨 가며 곱하고 더하는 계산. backbone이 사진을 특징으로 바꿀 때 하는 계산이다."),
        ("ResNet-18", "사진 분류에 널리 쓰는 신경망 구조. 합성곱 층 17개와 마지막 층 1개를 쌓았다."),
        ("ImageNet (이미지넷)", "물체 사진 약 128만 장을 1,000가지로 나눈 큰 데이터."),
        ("사전학습 (pretraining)", "큰 데이터(여기서는 ImageNet)로 미리 학습해 둔 가중치를 다른 일의 출발점으로 쓰는 것."),
        ("shortcut (지름길)", "정답과 우연히 함께 나타나는 쉬운 단서. 여기서는 오른쪽 아래 24×24 색 패치."),
        ("backbone (몸통)", "사진을 특징값 512개로 바꾸는 앞쪽 층 전체."),
        ("head (머리)", "특징값 512개를 클래스 점수 10개로 바꾸는 마지막 층 하나 (가중치 5,130개)."),
        ("LP, linear probing (선형 탐사)", "사전학습 backbone을 얼리고 head만 학습. BatchNorm(층마다 출력값의 평균·크기를 맞추는 단계)이 "
                                       "쓰는 평균·분산도 사전학습 값으로 고정."),
        ("FT, full fine-tuning (전체 미세조정)", "LP와 같은 사전학습 가중치에서 시작해 모든 층을 학습."),
        ("Scratch (처음부터 학습)", "무작위 가중치에서 모든 층을 학습. 계획서의 참고 기준."),
        ("p (상관 강도)", "학습 사진에 정답 클래스의 색이 붙을 확률. 0.10 = 무관(10색 중 아무거나와 같음), 0.99 = 100장 중 99장."),
        ("x+ / x−", "같은 테스트 사진에 정답 색 / 틀린 색(9색 중 하나) 패치를 붙인 것."),
        ("ID / OOD", "ID(in-distribution, 학습 분포) = 학습과 같은 p의 테스트. OOD(out-of-distribution, 학습 분포 밖) = 다른 p."),
        ("PSR, prediction switch rate (예측 전환율)", "x+와 x−의 예측이 다른 사진의 비율. 패치 색만 바꿨는데 답이 바뀐 비율. 주 지표."),
        ("패치 따라가기 (보정 전 / 우연 보정)", "보정 전 = x−에서 모델이 패치 색이 가리키는 (틀린) 클래스를 고른 비율. "
         "우연 보정 = 보정 전 − 우연 수준(패치가 없을 때 그 클래스를 고른 비율). 모델끼리 비교할 때는 우연 보정을 쓴다. "
         "정확도가 낮은 모델은 우연 수준 자체가 높기 때문이다."),
        ("차이의 차이 (DiD, difference-in-differences)", "두 전략의 차이가 p = 0.10일 때와 비교해 얼마나 바뀌었나. "
         "같은 지표의 p = 0.10 차이(예: LP − FT의 PSR 차이)를 빼서 p를 올린 효과만 남긴다."),
        ("logit (로짓)", "softmax(클래스 점수 10개를 합이 1인 확률로 바꾸는 계산)를 거치기 전의 클래스 점수. z_y = 정답 클래스 y의 점수."),
        ("Δy", "z_y(x+) − z_y(x−). 패치 색을 바꿨을 때 정답 점수가 바뀐 양. 크기 |Δy|를 패치 민감도라고 부른다. "
               "|Δy|는 모델 안에서만 비교한다 (logit 크기는 모델마다 달라서)."),
        ("정답 확률 변화", "정답 클래스 확률의 x+ − x−. 모델끼리 비교할 때 Δy 대신 쓴다."),
        ("기울기 (gradient)", "픽셀이나 가중치 같은 값을 아주 조금 바꿀 때 점수(또는 손실)가 바뀌는 정도."),
        ("IG, Integrated Gradients (적분 기울기)", "기준 이미지에서 실제 사진까지 직선으로 64걸음 가며, 걸음마다 픽셀별 기울기를 잰다. "
         "픽셀 하나의 기여 A는 (그 픽셀값 − 기준 이미지의 값) × 64걸음 기울기의 평균이다. 모든 픽셀의 기여를 더하면 "
         "'실제 점수 − 기준 점수'가 된다(64걸음 근사의 오차가 완전성 잔차). 그래서 기준과 값이 같은 픽셀은 기여가 0이다."),
        ("TF32", "NVIDIA GPU가 32비트 곱셈을 빠르게 하려고 유효 숫자를 약 3자리로 줄여 곱하는 방식. 덧셈은 32비트 그대로 한다. "
                 "IEEE fp32는 줄이지 않는 표준 32비트 계산이다."),
        ("baseline (기준 이미지)", "IG의 출발점. 주 기준은 회색(픽셀값을 0~1로 나타낼 때 0.5)이다. 기준 이미지를 바꿔도 결과가 같은지 "
                                   "보려고, 같은 사진을 가우시안 흐림(가까운 픽셀일수록 크게 쳐서 평균 내는 흐림, 폭 σ = 16픽셀, 224px 기준)으로 "
                                   "흐린 것(흐림 기준)도 썼다."),
        ("R_patch (IG 패치 몫)", "모든 픽셀의 기여 크기 |A| 합 중 패치 안의 몫. 'IG 패치 몫/넓이'는 이것을 패치 넓이 비율로 나눈 값이고, "
         "1이면 패치가 넓이만큼만 기여한 것이다. 다만 1이 '패치를 안 쓴다'는 기준은 아니다. 패치 색 10개 중 8개는 세 채널"
         "(빨강·초록·파랑 값, 0~1 눈금)이 모두 0 또는 1이라 회색(0.5)과의 차이가 커서, 기울기가 같아도 기여가 커질 수 있다. "
         "그래서 같은 전략의 p = 0.10 모델과 비교한다. IG 상관 = ρ(R_patch, |Δy|)."),
        ("패치 안 |A| 합", "IG 기여 크기를 패치 안에서만 더한 값. IG 패치 몫은 이것을 사진 전체의 합으로 나눈 비율이다."),
        ("히트맵 (heatmap)", "픽셀마다 IG 기여 크기를 색 진하기로 칠한 그림."),
        ("가림 약함", "패치 색과 상관없이 사진 일부가 가려지거나 바뀌기만 해도 답이 바뀌는 약함. 같은 넓이 회색 사각형을 덧대 잰다."),
        ("completeness residual (완전성 잔차)", "IG 기여의 합과 '실제 점수 − 기준 점수'의 상대 차이. 작을수록 적분이 정확."),
        ("Spearman ρ (순위 상관)", "두 값의 순위가 얼마나 같이 움직이는지 (−1~1). 1이면 순위가 똑같다."),
        ("클래스 안 ρ", "클래스 10개 각각에서 따로 구한 Spearman ρ의 평균. 클래스끼리의 차이(패치 색이 클래스마다 달라 몫도 다르다)를 "
         "빼고 같은 클래스 사진끼리의 관계만 본다."),
        ("bootstrap (부트스트랩)", "테스트 사진을 복원 추출로 다시 뽑아 수치의 흔들림을 재는 방법(정확도·PSR 같은 지표는 10,000장을 "
         f"{S['notes']['bootstrap_images']:,}번, IG 상관은 IG 1,000장을 {S['notes']['bootstrap_ig']:,}번; seed 평균과 전략 사이 차이는 "
         "클래스 안에서 함께 뽑는다). 가운데 95%를 구간으로 쓴다."),
        ("seed (시드)", "무작위 값(head의 처음 가중치, 학습 사진 순서 섞기, 패치 색 뽑기)을 정하는 번호. 본 학습은 같은 설정을 "
                        "seed 0·1·2로 세 번 했다."),
        ("체크포인트", "학습 도중 저장한 가중치. 이 실험은 에폭마다 검증 정확도를 재서 가장 높은 에폭의 가중치를 골랐다."),
        ("batch (묶음)", "가중치를 한 번 고칠 때 함께 보는 학습 사진 묶음. 여기서는 256장."),
        ("learning rate (학습률) / weight decay (가중치 감쇠)", "한 번에 가중치를 얼마나 움직일지 / 가중치를 0쪽으로 당기는 세기."),
        ("epoch (에폭)", "학습 사진 45,000장을 한 번 다 보는 것."),
    ]
    gl = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in gloss)
    return f"""
<p class="q">무엇을 어떻게 비교했나?</p>
<p class="a">같은 편향 데이터로 세 학습 방식(LP·FT·Scratch)의 ResNet-18을 학습시키고, 테스트 사진의 패치 색만 바꿨을 때 예측이 얼마나 흔들리는지 쟀다.</p>
<div class="prose">
<p><b>지름길(shortcut)</b>은 정답과 우연히 함께 나타나는 쉬운 단서다. 이 실험의 지름길은 사진 오른쪽 아래의 24×24 색 패치다.
클래스마다 색이 하나씩 정해져 있다(맨 위 색 띠). 학습 사진에는 확률 <b>p</b>로 정답 클래스의 색을 붙이고,
나머지 1−p는 틀린 9색 중 하나를 같은 확률로 붙인다(계획서 식 1).</p>
<ul class="tight">
<li>p = 0.10: 정답 색이 붙을 확률이 10색 중 아무거나 고를 때와 같다. 패치는 정답에 대해 아무것도 알려 주지 않는다(기준 조건).</li>
<li>p = 0.90: 학습 사진 10장 중 9장은 패치 색만 봐도 정답이다.</li>
<li>p = 0.99: 100장 중 99장이 그렇다. 물체를 보지 않고 패치만 봐도 학습 정확도 99%가 나온다.</li>
</ul>
<p>데이터는 CIFAR-10(32×32 컬러 사진, 10클래스)이다. 학습용 50,000장을 학습 45,000장과 검증 5,000장(클래스마다 500장)으로 고정해 나눴다.
테스트 10,000장은 학습에도 하이퍼파라미터(학습률처럼 학습 전에 사람이 정하는 설정값) 선택에도 쓰지 않았다. 모든 사진을 224×224로 키운 뒤 패치를 붙인다.
학습 때의 무작위 자르기·뒤집기는 패치를 붙이기 전에 하므로 패치가 잘려 나가지 않는다.</p>
</div>
{strat_table}
<div class="prose">
<p>세 전략의 head는 seed마다 같은 값으로 새로 초기화했다. LP와 FT는 같은 ImageNet 가중치(사진 모델 라이브러리 torchvision이 주는
IMAGENET1K_V1)에서 시작한다.
그래서 LP와 FT는 시작 가중치가 같고, backbone을 얼렸는가와 그에 맞춰 따로 고른 학습 설정(학습률 {cfg("lp", "lr")}·{cfg("ft", "lr")},
에폭 {cfg("lp", "epochs")}·{cfg("ft", "epochs")})이 다르다. Main RQ의 직접 비교는 LP vs FT다.
Scratch vs LP는 시작 가중치와 학습하는 층이 둘 다 다르다. 그래서 둘의 차이가 얼린 것 때문인지 사전학습 가중치 때문인지 가를 수 없고,
실제로 쓰는 두 학습 방식을 비교한 결과로만 읽는다(계획서의 해석 경계).</p>
<p>평가는 테스트 사진 10,000장마다 변형 24개를 만들어 같은 사진의 변형끼리 짝지어 한다.
틀린 색 w와 "정답 색을 붙일지"를 정하는 난수 u는 사진마다 한 번 뽑아 모든 모델에 똑같이 쓴다. 정확도는 네 조건에서 잰다.</p>
<ul class="tight">
<li><b>ID</b>: 학습과 같은 p로 패치를 붙인 테스트.</li>
<li><b>독립 OOD</b>(out-of-distribution, 학습 분포 밖): p = 0.10의 규칙으로 패치를 붙인 테스트. 패치가 정답과 무관하다.</li>
<li><b>충돌 OOD</b>: 모든 사진에 틀린 색 패치(x−). 패치가 오답 클래스를 가리킨다.</li>
<li><b>패치 없음</b>: 패치를 붙이지 않은 원래 사진(224×224로 키운 것).</li>
</ul>
<p>패치 의존은 같은 사진에 정답 색 패치를 붙인 <b>x+</b>와 틀린 색 패치를 붙인 <b>x−</b>를 비교해 잰다.
주 지표는 PSR(둘의 예측이 다른 사진의 비율)이고, 정답 확률 변화와 패치 따라가기(보정 전·우연 보정)를 함께 본다.</p>
</div>
{F["fig0_inputs"]}
<h3 id="terms">용어 풀이</h3>
<dl class="gl">{gl}</dl>
"""


ADDED = [  # 계획서에 없던 분석: (이름, 무엇을 왜, 어느 판정에 쓰였나)
    ("패치 따라가기(보정 전·우연 보정).", "x−에서 틀린 색이 가리키는 클래스로 답한 비율(보정 전)과, 거기서 패치가 없을 때 그 클래스를 "
     "고른 비율을 사진마다 뺀 값(우연 보정). 계획서의 짝 비교 지표는 PSR·Δy·정답 확률 변화뿐이라 둘 다 새로 더했다. 정확도가 낮은 "
     "모델일수록 우연 수준이 높아서 뺐다. Main RQ(세 p 모두)·H1·H2, SQ1의 p = 0.90 판정, 위치·흐림 시험, 모델 단위 비교, 한계 절에 쓰였다."),
    ("차이의 차이.", "두 전략의 차이에서 p = 0.10일 때의 차이를 뺐다. SQ1(p의 효과)을 재는 방법으로 정했다."),
    ("전환 방향 나누기.", "PSR을 '패치 색의 클래스로 바뀜'과 '다른 클래스로 바뀜'으로 나눴다. Main RQ의 p = 0.90 해석에 쓰였다."),
    ("클래스 안 ρ.", "계획서의 ρ(R_patch, |Δy|)를 클래스 10개 각각에서 구해 평균했다. 클래스마다 패치 색이 달라서다. H3·SQ2 판정에 쓰였다."),
    ("패치 안 |A| 합의 ρ.", "R_patch(비율) 대신 패치 안 기여 크기로 같은 ρ를 구했다. H3의 '패치 안 기여의 크기'라는 말에 더 가깝다. "
     "H3·SQ2 판정에 쓰였다."),
    ("IG 상관의 전략 간 차이.", "1,000장을 클래스 안에서 함께 다시 뽑아 seed 평균 ρ의 차이와 구간을 냈다. H3·SQ2 판정에 쓰였다."),
    ("모델 단위 ρ.", "모델마다 평균 IG 패치 몫/넓이와 PSR의 순위 상관(모델 전체, 조건 평균, 같은 p 안). SQ2에 쓰였다."),
    ("|Δy| 순위의 seed 사이 ρ.", "같은 사진의 |Δy| 순위가 seed를 바꿔도 다시 나오는지. '한계' 절에 쓰였다."),
    ("seed 사이 t 구간.", "본문이 방향을 말하는 비교와 값 일부에 대해 seed 3개의 차이만으로 95% 구간(자유도 2)을 냈다. seed 3개가 "
     "모두 같은 부호인 것은 차이가 없어도 4분의 1의 확률로 생겨서다. '한계' 절에 쓰였다."),
]
if S.get("followup") and (S.get("tuned") or {}).get("lp"):
    _c, _t = S["followup"]["config"], S["tuned"]["lp"]
    _wd = "weight decay 없이" if _c["wd"] == 0 else f"weight decay를 바꿔(wd {_c['wd']:g})"  # 숫자 바로 뒤 조사를 피한다
    ADDED.append(("LP head를 더 오래 학습한 후속 실험.",
                  f"'학습이 충분했나' 절의 열린 질문(p = {_c['p']:.2f}의 LP가 패치를 덜 쓴 것이 head를 덜 학습한 탓인가)을 보려고, "
                  f"p = {_c['p']:.2f}의 LP를 같은 seed·학습률로 {_c['epochs']}에폭 동안 {_wd} 다시 학습해 "
                  f"평가했다(본 실험은 {_t['epochs']}에폭, weight decay {_t['wd']:g}. IG는 하지 않았다). "
                  "결과를 본 뒤 정한 실험이다. Main RQ 해석, '학습이 충분했나' 절, 한계 절에 쓰였다."))


def changes_section():
    tu = S.get("tuned") or {}
    ext = []
    for st in STRATS:
        if st not in tu:
            continue
        t = tu[st]
        if t["extended_lrs"]:
            best_by_lr = {}
            for r in t["grid"]:
                best_by_lr[r["lr"]] = max(best_by_lr.get(r["lr"], 0), r["best_val_acc"])
            path = " → ".join(f'{lr:g}({100 * best_by_lr[lr]:.2f}%)' for lr in sorted(best_by_lr))
            ext.append(f'<li>{NAME[st]}: 넓혀 본 학습률 {", ".join(f"{x:g}" for x in t["extended_lrs"])}. '
                       f'학습률별 최고 검증 정확도 {path}. 고른 값: 학습률 {t["lr"]:g}, weight decay {t["wd"]:g}.</li>')
    n_cfg = ", ".join(f"{NAME[st]} {len(tu[st]['grid'])}개" for st in STRATS if st in tu)
    edge = ""  # 넓힌 뒤에도 맨 위 학습률을 고른 전략 (사다리 끝이라 더 넓힐 수 없다)
    for st in STRATS:
        t = tu.get(st)
        if not t or not t["extended_lrs"] or t["lr"] != max(r["lr"] for r in t["grid"]):
            continue
        best = {}
        for r in t["grid"]:
            best[r["lr"]] = max(best.get(r["lr"], 0), r["best_val_acc"])
        lrs = sorted(best)
        steps = [best[lrs[i + 1]] - best[lrs[i]] for i in range(len(lrs) - 1)]
        mean3 = R.get(f"best_val_acc.{st}.10")  # 같은 설정(p = 0.10)을 본 학습 seed 3개로 돌린 평균
        txt = (f'{NAME[st]}는 {len(t["extended_lrs"])}번 넓힌 뒤에도 맨 위 학습률을 골랐다(학습률 {lrs[-1]:g}, '
               '미리 정해 둔 후보 목록에 더 큰 값이 없다). '
               f'학습률을 한 칸 올릴 때 검증 정확도가 오른 폭은 '
               + ", ".join(f"{lrs[i]:g} → {lrs[i + 1]:g}에서 {fx(100 * steps[i], 2, True)}%p"
                           for i in range(max(0, len(steps) - 2), len(steps)))
               + "다.")
        if mean3 is not None:
            txt += (f" 같은 설정을 본 학습 seed 3개로 돌린 평균은 {fx(100 * mean3, 2)}%로, 튜닝 한 번의 "
                    f"{fx(100 * best[lrs[-1]], 2)}%와 {fx(abs(100 * (best[lrs[-1]] - mean3)), 2)}%p 차이다.")
        if len(steps) >= 2 and 0 < steps[-1] < steps[-2] and mean3 is not None and mean3 <= best[lrs[-1]]:
            txt += " 오른 폭이 줄고 있고 마지막 폭이 seed를 바꿨을 때의 차이와 비슷한 크기라, 더 큰 학습률이 정확도를 크게 올렸을 가능성은 작다고 본다."
        edge += f"<li>{txt}</li>"
    n_prec = len(S.get("precision") or {})
    rho_min = min((x["rho_tf32_vs_ieee"] for x in (S.get("precision") or {}).values()), default=None)
    prec_txt = (f'차이는 <a href="#precision">IG 정밀도 확인</a> 절에서 모델 {n_prec}개로 쟀고, R_patch 순위 상관이 모두 '
                f'{fx(np.floor(rho_min * 1000) / 1000, 3)} 이상이었다.' if n_prec else
                '차이는 본 학습 27회가 끝난 뒤 seed 0 모델 6개로 잰다(<a href="#precision">IG 정밀도 확인</a> 절).')
    added = "".join(f"<li><b>{a}</b> {b}</li>" for a, b in ADDED)
    return f"""
<p class="q">계획서대로 했나?</p>
<p class="a">데이터·전략·주 지표(PSR)·seed 평균과 사진 bootstrap 구간은 계획서대로 했고, 튜닝 규칙과 그에 따른 튜닝 예산(설정 수 {n_cfg})·실행 순서·IG 정밀도를 바꿨으며, 계획서에 없던 분석 {len(ADDED)}가지를 더했다.</p>
<h3>바꾼 것: 튜닝 규칙과 튜닝 예산, 실행 순서, IG 정밀도</h3>
<ol class="prose">
<li><b>학습률 격자 끝 규칙 (튜닝 도중 추가).</b> 처음 격자는 전략마다 학습률 3개 × weight decay 2개 = 6개였다.
튜닝 도중 LP와 FT의 최고 학습률이 격자 위쪽 끝에 걸렸다. 그대로 쓰면 덜 학습된 모델을 "지름길에 강하다"고 잘못 읽을 수 있다(계획서가 경고한 혼동).
그래서 "최고 학습률이 격자 끝이면 그 방향으로 한 칸 더, weight decay 2개 모두 돌려 본다"는 규칙을 세 전략에 똑같이, 최대 2번 적용했다.
<ul class="tight">{"".join(ext)}{edge}</ul></li>
<li><b>튜닝 예산이 전략마다 달라졌다.</b> 위 규칙 때문에 설정 수가 {n_cfg}다. 계획서의 "같은 탐색 예산"은 처음 격자 6개까지만 지켜졌다.</li>
<li><b>실행 순서.</b> 계획서는 Scratch·LP 12회를 먼저 돌려 보고 넓히기로 했다. 실제로는 작은 부분집합으로 학습→평가→IG가 끝까지 도는지 확인한 뒤
27회를 한 번에 돌렸다(LP → FT → Scratch 순).</li>
<li><b>IG 정밀도.</b> 테스트 평가는 convolution까지 IEEE fp32(표준 32비트 부동소수점)로 했다.
IG는 값을 32비트로 저장한 채 convolution의 곱셈만 GPU의 TF32 방식으로 했다(곱셈 유효 숫자 약 3자리, IEEE보다 약 2.5배 빠름).
{prec_txt}</li>
</ol>
<h3>계획서에 없던 분석 {len(ADDED)}가지 (어느 판정에 쓰였나)</h3>
<ul class="prose tight">{added}</ul>
<h3>계획서에 수치가 없어서 정한 것: 튜닝 고정·에폭·패치 추출·가림·흐림·구간 계산</h3>
<ul class="prose tight">
<li>학습률·weight decay는 p = 0.10, 튜닝 전용 seed 100에서 골라 모든 p와 본 seed 0·1·2에 고정했다.
p마다 고르면 p의 효과(SQ1)와 튜닝의 효과가 섞이기 때문이다. 체크포인트(에폭)는 계획서대로 학습과 같은 p의 검증 정확도로 골랐다.</li>
<li>에폭: LP 15, FT 10, Scratch 40. 가중치를 고치는 규칙(옵티마이저)은 SGD(확률적 경사 하강법: 사진 묶음마다 손실의 기울기
반대쪽으로 가중치를 조금씩 옮긴다)이고, 이전 이동 방향을 0.9만큼 이어 가는 momentum과 Nesterov 방식(한 걸음 앞에서 기울기를 재는
momentum)을 썼다. 한 번에 보는 사진 묶음(batch)은 256장이다.
첫 1에폭 동안 학습률을 0에서 올리고(warmup) 그 뒤 cosine 곡선(처음과 끝에는 천천히, 중간에는 빠르게 줄어드는 곡선)으로 0까지 내린다.
학습은 곱셈 대부분을 bf16(16비트 수)으로 하는 자동 혼합 정밀도로 했다.</li>
<li>패치 추출은 p끼리 묶었다. 사진마다 난수 u와 틀린 색 w를 한 번 뽑고 "u &lt; p이면 정답 색, 아니면 w"로 정한다.
그래서 p = 0.90에서 정답 색이 붙은 사진은 p = 0.99에서도 정답 색이다. 같은 seed 안에서는 전략과 무관하게 같은 패치다.</li>
<li>무작위 자르기로 생기는 빈 곳은 반사 패딩(가장자리 픽셀을 거울처럼 비춰 채우기)으로 채웠다. 0으로 채우면 검은 띠가 생기는데, 검정이 클래스 7(말)의 패치 색이라서 피했다.</li>
<li>가림 대조군: x+에 같은 넓이(24×24)의 회색 사각형을 가운데 또는 왼쪽 위에 덧댔다.</li>
<li>물체 흐리기(계획서의 선택 항목): 224px 기준 가우시안 σ = 4, 8, 16. IG의 흐린 기준 이미지는 σ = 16.</li>
<li>IG 사진: 클래스마다 100장인 1,000장. 128단계 수렴 확인은 그중 200장, 패치 크기별 IG는 500장.</li>
<li>Bootstrap: 테스트 사진 10,000장을 2,000번 복원 추출했다. 모든 모델이 같은 추출 표를 쓰므로 모델 간 차이도 사진끼리 짝지어진다.
이 구간은 사진 표집만 반영한다. 학습 seed에 따른 흔들림은 같은 seed끼리의 차이(seed 줄)와 seed 표준편차로 따로 적었다(계획서 5절).</li>
<li>IG 상관의 구간: 모델마다 1,000장을 1,000번 복원 추출했다. seed 평균과 전략 사이 차이의 구간은 1,000장을 클래스 안에서
(클래스마다 100장씩) 함께 재추출하는 표 하나를 모든 모델이 같이 써서 짝지었다.</li>
</ul>
"""


def repro_section():
    info = run_info()
    partial = "" if S["n_models"] >= 27 else f" 지금까지 끝난 본 학습 {S['n_models']}회 기준."
    fc = (S.get("followup") or {}).get("config")
    fu_what = ((f"p = {fc['p']:.2f}의 LP를 같은 seed·학습률로 {fc['epochs']}에폭 동안 "
                + ("weight decay 없이" if fc["wd"] == 0 else f"weight decay를 바꿔(wd {fc['wd']:g})") + " 다시 학습하고")
               if fc else "p = 0.99의 LP를 head만 더 오래, weight decay 없이 다시 학습하고")
    fu_run = ""
    if info["fu_dates"] and info["fu_total_min"] is not None:
        d0, d1 = info["fu_dates"]
        fu_run = (f" 후속 실험은 {d0 if d0 == d1 else d0 + '~' + d1}에 따로 돌렸다(GPU 시간: 학습 {info['fu_train_min']:.0f}분, "
                  f"평가까지 {info['fu_total_min']:.0f}분).")
    return f"""
<p class="q">어떻게 다시 돌리나?</p>
<p class="a">PowerShell에서 <code>05_프로젝트\\experiment</code> 폴더로 가서 아래 명령을 차례로 실행하면 되고, 끝난 단계는 건너뛰므로 중간에 멈춰도 같은 명령으로 이어진다.</p>
<pre>..\\..\\.venv\\Scripts\\python.exe driver.py tune
..\\..\\.venv\\Scripts\\python.exe driver.py select
..\\..\\.venv\\Scripts\\python.exe driver.py final
..\\..\\.venv\\Scripts\\python.exe driver.py lplong
..\\..\\.venv\\Scripts\\python.exe ig_cases.py
..\\..\\.venv\\Scripts\\python.exe export_meta.py
..\\..\\.venv\\Scripts\\python.exe figure_inputs.py
python analyze.py
python report.py</pre>
<ol class="prose tight">
<li><code>driver.py tune</code>: 전략마다 학습률 3개 × weight decay 2개를 p = 0.10, seed 100으로 학습한다.</li>
<li><code>driver.py select</code>: 격자 끝 규칙에 따라 필요하면 학습률을 한 칸씩(최대 2번) 넓혀 weight decay 2개로 더 학습한 뒤(GPU),
설정을 골라 <code>results\\tuned.json</code>에 적는다.</li>
<li><code>driver.py final</code>: 본 학습 27회와 평가·IG, 끝난 뒤 IG 정밀도 확인까지 한다.</li>
<li><code>driver.py lplong</code>: 후속 실험. {fu_what} 평가한다(IG는 하지 않는다).</li>
<li><code>ig_cases.py</code>: IG가 어긋난 사례의 히트맵을 다시 계산한다(GPU, final 다음).</li>
<li><code>export_meta.py</code>: 분석용 라벨·난수를 <code>results\\test_meta.npz</code>에 적는다.</li>
<li><code>figure_inputs.py</code>: 그림 0(입력 예시)을 CPU로 그린다.</li>
<li><code>analyze.py</code>: 지표·통계·그림을 만든다(<code>results\\summary.json</code>, <code>per_model.csv</code>, <code>figures\\</code>).</li>
<li><code>report.py</code>: 이 문서(<code>docs\\report.html</code>)를 만든다.</li>
</ol>
<ul class="prose tight">
<li>GPU: {info["gpu"]}. 학습·평가는 Python 3.12 가상환경 <code>설명 및 신뢰 가능한 AI\\.venv</code>에서 돌렸다.
직접 설치한 패키지는 torch {info["torch"]}, torchvision 0.29.1+cu130 두 개다. 나머지는 그 의존성으로 딸려 왔다(목록: <code>requirements-lock.txt</code>).</li>
<li>분석·그림·이 보고서는 전역 Python에 원래 있던 numpy {np.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__},
Pillow {PIL.__version__}(matplotlib의 의존성, 입력 예시·히트맵 그림을 JPEG로 넣을 때)로 만들었다.</li>
<li>데이터: CIFAR-10 binary 판 (md5 <code>c32a1d4ab5d03f1284b67883e8d87530</code>, 공식 값과 같음. md5는 파일 내용으로 계산한 지문 값이라,
같으면 공식 판과 같은 파일이다). 사전학습 가중치: torchvision <code>ResNet18_Weights.IMAGENET1K_V1</code>.</li>
<li>고정 난수: 학습·검증 분할 20261023, 테스트 패치·IG 부분집합 20261031, 본 학습 seed 0·1·2, 튜닝 seed 100.</li>
<li>실행: {info["start"]} 시작, 끝 {info["end"]}. 학습에만 GPU {info["train_hours"]:.1f}시간 (튜닝 포함, 평가·IG 제외).{partial}{fu_run}</li>
<li>코드: <code>common.py</code>(데이터·패치·모델), <code>train.py</code>(학습 1회), <code>evaluate.py</code>(테스트 변형 24개의 logit),
<code>ig.py</code>(IG), <code>driver.py</code>(단계 지휘), <code>ig_cases.py</code>(어긋난 사례), <code>export_meta.py</code>(분석용 메타),
<code>figure_inputs.py</code>(입력 예시 그림), <code>analyze.py</code>(지표·통계·그림), <code>report.py</code>(이 문서). 설계 결정은 <code>README.md</code>.</li>
</ul>
"""


def seed_t(k, scale=1.0):
    """seed별 차이로 낸 95% t 구간 (자유도 n−1). seed가 2개보다 적으면 None."""
    s = np.asarray(R.get(k + ".seeds") or [], dtype=float) * scale
    if len(s) < 2:
        return None
    t = {2: 12.706, 3: 4.303}.get(len(s), 2.776)
    h = t * s.std(ddof=1) / np.sqrt(len(s))
    return s.mean() - h, s.mean() + h


def limits():
    items = [
        "합성 지름길 하나(10색 패치, 위치 하나)만 봤다. 배경·질감처럼 자연스러운 지름길에서는 결과가 다를 수 있다.",
        "사전학습 가중치 하나(ImageNet 지도학습(사람이 붙인 정답으로 학습) ResNet-18)만 썼다. 자기지도 학습(정답 없이 사진 자체로 학습)이나 "
        "더 큰 모델의 backbone은 다른 특징을 가진다.",
        "CIFAR-10의 32×32 사진을 224×224로 키워 흐릿하다. ImageNet 특징이 이런 사진에 덜 맞아서 LP의 정확도 상한이 낮은 것으로 보인다"
        "(따로 확인하지 않았다).",
        "seed가 3개라 seed 표준편차 자체가 거칠다. 괄호 안 bootstrap 구간은 테스트 사진 표집만 반영하므로, 학습을 다시 했을 때의 "
        "흔들림은 seed 줄(같은 seed끼리의 차이)로 따로 봐야 한다.",
        "IG는 회색 기준 이미지를 쓰므로 실제 픽셀이 회색에 가까운 곳은 기여가 0에 가깝게 나온다(기여는 픽셀값 − 기준값에 비례). "
        "히트맵의 흰 윤곽선은 이 때문일 수 있다(해석: 32px 사진을 키울 때 밝은 곳과 어두운 곳 사이가 회색이 된다).",
        "IG 상관 ρ에는 클래스끼리의 차이(패치 색, 물체 종류)도 들어간다. 그래서 클래스 안 ρ를 함께 적었다.",
    ]
    fc = (S.get("followup") or {}).get("config")
    if has("final_train_acc.lp.99") and fc:
        wd = "weight decay 없이" if fc["wd"] == 0 else f"weight decay를 바꿔(wd {fc['wd']:g})"
        items.append(f"학습률·weight decay는 p = 0.10에서만 골랐다. p = {fc['p']:.2f}의 LP는 후속 실험에서 head를 더 오래"
                     f"({fc['epochs']}에폭), {wd} 학습해 봤지만(학습 정확도 {pc('final_train_acc.lp.99')} → "
                     f"{pc('final_train_acc.lplong.99')}, '학습이 충분했나' 절), 학습률은 그대로 두었고 FT·Scratch의 설정도 "
                     f"p = {fc['p']:.2f}에서 다시 고르지 않았다.")
    elif has("final_train_acc.lp.99"):
        items.append("학습률·weight decay는 p = 0.10에서만 골랐다. p = 0.99에서 가장 좋은 값은 다를 수 있다('학습이 충분했나' 절: "
                     f"p = 0.99의 LP는 학습 정확도 {pc('final_train_acc.lp.99')}에서 멈췄고, head를 더 오래 학습해 보지는 않았다).")
    # 클래스별 IG 패치 몫: 보라(9)·초록(1)·주황(8)의 순위 (p = 0.90·0.99의 6조건)
    conds = [(st, q) for st in STRATS for q in ("90", "99") if R.get(f"ig.r_norm_by_class.{st}.{q}")]
    if conds:
        ranks = [np.argsort(np.argsort(R[f"ig.r_norm_by_class.{st}.{q}"])) for st, q in conds]  # 0 = 가장 작음
        n9 = sum(int(r[9] == 0) for r in ranks)
        n1 = sum(int(r[1] <= 2) for r in ranks)
        n8 = sum(int(r[8] <= 2) for r in ranks)
        items.append(f"p = 0.90·0.99의 {len(conds)}조건 중 {n9}조건에서 클래스 9(보라)의 IG 패치 몫이 가장 작다. 보라·주황은 한 채널"
                     "(빨강·초록·파랑 값 중 하나)이 128/255 = 0.502로 회색 기준(0.5)과 0.002만 달라 덜 잡힐 수 있다. 그러나 초록(클래스 1)도 "
                     f"{n1}조건에서 아래 셋 안에 들고 주황(클래스 8)은 {n8}조건에서만 아래 셋 안에 들어서, 이 실험으로 원인을 가르지는 않았다. "
                     "클래스 안 ρ는 같은 색 패치끼리만 비교하므로 이 영향을 받지 않는다.")
    if all(has(f"ig.absdy_seed_rho.{st}.10") for st in STRATS):
        items.append(
            f"LP p = 0.10의 IG 상관이 작은 것(ρ = {num('ig.rho_gray.lp.10')})은 |Δy|가 잡음이라서가 아니다. 같은 사진의 |Δy| 순위가 "
            f"seed를 바꿔도 다시 나온다(seed 사이 ρ = {num('ig.absdy_seed_rho.lp.10')}). FT·Scratch p = 0.10은 |Δy| 순위가 "
            f"seed마다 다르다(seed 사이 ρ: FT {num('ig.absdy_seed_rho.ft.10')}, Scratch {num('ig.absdy_seed_rho.scratch.10')}).")
    kept = R.get("blur_patch_retained")
    if kept is not None:
        items.append(f"흐림 기준 이미지는 x+를 흐린 것이라 패치 자리에 패치 색이 평균 {fx(100 * kept, 0)}% 섞여 남는다. "
                     "그래서 흐림 기준 IG는 패치를 지운 그림과 비교한 값이 아니고, 회색 기준과 재는 양이 다르다.")
    cls = R.get("dprob_by_class.lp.10")
    if has("dprob.lp.10") and cls:
        k = int(np.argmax(cls))
        items.append(f"LP는 p = 0.10에서도 패치 색에 아주 약하게 반응한다(정답 확률 변화 {ppx('dprob.lp.10', 2)}, 가장 큰 것은 "
                     f"{CLASSES[k]} 클래스의 {COLOR_NAMES[k]} 패치 {fx(100 * cls[k], 1, True)}%p). 같은 패치로 학습한 FT는 "
                     f"{ppx('dprob.ft.10', 2)}다. 그래서 p의 효과는 p = 0.10 모델을 뺀 값(p = 0.10 → 0.90·0.99 변화, 차이의 차이)으로 읽는다.")
    # 굵은 글씨 기준과 seed 기준: 본문이 인용하는 비교 가운데 seed 사이 t 구간이 0을 포함하는 것
    cited = [("c.psr.lp-ft.90", "LP − FT PSR (p = 0.90)"), ("c.psr.lp-ft.99", "LP − FT PSR (p = 0.99)"),
             ("c.psr.lp-scratch.90", "LP − Scratch PSR (p = 0.90)"), ("c.psr.lp-scratch.99", "LP − Scratch PSR (p = 0.99)"),
             ("c.follow_excess.lp-ft.90", "LP − FT 패치 따라가기(우연 보정) (p = 0.90)"),
             ("igc.rho_gray.lp-ft.90", "LP − FT IG 상관 (p = 0.90)"), ("igc.rho_gray.lp-ft.99", "LP − FT IG 상관 (p = 0.99)"),
             ("igc.rho_gray.lp-scratch.90", "LP − Scratch IG 상관 (p = 0.90)"),
             ("igc.rho_gray.lp-scratch.99", "LP − Scratch IG 상관 (p = 0.99)"),
             ("igc.rho_gray_within.lp-ft.99", "LP − FT 클래스 안 IG 상관 (p = 0.99)"),
             ("igc.rho_gray_abs.lp-ft.90", "LP − FT 패치 안 |A| 합의 IG 상관 (p = 0.90)"),
             ("igc.rho_gray_abs.lp-ft.99", "LP − FT 패치 안 |A| 합의 IG 상관 (p = 0.99)")]
    if S.get("followup"):  # 후속 실험: '학습이 충분했나' 절과 Main RQ 절이 방향을 말하는 비교
        cited += [("c.psr.lplong-lp.99", "후속 LP − 본 LP PSR (p = 0.99)"),
                  ("c.follow_excess.lplong-lp.99", "후속 LP − 본 LP 패치 따라가기(우연 보정) (p = 0.99)"),
                  ("c.dprob.lplong-lp.99", "후속 LP − 본 LP 정답 확률 변화 (p = 0.99)"),
                  ("c.acc_conf.lplong-lp.99", "후속 LP − 본 LP 충돌 OOD 정확도 (p = 0.99)"),
                  ("c.psr.lplong-ft.99", "후속 LP − FT PSR (p = 0.99)"),
                  ("c.follow_excess.lplong-ft.99", "후속 LP − FT 패치 따라가기(우연 보정) (p = 0.99)"),
                  ("c.psr_none.lplong-ft.99", "후속 LP − FT 패치 제거 전환율 (p = 0.99)")]
    weak =[lab for k, lab in cited if has(k) and seed_t(k) and seed_t(k)[0] <= 0 <= seed_t(k)[1]]
    # 값(차이가 아닌 것)도 하나: Scratch p = 0.99의 IG 상관 (seed별 값으로)
    levels = [("ig.rho_gray.scratch.99", "Scratch의 IG 상관 (p = 0.99)")]
    for k, lab in levels:
        vs = np.asarray(R.get(k + ".values") or [], dtype=float)
        if len(vs) == 3:
            h = 4.303 * vs.std(ddof=1) / np.sqrt(3)
            if vs.mean() - h <= 0 <= vs.mean() + h:
                weak.append(lab)
    items.append("굵은 글씨의 기준 중 'seed 3개가 모두 같은 부호'는 차이가 없어도 4분의 1의 확률로 생긴다. 그래서 본문이 방향을 말하는 "
                 f"비교와 값 {len(cited) + len(levels)}개는 seed 사이 t 구간(seed 3개의 값만으로 낸 95% 구간, 자유도 2)으로도 확인했다. "
                 + (f"그 구간이 0을 포함하는 것: {', '.join(weak)}. 이것들은 학습을 다시 했을 때 방향이 바뀔 수 있다."
                    if weak else "모두 0을 포함하지 않는다."))
    return items


# ---------------------------------------------------------------- 결과 해석 글 (report_text.py)
def read_first():
    """머리말 바로 아래 '읽기 전에' 상자: 머리말·핵심·한눈에 보기가 쓰는 말을 처음 나오기 전에 푼다."""
    notes = S.get("notes") or {}
    fc = (S.get("followup") or {}).get("config")
    wd_main = " · ".join(f"{w:g}" for w in sorted({S["tuned"][st]["wd"] for st in STRATS}))
    wd_fu = "" if not fc else (f", 후속 실험은 {fc['wd']:g}(당기지 않음)" if fc["wd"] == 0 else f", 후속 실험은 {fc['wd']:g}")
    items = [
        ("CIFAR-10 · ResNet-18", "32×32 컬러 사진 10클래스 데이터 · 층 18개를 쌓은 사진 분류용 신경망 구조."),
        ("사전학습", "큰 사진 데이터(ImageNet)로 미리 학습해 둔 가중치(모델 안에서 학습으로 정해지는 값)를 출발점으로 쓰는 것."),
        ("backbone (몸통) · head (머리)", "사진을 특징값 512개로 바꾸는 앞쪽 층 전체 · 특징값을 클래스 점수 10개로 바꾸는 마지막 층."),
        ("LP · FT · Scratch", "LP(linear probing, 선형 탐사) = 사전학습 backbone을 얼리고 head만 학습. FT(full fine-tuning, 전체 미세조정) = "
                              "같은 사전학습 가중치에서 모든 층 학습. Scratch(처음부터 학습) = 무작위 가중치에서 모든 층 학습."),
        ("weight decay (가중치 감쇠)", f"학습 때 가중치를 0 쪽으로 당기는 세기. 본 학습은 {wd_main}{wd_fu}."),
        ("p","학습 사진에 정답 클래스의 색 패치가 붙을 확률(0.10 = 무관, 0.99 = 100장 중 99장)."),
        ("x+ · x−", "같은 테스트 사진에 정답 색 · 틀린 색 패치를 붙인 것."),
        ("PSR (주 지표)", "패치 색만 바꿨을 때(x+ → x−) 답이 바뀐 사진의 비율."),
        ("패치 따라가기(우연 보정)", "x−에서 틀린 색이 가리키는 클래스로 답한 비율 − 패치가 없을 때 그 클래스를 고른 비율."),
        ("패치 제거 전환율", "x+에서 패치를 떼었을 때 답이 바뀐 사진의 비율. 패치 색이 아니라 패치가 있는지만 바꾸는 대조군이다."),
        ("충돌 OOD 정확도", "OOD(out-of-distribution, 학습 분포 밖). 모든 사진에 틀린 색 패치(x−)를 붙였을 때의 정확도."),
        ("차이의 차이", "두 전략의 차이에서 p = 0.10일 때의 차이를 뺀 값(p를 올린 효과만 남긴다)."),
        ("IG · IG 패치 몫(R_patch)", "IG(Integrated Gradients, 적분 기울기) = 정답 점수를 픽셀마다 나눠 주는 설명 방법 · 그 기여 크기 중 패치 안의 몫."),
        ("패치 안 |A| 합", "IG 기여 크기를 패치 안에서만 더한 값. IG 패치 몫은 이것을 사진 전체의 합으로 나눈 비율이다."),
        ("패치 민감도 |Δy|", "패치 색만 바꿨을 때(x+ → x−) 정답 클래스 점수가 바뀐 크기. 사진마다 잰다."),
        ("ρ (Spearman 순위 상관)", "두 값의 순위가 같이 움직이는 정도(−1~1). 클래스 안 ρ = 클래스마다 따로 구한 ρ의 평균."),
        ("seed", "같은 설정을 무작위 값만 바꿔 세 번(0·1·2) 학습할 때의 번호."),
        ("%p · [a, b]", "퍼센트포인트(비율끼리의 차이) · 테스트 사진을 복원 추출로 다시 뽑아 잰 95% 구간(정확도·PSR 같은 지표는 10,000장을 "
                       f"{notes.get('bootstrap_images', 2000):,}번, IG 상관은 IG 1,000장을 {notes.get('bootstrap_ig', 1000):,}번)."),
    ]
    dl = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items)
    return (f'<details class="first"><summary>읽기 전에: 이 보고서가 쓰는 말 {len(items)}개 (눌러서 펴기)</summary><dl class="gl">{dl}</dl>'
            '<p class="tcap">Main RQ(주 연구 질문)·SQ(하위 질문)·H(가설)는 계획서의 질문 번호다. 나머지 용어는 1절 끝 \'용어 풀이\'에 있다.</p>'
            '</details>')


def wait_text(label):
    done = S["n_models"] >= 27 and sum(1 for m in S["per_model"].values() if m.get("ig")) >= 27
    return dict(title=f"{label} ({'해석 글 작성 전' if done else '해석은 결과가 다 나온 뒤에 쓴다'})", q="", a="", intro="", outro="")


def load_texts():
    try:
        import report_text
        return report_text.texts(R, H, S)
    except ModuleNotFoundError:
        return {}


CSS = """
/* 레이아웃: 가운데 읽기 칸(본문 44rem) + 표·그림은 넓은 칸(70rem). 실험 노트처럼 차분한 회녹색 바탕에 강조색 하나(짙은 청록).
   그림판은 두 테마 모두 밝은 종이색이다 (matplotlib 그림 바탕과 같은 색). */
:root{
  --bg:#f3f5f3; --surface:#ffffff; --ink:#1a1d1b; --ink2:#59605b; --rule:#d6dbd7; --rule2:#98a19c;
  --accent:#0d6964; --accent-soft:#dcecea;
  --plate:#fcfcfb; --plate-ink:#3a3e3b; --plate-rule:#dfe1dc;
  --yes:#1b7a3b; --no:#b0281e; --mixed:#8a5b00; --wait:#59605b;
  --lp:#2a78d6; --ft:#eb6834; --sc:#1baf7a;
  --f-body:"Malgun Gothic","맑은 고딕","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;
  --f-mono:"Nanum Gothic Coding","D2Coding",Consolas,"Malgun Gothic",monospace;
  --measure:44rem; --wide:70rem;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#121614; --surface:#1a1f1d; --ink:#e3e7e4; --ink2:#a1a9a4; --rule:#2b3330; --rule2:#56615c;
    --accent:#5cc3b9; --accent-soft:#163330;
    --plate:#fcfcfb; --plate-ink:#3a3e3b; --plate-rule:#3a433f;
    --yes:#63c988; --no:#f2877c; --mixed:#e2b45c; --wait:#a1a9a4;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#121614; --surface:#1a1f1d; --ink:#e3e7e4; --ink2:#a1a9a4; --rule:#2b3330; --rule2:#56615c;
  --accent:#5cc3b9; --accent-soft:#163330;
  --plate:#fcfcfb; --plate-ink:#3a3e3b; --plate-rule:#3a433f;
  --yes:#63c988; --no:#f2877c; --mixed:#e2b45c; --wait:#a1a9a4;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.72;
  word-break:keep-all;overflow-wrap:break-word;-webkit-text-size-adjust:100%}
.page{max-width:var(--wide);margin:0 auto;padding-block:40px 88px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.prose{max-width:var(--measure)}
p{margin:.6rem 0}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
code{font-family:var(--f-mono);font-size:.9em}
b,strong{font-weight:700}
header.top{display:grid;gap:12px;padding-bottom:28px}
.eyebrow{margin:0;color:var(--ink2);font-size:.82rem;letter-spacing:.05em}
h1{margin:0;font-size:clamp(1.55rem,1.05rem + 2vw,2.25rem);line-height:1.32;letter-spacing:-.01em;text-wrap:balance;max-width:28ch}
.sub{margin:0;color:var(--ink2);font-size:.94rem}
.pal{display:flex;flex-wrap:wrap;gap:6px 16px;margin:6px 0 0;padding:0;list-style:none;font-size:.8rem;color:var(--ink2)}
.pal li{display:flex;align-items:center;gap:6px;white-space:nowrap}
.sw{width:14px;height:14px;border:1px solid var(--rule2);flex:none}
.pal .k{font-family:var(--f-mono)}
.core{max-width:var(--measure);background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:14px 18px;margin:8px 0 0}
.core h2{font-size:1rem;margin:0 0 6px;color:var(--ink2);letter-spacing:.03em}
.core ul{margin:0;padding-left:1.2rem}
.core li{margin:.3rem 0}
nav.toc{margin:28px 0 8px;padding-top:20px;border-top:1px solid var(--rule)}
nav.toc p{margin:0 0 6px;font-size:.82rem;color:var(--ink2);letter-spacing:.05em}
nav.toc ol{margin:0;padding-left:1.4rem;columns:2 18rem;column-gap:2.5rem;font-size:.93rem}
nav.toc li{margin:.15rem 0;break-inside:avoid}
section.sec{padding-top:40px;margin-top:40px;border-top:1px solid var(--rule)}
section.sub{margin-top:30px}
h2{font-size:1.38rem;line-height:1.42;margin:0 0 12px;text-wrap:balance;max-width:var(--measure)}
h3{font-size:1.1rem;line-height:1.45;margin:30px 0 8px;text-wrap:balance;max-width:var(--measure)}
.no{font-family:var(--f-mono);color:var(--accent);margin-right:.55em;font-weight:700}
.q{margin:0;color:var(--ink2);max-width:var(--measure)}
.a{margin:.15rem 0 1rem;font-weight:700;font-size:1.05rem;max-width:var(--measure)}
ul.tight li,ol.tight li{margin:.28rem 0}
ol.prose>li{margin:.5rem 0}
.tcap{font-size:.85rem;color:var(--ink2);margin:1.4rem 0 .35rem;max-width:var(--measure);line-height:1.55}
.tw{overflow-x:auto;margin:0 0 1.2rem;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:.42rem .62rem;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{color:var(--ink2);font-weight:600;font-size:.78rem;line-height:1.35;border-bottom:1px solid var(--rule2);
  white-space:normal;min-width:5.2rem;vertical-align:bottom}
th:first-child,td.lab{text-align:left}
tbody tr:last-child td{border-bottom:0}
tr.grp td{border-top:1px solid var(--rule2)}
.pending{color:var(--ink2);font-style:italic}
td.pending{white-space:normal}
td.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
td.num small{display:block;font-size:.72rem;color:var(--ink2);line-height:1.35;font-weight:400}
td.hit{font-weight:700}
.sd{color:var(--ink2);font-size:.76rem;margin-left:.25em}
.dot{display:inline-block;width:.62em;height:.62em;border-radius:50%;margin-right:.45em;vertical-align:.04em}
.dot.lp{background:var(--lp)}.dot.ft{background:var(--ft)}.dot.scratch{background:var(--sc)}
.pv{color:var(--ink2);font-family:var(--f-mono);font-size:.78rem}
table.glance td{white-space:normal;text-align:left;font-size:.88rem;line-height:1.55}
table.glance td:first-child{font-weight:700;min-width:8rem}
table.glance td:nth-child(2){white-space:nowrap}
table.glance td:nth-child(3){min-width:16rem}
figure.plate{margin:1.2rem 0 1.4rem;background:var(--plate);border:1px solid var(--plate-rule);border-radius:8px;padding:12px}
figure.plate img{display:block;width:100%;height:auto}
figure.plate figcaption{color:var(--plate-ink);font-size:.84rem;line-height:1.55;margin-top:8px;max-width:var(--measure)}
.chip{display:inline-flex;align-items:center;gap:.3em;font-size:.76rem;font-weight:700;padding:.06rem .5rem;
  border:1.5px solid currentColor;border-radius:999px;white-space:nowrap;line-height:1.5}
.chip.yes{color:var(--yes)}.chip.no{color:var(--no)}.chip.mixed{color:var(--mixed)}.chip.wait{color:var(--wait)}
dl.gl{display:grid;grid-template-columns:minmax(8rem,15rem) minmax(0,1fr);gap:.45rem 1.2rem;max-width:var(--measure);font-size:.93rem}
dl.gl dt{font-weight:700}
dl.gl dd{margin:0}
pre{font-family:var(--f-mono);font-size:.82rem;background:var(--surface);border:1px solid var(--rule);border-radius:8px;
  padding:12px 14px;overflow-x:auto;line-height:1.6;max-width:var(--wide)}
.box{max-width:var(--measure);background:var(--accent-soft);border-radius:8px;padding:10px 16px;margin:1rem 0}
details{margin:.8rem 0}
summary{cursor:pointer;color:var(--accent);font-weight:600}
details.first{max-width:var(--measure);background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px 18px;margin:8px 0 0}
details.first dl.gl{margin:.6rem 0 0;font-size:.88rem}
details.first .tcap{margin:.6rem 0 0}
@media (max-width:600px){
  body{font-size:15.5px}
  dl.gl{grid-template-columns:minmax(0,1fr)}
  dl.gl dd{margin-bottom:.5rem}
  section.sec{padding-top:30px;margin-top:30px}
}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
"""


def build(artifact=False):
    T, F = build_tables(), build_figures()
    X = load_texts()
    info = run_info()
    n_models, n_ig = S["n_models"], sum(1 for m in S["per_model"].values() if m.get("ig"))

    def sec(sid, label, body, top=True):
        t = X.get(sid) or wait_text(label)
        qa = f'<p class="q">{t["q"]}</p><p class="a">{t["a"]}</p>' if t.get("q") else ""
        return (f'<section id="{sid}" class="{"sec" if top else "sub"}"><{"h2" if top else "h3"}>'
                f'{t["title"]}</{"h2" if top else "h3"}>{qa}<div class="prose">{t.get("intro", "")}</div>'
                f'{body}<div class="prose">{t.get("outro", "")}</div></section>')

    cases = S.get("ig_cases") or {}
    cases_line = (f'<p class="tcap">사례 사진의 IG를 같은 설정(회색 기준, 64단계, TF32)으로 다시 계산했을 때, 저장해 둔 R_patch와의 '
                  f'상대 차이는 모델 {len(cases)}개에서 최대 {fx(100 * max(v["max_rel_diff"] for v in cases.values()), 2)}%였다. '
                  f'(convolution 알고리즘은 고정하지 않고 다시 돌렸다.)</p>' if cases else "")
    E = {  # 절마다 넣을 표·그림 (해석 글 사이)
        "overall": T["acc"] + F["fig1_accuracy"] + T["rel"] + F["fig2_patch_reliance"],
        "rq": T["c_lp_ft"],
        "h1": T["c_lp_scratch"] + T["c_ft_scratch"],
        "h2": T["peff"],
        "sq1": T["did"],
        "pos": T["pos"] + F["fig3_position"],
        "size": T["size"] + F["fig4_size"],
        "ctrl": T["ctrl"],
        "blur": T["blur"] + F["fig5_blur"],
        "rho": T["ig"] + T["igc"] + F["fig8_ig_rho"] + F["fig7_ig_scatter"] + T["igq"],
        "model": T["ml"] + F["fig9_model_level"],
        "maps": F["fig10_heatmaps_lp"] + F["fig10_heatmaps_ft"] + F["fig10_heatmaps_scratch"] + F["fig11_common_heatmaps"],
        "cases": cases_line + F["fig12_cases_p0.99"] + F["fig12_cases_p0.90"],
        "train": T["train"] + F["fig6_curves"] + T["tune"] + (T["lplong"] + F["fig13_lplong"] if S.get("followup") else ""),
        "precision": T["prec"],
    }
    top = [  # (id, 차례 이름, 본문)
        ("design", "실험 설계와 용어", None),
        ("overall", "전체 결과 표", E["overall"]),
        ("rq", "Main RQ: LP vs FT", E["rq"]),
        ("h1", "H1: LP vs Scratch", E["h1"]),
        ("h2", "H2: p = 0.99에서 LP", E["h2"]),
        ("sq1", "SQ1: 상관 강도에 따른 차이", E["sq1"]),
        ("stress", "조건 바꾸기 시험(stress test) 4가지", "".join(sec(s, lbl, E[s], top=False) for s, lbl in
                                             (("pos", "위치"), ("size", "크기"), ("ctrl", "패치 제거·회색 가림"), ("blur", "물체 흐리기")))),
        ("ig", "SQ2·H3: IG와 패치 민감도", "".join(sec(s, lbl, E[s], top=False) for s, lbl in
                                             (("rho", "모델 안 상관"), ("model", "모델 단위 비교"), ("maps", "히트맵 예시"),
                                              ("cases", "IG가 어긋난 사례")))),
        ("train", "학습이 충분했나", E["train"]),
        ("precision", "IG 정밀도 확인", E["precision"]),
        ("changes", "계획서와 달라진 점", None),
        ("limits", "한계", None),
        ("repro", "재현 방법", None),
    ]
    fixed = {
        "design": ("실험은 사진 구석의 색 패치를 정답과 짝지어 지름길을 만든다", design_section(F)),
        "changes": (f"계획서와 달라진 것은 튜닝 규칙·튜닝 예산·실행 순서·IG 정밀도이고, 계획서에 없던 분석 {len(ADDED)}가지를 더했다",
                    changes_section()),
        "limits": ("결론은 이 합성 패치와 사전학습 가중치 하나에 한정된다",
                   '<p class="q">이 결과를 어디까지 믿을 수 있나?</p><p class="a">통제된 합성 지름길 하나, 사전학습 가중치 하나, '
                   'seed 3개에서 얻은 결과라, 자연스러운 지름길이나 다른 사전학습으로 넓히려면 따로 확인해야 한다.</p>'
                   f'<ul class="prose tight">{"".join(f"<li>{x}</li>" for x in limits())}{X.get("limits_extra", "")}</ul>'),
        "repro": ("같은 명령 아홉 줄로 처음부터 다시 만들 수 있다", repro_section()),
    }
    parts = []
    for i, (sid, label, body) in enumerate(top, start=1):
        if sid in fixed:
            title, inner = fixed[sid]
            parts.append(f'<section id="{sid}" class="sec"><h2><span class="no">{i}</span>{title}</h2>{inner}</section>')
        else:
            s = sec(sid, label, body)
            parts.append(s.replace("<h2>", f'<h2><span class="no">{i}</span>', 1))
    toc = "".join(f'<li><a href="#{sid}">{label}</a></li>' for sid, label, _ in top)
    pal = "".join(f'<li><span class="sw" style="background:rgb({r},{g},{b})"></span><span class="k">{k}</span> {CLASSES[k]} · '
                  f'{COLOR_NAMES[k]}</li>' for k, (r, g, b) in enumerate(META["palette"].tolist()))
    glance = X.get("glance")
    if glance:
        rows = "".join(f"<tr><td>{q}</td><td>{chip(kind, lbl)}</td><td>{a}</td><td>{ev}</td></tr>"
                       for q, kind, lbl, a, ev in glance)
        glance_html = (f'<div class="tw"><table class="glance"><thead><tr><th>질문</th><th>판정</th><th>답</th><th>근거</th></tr>'
                       f'</thead><tbody>{rows}</tbody></table></div>')
    elif n_models >= 27 and n_ig >= 27:
        glance_html = (f'<p class="prose">{chip("wait", "해석 글 작성 전")} 모델 27개의 평가와 IG가 모두 끝났다. '
                       '결과 해석 글(report_text.py)은 아직 없다.</p>')
    else:
        glance_html = f'<p class="prose">{chip("wait", "진행 중")} 모델 {n_models}/27개 평가, {n_ig}/27개 IG 완료. 해석은 27개가 다 끝난 뒤에 쓴다.</p>'
    core = X.get("core", "")
    core_html = f'<div class="core"><h2>핵심</h2><ul>{core}</ul></div>' if core else ""
    title = "LP·FT 패치 지름길 실험"
    head = (f'<title>{title}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Nanum+Gothic+Coding:wght@400;700&display=swap">\n'
            f"<style>{CSS}</style>\n")
    body = f"""<div class="page">
<header class="top">
<p class="eyebrow">설명 및 신뢰 가능한 AI · 팀 프로젝트 실험 결과 · {info["end"][:10] if info["end"][0].isdigit() else "진행 중"}</p>
<h1>{X.get("headline") or "사전학습 backbone을 얼리면 색 패치 지름길에 덜 기대는가"}</h1>
<p class="sub">CIFAR-10 · ResNet-18 · LP / FT / Scratch × 상관 강도 p 0.10·0.90·0.99 × seed 3개 = 모델 27개{"" if n_models >= 27 else f" (지금 {n_models}개 완료)"}</p>
<ul class="pal" aria-label="클래스별 패치 색">{pal}</ul>
{read_first()}
{core_html}
</header>
<section id="glance"><h2 style="font-size:1.1rem;margin:22px 0 8px">한눈에 보기</h2>{glance_html}</section>
<nav class="toc" aria-label="차례"><p>차례</p><ol>{toc}</ol></nav>
{"".join(parts)}
</div>"""
    full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</head>\n<body>\n{body}\n</body>\n</html>\n")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "report.html").write_text(full, encoding="utf-8")
    if artifact:
        (DOCS / "report_artifact.html").write_text(head + body + "\n", encoding="utf-8")
    size = len(full.encode("utf-8"))
    print(f"report.html {size / 1e6:.1f} MB | models {n_models} | texts {'yes' if X else 'no'}")
    if size > 16e6:
        print("경고: 16 MB를 넘는다 (Artifact 게시 한도)")


if __name__ == "__main__":
    build(artifact="--artifact" in sys.argv)
