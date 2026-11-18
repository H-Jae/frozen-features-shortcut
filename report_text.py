# 결과 해석 글. report.py가 texts(R, H, S)를 불러 제목·절 제목·질문·굵은 답·본문을 넣는다.
# 숫자는 모두 R(summary.json을 편 값)과 S에서 H의 서식 함수로 넣는다. 손으로 적은 숫자는 없다.
# 방향을 말하는 문장('낮다', '같은 방향' 같은 말)은 need()로 데이터와 맞는지 확인한다. 어긋나면 보고서를 만들지 않고 멈춘다.
# (시험용: 환경 변수 REPORT_TEXT_WARN=1이면 멈추지 않고 어긋난 문장만 출력한다.)
# 숫자 바로 뒤에는 조사를 붙이지 않는다(읽는 소리에 따라 조사가 달라져서). %·%p·배 뒤에는 붙여도 된다.
# '이상'에는 내림, '이하'에는 올림한 경계를 쓴다(반올림하면 경계가 틀릴 수 있어서). '모두'는 모델 하나하나의 값으로 확인한다.
import math
import os

import numpy as np

CLASS = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]
COLOR = ["빨강", "초록", "파랑", "노랑", "자홍", "청록", "흰색", "검정", "주황", "보라"]
STS = ("lp", "ft", "scratch")


def texts(R, H, S):
    pc, ppx, pp, num, fx, excl0 = H.pc, H.ppx, H.pp, H.num, H.fx, H.excl0

    def need(cond, what):
        if cond:
            return
        if os.environ.get("REPORT_TEXT_WARN"):
            print("데이터와 맞지 않는 문장:", what)
            return
        raise AssertionError(f"report_text.py의 문장이 데이터와 맞지 않는다: {what}")

    def v(k):
        return R[k]

    def vals(k):
        return R[k + ".values"]

    def floor_fx(x, d):
        return fx(math.floor(x * 10 ** d + 1e-9) / 10 ** d, d)

    def ceil_fx(x, d):
        return fx(math.ceil(x * 10 ** d - 1e-9) / 10 ** d, d)

    def sd_line(k):
        """seed별 차이: 'seed별 −16.1 · −19.0 · −19.0%p'"""
        return "seed별 " + " · ".join(fx(100 * x, 1, True) for x in R[k + ".seeds"]) + "%p"

    def same_sign(k, sign):
        s = R.get(k + ".seeds") or []
        return len(s) == 3 and all(x * sign > 0 for x in s)

    def ppx_r(k):
        """ρ의 차이: '+0.22 [+0.16, +0.28]'"""
        return f"{fx(R[k], 2, True)} [{fx(R[k + '.lo'], 2, True)}, {fx(R[k + '.hi'], 2, True)}]"

    def gap(k):
        """'낮다·높다·떨어진다' 앞에 쓰는 차이의 크기 (부호 없이): '18.1%p'"""
        return f"{fx(abs(100 * R[k]), 1)}%p"

    def pcp(k):
        """%p 단위 값 (차이 꼴 지표의 평균): '40.0%p'"""
        return f"{fx(100 * R[k], 1)}%p"

    def rank_rho(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    T = {}

    # ------------------------------------------------------------ 제목 (Main RQ의 답)
    need(excl0("c.psr.lp-ft.99") == -1 and same_sign("c.psr.lp-ft.99", -1), "p=0.99 LP−FT PSR < 0 (구간, seed 3개)")
    need(excl0("c.psr.lp-ft.90") == 1 and excl0("c.psr_none.lp-ft.99") == 1, "p=0.90 PSR·p=0.99 패치 제거는 LP가 더 흔들림")
    T["headline"] = ("사전학습 backbone을 얼린 LP는 p = 0.99에서 FT보다 틀린 색 패치에 훨씬 덜 흔들리지만, "
                     "p = 0.90에서와 패치를 떼는 대조군에서는 그렇지 않다")

    def josa(word, a, b):
        """받침이 있으면 a(은·이), 없으면 b(는·가)."""
        c = ord(word[-1]) - 0xAC00
        return word + (a if 0 <= c < 11172 and c % 28 else b)

    # ------------------------------------------------------------ 전체
    pmax10 = max(max(vals(f"psr.{st}.10")) for st in STS)
    for st in STS:
        need(excl0(f"pe.psr.{st}.99") == 1, f"{st}: p=0.99에서 PSR이 p=0.10보다 크다")
    need(v("psr.scratch.99") > v("psr.ft.99") > v("psr.lp.99"), "p=0.99 PSR: Scratch > FT > LP")
    for st in STS:
        need(v(f"acc_id.{st}.99") > v(f"acc_id.{st}.10") and v(f"acc_conf.{st}.99") < v(f"acc_conf.{st}.10"),
             f"{st}: ID 정확도는 오르고 충돌 OOD 정확도는 떨어진다")
    T["overall"] = dict(
        title="p를 올리면 세 전략 모두 패치에 기대지만, 기대는 정도는 전략마다 크게 다르다",
        q="학습 때 패치와 정답의 상관 p를 올리면 무엇이 바뀌나?",
        a=(f"p = 0.99에서 패치 색만 바꿨을 때 답이 바뀐 사진의 비율(PSR)은 Scratch {pc('psr.scratch.99')}, "
           f"FT {pc('psr.ft.99')}, LP {pc('psr.lp.99')}로, p = 0.10의 {ceil_fx(100 * pmax10, 1)}% 이하에서 크게 올랐다."),
        intro=(
            "<p>첫 표는 테스트 정확도다. 학습과 같은 p로 패치를 붙인 ID 테스트에서는 p가 클수록 정확도가 오른다. "
            "정답 색 패치가 붙는 사진이 늘고, 모델이 그 패치를 쓰기 때문이다. 모든 사진에 틀린 색을 붙인 충돌 OOD에서는 "
            f"반대로 떨어진다. FT는 p = 0.10 → 0.99에서 ID {pc('acc_id.ft.10')} → {pc('acc_id.ft.99')}, "
            f"충돌 OOD {pc('acc_conf.ft.10')} → {pc('acc_conf.ft.99')}다.</p>"),
        outro=(
            "<p>둘째 표의 PSR만으로는 패치 의존과 가림 약함(패치 색과 상관없이 사진 일부가 가려지거나 바뀌기만 해도 답이 바뀌는 약함)을 "
            f"가를 수 없다. LP는 패치가 정답과 무관한 p = 0.10에서도 PSR이 {pc('psr.lp.10')}로 FT({pc('psr.ft.10')})보다 높고, "
            f"같은 넓이의 회색 사각형을 사진 가운데에 덧대도 {pc('psr_occ_center.lp.10')}의 답이 바뀐다(FT "
            f"{pc('psr_occ_center.ft.10')}). 그래서 전략끼리 비교할 때는 PSR과 함께 패치 따라가기(우연 보정)와 차이의 차이"
            "(뜻은 맨 위 '읽기 전에')를 본다. 패치 따라가기(우연 보정)는 답이 패치 색이 가리키는 클래스로 바뀐 경우만 세므로, "
            "가림 약함 때문에 다른 클래스로 바뀐 답은 들어가지 않는다. 차이의 차이는 p = 0.10에서 이미 있던 전략 차이"
            f"(LP − FT PSR {ppx('c.psr.lp-ft.10')}, Main RQ 절: 그중 {fx(100 * (1 - v('c.psr_to_w.lp-ft.10') / v('c.psr.lp-ft.10')), 0)}%"
            "는 패치 색이 아닌 다른 클래스로 바뀐 것)를 빼므로, p를 올려 생긴 차이만 남긴다.</p>"),
    )

    # ------------------------------------------------------------ Main RQ: LP vs FT
    for k in ("psr", "follow", "follow_excess", "dprob"):
        need(excl0(f"c.{k}.lp-ft.99") == -1, f"p=0.99 LP−FT {k} < 0")
    need(excl0("c.acc_conf.lp-ft.99") == 1, "p=0.99 LP−FT 충돌 OOD 정확도 > 0")
    need(excl0("c.psr_none.lp-ft.99") == 1 and same_sign("c.psr_none.lp-ft.99", 1), "p=0.99 패치 제거 전환율: LP > FT")
    need(excl0("did.acc_none.lp-ft.99") == 0, "p=0.99 패치 없음 정확도의 p 효과: LP·FT 차이 없음")
    need(excl0("c.psr_occ_center.lp-ft.99") == 1 and excl0("c.acc_id.lp-ft.99") == -1, "p=0.99 가림 약함·ID 정확도는 LP가 나쁨")
    need(excl0("c.psr.lp-ft.90") == 1 and excl0("c.acc_conf.lp-ft.90") == -1, "p=0.90 주 지표: LP가 더 흔들림")
    need(excl0("c.follow_excess.lp-ft.90") == -1 and same_sign("c.follow_excess.lp-ft.90", -1), "p=0.90 따라가기(우연 보정): LP < FT")
    need(excl0("c.psr_to_w.lp-ft.90") == 0 and max(abs(v("c.psr_to_w.lp-ft.90.lo")), abs(v("c.psr_to_w.lp-ft.90.hi"))) < 0.01
         and excl0("c.psr_other.lp-ft.90") == 1, "p=0.90: 패치 색 클래스로 바뀐 비율 차이는 1%p 안, 다른 클래스로는 LP가 큼")
    need(excl0("c.psr_none.lp-ft.90") == 1, "p=0.90 패치 제거 전환율: LP > FT")
    need(v("c.psr.lp-ft.10") > 0 and v("c.psr_occ_tl.lp-ft.10") > 0 and v("c.psr_occ_center.lp-ft.10") > 0,
         "p=0.10: PSR·가림 전환율 모두 LP가 큼")
    need(excl0("c.follow_excess.lp-ft.10") == 0 and excl0("c.dprob.lp-ft.10") == 1, "p=0.10: 따라가기 차이 없음, 확률 변화는 LP가 큼")
    other10 = 1 - v("c.psr_to_w.lp-ft.10") / v("c.psr.lp-ft.10")
    need(other10 > 0.8, "p=0.10: LP−FT PSR 차이의 80% 넘게가 패치 색이 아닌 클래스로 바뀐 것")
    need(v("psr_tl.ft.99") < 0.1 * v("psr_br.ft.99"), "FT p=0.99: 왼쪽 위 PSR이 학습 자리의 10% 미만")
    # 후속 실험(p = 0.99 LP의 head를 더 오래, weight decay 없이): Main RQ 해석·'학습이 충분했나' 절·핵심이 같이 쓴다
    fc = (S.get("followup") or {}).get("config")
    need(fc is not None and fc["wd"] == 0, "후속 실험(lplong) 결과가 있고 weight decay가 0이다")
    tu_lp = S["tuned"]["lp"]
    need(abs(fc["p"] - 0.99) < 1e-9 and fc["lr"] == tu_lp["lr"] and fc["epochs"] > tu_lp["epochs"],
         "후속 실험: p = 0.99, 본 LP와 같은 학습률, 더 많은 에폭")
    times = fc["epochs"] / tu_lp["epochs"]  # 본 LP보다 몇 배 오래
    # FT와의 PSR 차이 중 후속에서도 남은 몫 = (후속 LP − FT) / (본 LP − FT): bootstrap 구간과 seed별 값으로 확인한다
    kk = "psr_gap_kept.lplong.99"
    ks = R[kk + ".seeds"]
    kept_t_lo = float(np.mean(ks) - 4.303 * np.std(ks, ddof=1) / np.sqrt(len(ks)))  # seed 사이 t 구간(자유도 2)의 아래 끝
    need(excl0("c.psr.lplong-lp.99") == 1 and same_sign("c.psr.lplong-lp.99", 1), "후속 LP − 본 LP PSR > 0")
    need(excl0("c.psr.lplong-ft.99") == -1 and same_sign("c.psr.lplong-ft.99", -1), "후속 LP − FT PSR < 0")
    need(len(ks) == 3 and v(kk + ".lo") > 0.75 and kept_t_lo > 0.75 and v(kk + ".hi") < 1,
         "FT와의 PSR 차이: 4분의 3 넘게 남고(bootstrap·seed t 구간의 아래 끝), 줄기는 줄었다(구간 위 끝 < 1)")
    T["rq"] = dict(
        title="p = 0.99에서 LP는 FT보다 틀린 색 패치에 훨씬 덜 흔들리지만, 패치를 떼었을 때는 더 흔들린다",
        q="Main RQ: 같은 사전학습 가중치에서 backbone을 얼리면(LP) 모든 층을 학습할 때(FT)보다 패치에 덜 기대나?",
        a=(f"주 지표 PSR로 보면 p = 0.99에서만 그렇다(LP {pc('psr.lp.99')} 대 FT {pc('psr.ft.99')}; "
           f"p = 0.90에서는 LP {pc('psr.lp.90')} 대 FT {pc('psr.ft.90')})."),
        intro=(
            "<p>표의 값은 LP − FT다. 패치 색을 바꾸는 지표(PSR, 패치 따라가기, 정답 확률 변화)는 음수면 LP가 덜 기댄 것이다. "
            "패치 제거 전환율은 패치가 있는지만 바꾸는 대조군이라, 양수면 패치가 없어질 때 LP가 더 흔들린 것이다. 회색 가림 전환율은 "
            "패치와 무관한 가림 약함을 보는 대조군이라, 양수면 LP가 더 약한 것이다. 정확도는 양수면 LP가 나은 것이다.</p>"
            "<ul class=\"tight\">"
            f"<li><b>p = 0.99: 패치 색을 바꾸는 지표는 모두 LP가 덜 기댔다고 말한다.</b> PSR {pp('c.psr.lp-ft.99')}, "
            f"{sd_line('c.psr.lp-ft.99')}. 패치 따라가기(우연 보정)는 {gap('c.follow_excess.lp-ft.99')}, 정답 확률 변화는 "
            f"{gap('c.dprob.lp-ft.99')} 작고, 틀린 색 패치를 붙여도 LP가 {gap('c.acc_conf.lp-ft.99')} 더 많이 맞힌다. "
            f"<b>패치를 떼는 대조군은 반대다.</b> 패치를 떼면 답이 바뀐 사진이 LP {pc('psr_none.lp.99')}, FT "
            f"{pc('psr_none.ft.99')}로 LP가 {gap('c.psr_none.lp-ft.99')} 많고, p = 0.10에서 이미 있던 차이"
            f"({ppx('c.psr_none.lp-ft.10')})를 빼도 LP가 {gap('did.psr_none.lp-ft.99')} 많다({pp('did.psr_none.lp-ft.99')}). "
            "p = 0.10 모델보다 줄어든 패치 없음 정확도는 "
            f"LP {gap('pe.acc_none.lp.99')}, FT {gap('pe.acc_none.ft.99')}로 비슷하다(차이의 차이 {pp('did.acc_none.lp-ft.99')}). "
            f"회색 가림 전환율({ppx('c.psr_occ_center.lp-ft.99')})과 ID 정확도({ppx('c.acc_id.lp-ft.99')})도 LP 쪽이 나쁘다.</li>"
            f"<li><b>p = 0.90: 주 지표로는 LP가 더 흔들린다.</b> PSR {pp('c.psr.lp-ft.90')}, 충돌 OOD 정확도 "
            f"{ppx('c.acc_conf.lp-ft.90')}. 패치 따라가기(우연 보정)만 LP가 {gap('c.follow_excess.lp-ft.90')} 덜 따라간다"
            "(seed 3개 모두 같은 방향).</li>"
            f"<li><b>p = 0.10: 패치가 정답과 무관한데도 LP의 PSR이 {gap('c.psr.lp-ft.10')} 높다.</b> 이 차이 중 패치 색의 "
            f"클래스로 바뀐 것은 {gap('c.psr_to_w.lp-ft.10')}이고, 나머지 {fx(100 * other10, 0)}%는 다른 클래스로 바뀐 것이다. "
            f"패치 따라가기(우연 보정)의 차이는 구간이 0을 포함한다({pp('c.follow_excess.lp-ft.10')}). 같은 넓이 회색 사각형도 "
            f"LP를 더 흔든다(왼쪽 위 {ppx('c.psr_occ_tl.lp-ft.10')}, 가운데 {ppx('c.psr_occ_center.lp-ft.10')}). 그래서 이 "
            f"차이는 주로 가림 약함이다. 다만 LP는 p = 0.10에서도 패치 색에 약하게 반응한다(정답 확률 변화 "
            f"{ppx('c.dprob.lp-ft.10')}).</li></ul>"),
        outro=(
            "<p><b>p = 0.90에서 LP의 PSR이 높은 이유(해석).</b> 바뀐 답이 패치 색의 클래스로 간 비율은 LP와 FT의 차이가 "
            f"1%p 안이다({pp('c.psr_to_w.lp-ft.90')}). 차이는 다른 클래스로 바뀐 사진에서 난다({pp('c.psr_other.lp-ft.90')}; "
            f"p = 0.10의 차이를 빼도 {ppx('did.psr_other.lp-ft.90')}). LP는 맞는 색 패치를 정답의 보조 증거로 쓰다가, 패치가 바뀌면 "
            "그 증거를 잃고 물체만으로는 정답을 지키지 못한다는 해석과 맞는다. 패치를 뗐을 때도 답이 바뀐 사진이 LP가 "
            f"{gap('c.psr_none.lp-ft.90')} 많다(p = 0.10의 차이를 빼도 {ppx('did.psr_none.lp-ft.90')}).</p>"
            "<p><b>p = 0.99에서 FT가 더 흔들리는 이유(해석, 따로 확인하지 않았다).</b> FT는 backbone까지 바꿀 수 있으므로 학습 "
            f"자리의 패치 색을 직접 읽는 특징을 만들 수 있다. 패치를 왼쪽 위로 옮기면 FT의 PSR이 {pc('psr_br.ft.99')} → "
            f"{pc('psr_tl.ft.99')}로 주는 것이 이 해석과 맞는다. LP는 얼린 특징 512개에 가중치를 곱해 더하는 계산(선형 결합)만 "
            f"바꿀 수 있다. head를 {times:g}배 오래, weight decay 없이 학습한 후속 실험에서는 LP의 PSR이 "
            f"{gap('c.psr.lplong-lp.99')} 올랐지만, FT와의 PSR 차이는 {gap('c.psr.lp-ft.99')}에서 {gap('c.psr.lplong-ft.99')}로 "
            "줄었을 뿐이다('학습이 충분했나' 절, 학습률은 바꾸지 않았다).</p>"),
    )
    need(excl0("did.psr_none.lp-ft.99") == 1 and excl0("did.psr_none.lp-ft.90") == 1 and excl0("did.psr_other.lp-ft.90") == 1,
         "패치 제거·다른 클래스 전환: p=0.10 차이를 빼도 LP가 많음")

    # ------------------------------------------------------------ H1: LP vs Scratch
    for q in ("90", "99"):
        need(excl0(f"c.psr.lp-scratch.{q}") == -1 and same_sign(f"c.psr.lp-scratch.{q}", -1), f"p=0.{q} LP−Scratch PSR < 0")
        need(excl0(f"c.follow_excess.lp-scratch.{q}") == -1 and same_sign(f"c.follow_excess.lp-scratch.{q}", -1),
             f"p=0.{q} LP−Scratch 따라가기 < 0")
    need(abs(v("c.psr.lp-scratch.99")) > 5 * abs(v("c.psr.lp-scratch.90")), "LP−Scratch PSR: 0.99가 0.90의 5배 넘게 큼")
    need(excl0("c.acc_conf.lp-scratch.90") == 0 and excl0("c.acc_conf.lp-scratch.10") == -1
         and excl0("did.acc_conf.lp-scratch.90") == 1 and same_sign("did.acc_conf.lp-scratch.90", 1),
         "충돌 OOD: p=0.90 차이 구간이 0 포함, p=0.10에는 LP가 낮음, 차이의 차이는 LP 쪽")
    need(excl0("c.acc_conf.lp-scratch.99") == 1, "p=0.99 LP−Scratch 충돌 OOD 정확도 > 0")
    need(v("c.psr.lp-ft.90") > 0 > v("c.psr.ft-scratch.90"), "p=0.90: LP−FT > 0, FT−Scratch < 0")
    need(v("c.psr.lp-ft.99") < 0 and v("c.psr.ft-scratch.99") < v("c.psr.lp-ft.99"), "p=0.99: 두 몫 모두 음수, FT−Scratch가 더 큼")
    T["h1"] = dict(
        title="LP는 p = 0.90·0.99 모두 Scratch보다 패치 색 교체에 덜 흔들리고, 차이는 p = 0.99에서 훨씬 크다",
        q="H1: 상관이 중간 이상(p = 0.90, 0.99)일 때 LP가 Scratch보다 패치 교체에 덜 흔들리나?",
        a=(f"그렇다: LP − Scratch의 PSR은 p = 0.90에서 {ppx('c.psr.lp-scratch.90')}, p = 0.99에서 "
           f"{ppx('c.psr.lp-scratch.99')}이고, 두 p 모두 seed 3개가 같은 방향이다."),
        intro=(
            "<p>이 비교는 시작 가중치와 학습하는 층이 둘 다 달라(1절 전략 표), 차이를 얼린 효과로만 읽을 수 없다. "
            "아래 끝에서 두 부분으로 나눠 본다.</p>"
            "<ul class=\"tight\">"
            f"<li><b>p = 0.99</b>: Scratch는 패치 색만 바꿔도 {pc('psr.scratch.99')}의 답이 바뀌고, 틀린 색 패치를 붙이면 "
            f"{pc('acc_conf.scratch.99')}만 맞히며, 패치를 떼면 정확도가 {pc('acc_none.scratch.99')}로 떨어진다. 물체가 아니라 "
            f"패치로 답을 정하는 모델이 됐다. LP − Scratch의 충돌 OOD 정확도 차이는 {ppx('c.acc_conf.lp-scratch.99')}다.</li>"
            f"<li><b>p = 0.90</b>: PSR 차이는 {gap('c.psr.lp-scratch.90')}로 작고, 패치 따라가기(우연 보정)는 LP가 "
            f"{gap('c.follow_excess.lp-scratch.90')} 덜 따라간다. 충돌 OOD 정확도는 두 전략이 비슷하지만"
            f"({pp('c.acc_conf.lp-scratch.90')}), LP는 p = 0.10에서 이미 {gap('c.acc_conf.lp-scratch.10')} 낮았다. p를 올려 잃은 "
            f"충돌 OOD 정확도는 LP {gap('pe.acc_conf.lp.90')}, Scratch {gap('pe.acc_conf.scratch.90')}로, LP가 "
            f"{gap('did.acc_conf.lp-scratch.90')} 적게 잃었다(차이의 차이).</li></ul>"),
        outro=(
            "<p>둘째 표(FT − Scratch)와 함께 보면 LP − Scratch를 두 부분으로 나눌 수 있다. LP − Scratch = (LP − FT) + (FT − Scratch)이다. "
            "앞에는 backbone을 얼린 것과 전략마다 고른 학습 설정의 차이가, 뒤에는 사전학습 가중치와 학습 설정의 차이가 함께 들어 있다. "
            "두 부분을 따로 실험으로 가르지는 않았다.</p><ul class=\"tight\">"
            f"<li>p = 0.90의 PSR: LP − FT {ppx('c.psr.lp-ft.90')}, FT − Scratch {ppx('c.psr.ft-scratch.90')}. 두 부분이 반대 방향이라, "
            "LP는 FT보다는 더 흔들리고 Scratch보다는 덜 흔들린다.</li>"
            f"<li>p = 0.99의 PSR: LP − FT {ppx('c.psr.lp-ft.99')}, FT − Scratch {ppx('c.psr.ft-scratch.99')}. 두 부분이 같은 방향이고, "
            "FT − Scratch 부분이 더 크다.</li></ul>"),
    )

    # ------------------------------------------------------------ H2: p=0.99에서 LP
    need(excl0("pe.follow_excess.lp.99") == 1 and excl0("pe.acc_conf.lp.99") == -1, "LP p=0.99: 따라가기 증가, 충돌 정확도 감소")
    need(v("pe.follow_excess.lp.99") < v("pe.follow_excess.ft.99"), "p=0.99 따라가기 증가: LP < FT")
    need(excl0("pe.acc_none.lp.99") == -1 and excl0("pe.acc_none.ft.99") == -1, "p=0.99: 패치 없음 정확도 감소 (LP, FT)")
    ratio = v("pe.follow_excess.lp.99") / v("pe.follow_excess.ft.99")
    need(excl0("did.acc_none.lp-ft.99") == 0, "p=0.99 패치 없음 정확도의 p 효과: LP·FT 차이 없음")
    T["h2"] = dict(
        title="LP도 p = 0.99에서 패치를 쓰고, 틀린 색 패치는 FT보다 훨씬 덜 따라가지만 패치를 떼는 대조군에서는 FT보다 덜 기대지 않는다",
        q="H2: 패치가 100장 중 99장에서 맞는 p = 0.99에서는 얼린 backbone(LP)도 패치를 쓰나?",
        a=(f"쓴다: p = 0.10 모델보다 패치 따라가기(우연 보정)가 {gap('pe.follow_excess.lp.99')} 늘고 충돌 OOD 정확도가 "
           f"{gap('pe.acc_conf.lp.99')} 떨어지는데, FT는 같은 값이 {gap('pe.follow_excess.ft.99')} 늘고 "
           f"{gap('pe.acc_conf.ft.99')} 떨어진다."),
        intro=(
            "<p>얼린 ImageNet 특징값 가운데에도 패치 색에 반응하는 값이 있다. head(마지막 층)만 학습해도 그 값에 큰 가중치를 주면 "
            "패치를 쓸 수 있다. 계획서가 말한 '선형으로 접근할 수 있는 지름길'이 이것이다. 표는 같은 전략 안에서 p를 올렸을 때의 "
            "변화이고, 같은 seed끼리 짝지었다. H2는 지지된다.</p>"),
        outro=(
            f"<p>LP의 패치 따라가기(우연 보정) 증가는 FT의 {fx(100 * ratio, 0)}% 크기다. 패치 없는 사진의 정확도는 LP와 FT가 "
            f"p = 0.10 모델보다 비슷하게 낮아졌다(Main RQ 절: 차이의 차이 {pp('did.acc_none.lp-ft.99')}). 해석: head가 물체 특징에 "
            "주던 가중치 일부를 패치 쪽으로 옮겼을 수 있다(따로 확인하지 않았다).</p>"),
    )

    # ------------------------------------------------------------ SQ1
    need(excl0("did.psr.lp-ft.90") == 1 and excl0("did.psr.lp-ft.99") == -1, "LP−FT 차이의 차이: 0.90 양수, 0.99 음수")
    need(excl0("did.psr.lp-scratch.90") == -1 and excl0("did.psr.lp-scratch.99") == -1, "LP−Scratch 차이의 차이: 둘 다 음수")
    need(excl0("did.follow_excess.lp-ft.90") == -1, "LP−FT 따라가기 차이의 차이 0.90: 음수")
    need(v("psr.ft.99") - v("psr.ft.90") > v("psr.lp.99") - v("psr.lp.90"), "0.90→0.99 PSR 증가: FT > LP")
    need(v("pe.psr.lp.90") > v("pe.psr.ft.90"), "0.10→0.90 PSR 증가: LP > FT")
    T["sq1"] = dict(
        title="FT와 비교한 LP의 PSR 이점은 p = 0.99에서만 나타나고, Scratch와 비교한 이점은 p = 0.90부터 있다",
        q="SQ1: 상관 강도 p가 커지면 전략 사이의 차이가 어떻게 바뀌나?",
        a=(f"p를 0.90까지 올리면 LP의 PSR이 FT보다 더 오르지만(차이의 차이 {ppx('did.psr.lp-ft.90')}), 0.99까지 올리면 FT가 훨씬 "
           f"더 올라 뒤집히고({ppx('did.psr.lp-ft.99')}), Scratch보다는 두 p 모두 덜 오른다(0.90 {ppx('did.psr.lp-scratch.90')}, "
           f"0.99 {ppx('did.psr.lp-scratch.99')})."),
        intro=(
            "<p>FT와의 차이가 고르게 커지지 않는 것은 p = 0.10 → 0.90에서는 LP의 PSR이 더 오르고"
            f"({ppx('pe.psr.lp.90')} 대 {ppx('pe.psr.ft.90')}), 0.90 → 0.99에서는 FT가 더 오르기"
            f"({fx(100 * (v('psr.ft.99') - v('psr.ft.90')), 1)}%p 대 {fx(100 * (v('psr.lp.99') - v('psr.lp.90')), 1)}%p) 때문이다.</p>"),
        outro=(
            "<p>패치 따라가기(우연 보정)로 보면 p = 0.90에서도 LP − FT 차이의 차이가 LP 쪽이다"
            f"({pp('did.follow_excess.lp-ft.90')}). 그래서 FT와 비교한 p = 0.90의 판정은 지표에 따라 갈린다"
            "(Main RQ 절: 주 지표 PSR로는 LP가 더 흔들린다).</p>"),
    )

    # ------------------------------------------------------------ 조건 바꾸기 시험
    for q in ("90", "99"):
        for st in STS:
            need(v(f"follow_excess_b16.{st}.{q}") > v(f"follow_excess_b0.{st}.{q}"), f"{st} p=0.{q}: 흐릴수록 따라가기 증가")
    for q in ("10", "90", "99"):
        need(v(f"psr_occ_center.lp.{q}") > max(v(f"psr_occ_center.ft.{q}"), v(f"psr_occ_center.scratch.{q}")),
             f"p=0.{q}: 회색 가림 전환율은 LP가 가장 큼")
    T["stress"] = dict(
        title="자리·크기를 바꾸면 FT·Scratch의 패치 의존은 줄고 LP의 의존은 남으며, 물체를 흐리면 모두 패치를 더 따라간다",
        q="학습 때와 다른 조건(자리, 크기, 가림, 흐림)에서도 패치 의존이 남나?",
        a=("자리를 옮기면 FT·Scratch의 패치 의존은 크게 줄고 LP의 의존은 남으며, 크기를 바꾸면 p = 0.90·0.99에서 FT·Scratch는 "
           "학습 크기에서 가장 크게, LP는 패치가 클수록 더 반응하고, 회색 가림에는 LP가 가장 많이 흔들리며, 물체를 흐리면 "
           "p = 0.90·0.99의 세 전략 모두 패치를 더 따라간다."),
        intro="", outro="")

    def corner_ratio(st):
        r = [v(f"psr_{c}.{st}.99") / v(f"psr_br.{st}.99") for c in ("tl", "tr", "bl")]
        return math.floor(100 * min(r)), math.ceil(100 * max(r))

    rl, rf, rs = corner_ratio("lp"), corner_ratio("ft"), corner_ratio("scratch")
    need(rl[0] > max(rf[1], rs[1]), "다른 구석 PSR 비율: LP의 최소가 FT·Scratch의 최대보다 큼")
    c4 = [v(f"psr_{c}.lp.10") for c in ("br", "tl", "tr", "bl")]
    fe_other = max(v(f"follow_excess_{c}.{st}.99") for c in ("tl", "tr", "bl") for st in STS)
    need(fe_other < 0.1, "다른 구석의 패치 따라가기(우연 보정) < 10%p")
    T["pos"] = dict(
        title="LP는 패치를 다른 구석으로 옮겨도 반응하고, FT·Scratch는 학습 자리에서 가장 크게 반응한다",
        q="패치를 학습 때와 다른 구석에 붙이면 어떻게 되나?",
        a=(f"p = 0.99에서 다른 세 구석의 PSR은 학습 자리 PSR의 LP {rl[0]}~{rl[1]}%, FT {rf[0]}~{rf[1]}%, "
           f"Scratch {rs[0]}~{rs[1]}%다."),
        intro="",
        outro=(
            f"<p>LP의 PSR은 p = 0.10에서도 네 구석 모두 {floor_fx(100 * min(c4), 1)}~{ceil_fx(100 * max(c4), 1)}%다(가림 약함). "
            f"그래서 p = 0.10과 비교하면 왼쪽 위 PSR은 LP {pc('psr_tl.lp.10')} → {pc('psr_tl.lp.99')}, "
            f"FT {pc('psr_tl.ft.10')} → {pc('psr_tl.ft.99')}다. 다른 구석에서 바뀐 답은 패치 색의 클래스로 간 것이 적다: "
            f"패치 따라가기(우연 보정)는 세 전략 모두 {ceil_fx(100 * fe_other, 0)}%p 이하다.</p>"
            "<p>해석(따로 확인하지 않았다): 세 전략 모두 마지막 층 앞에서 위치마다의 특징값을 사진 전체로 평균 내므로(global average "
            "pooling), 마지막 층은 패치가 어느 구석에 있었는지 알 수 없다. 그래서 자리에 따라 다르게 반응하려면 앞쪽 층의 특징이 자리에 "
            "따라 달라야 한다. LP는 앞쪽 층을 바꿀 수 없어 이런 특징을 만들지 못하고, FT와 Scratch는 앞쪽 층까지 바꿀 수 있다.</p>"),
    )
    for q in ("90", "99"):
        for st in ("ft", "scratch"):
            need(v(f"psr_s24.{st}.{q}") > max(v(f"psr_s12.{st}.{q}"), v(f"psr_s48.{st}.{q}")), f"{st} p=0.{q} PSR: 24px가 최대")
        need(v(f"psr_s48.lp.{q}") > v(f"psr_s24.lp.{q}") > v(f"psr_s12.lp.{q}"), f"LP p=0.{q} PSR: 클수록 큼")
        for st in STS:
            need(v(f"ig.r_norm_s24_mean.{st}.{q}") > max(v(f"ig.r_norm_s12_mean.{st}.{q}"), v(f"ig.r_norm_s48_mean.{st}.{q}")),
                 f"{st} p=0.{q}: IG 패치 몫/넓이 24px가 최대")

    def size_ratio(st, s):
        return f"{fx(100 * v(f'psr_s{s}.{st}.99') / v(f'psr_s24.{st}.99'), 0)}%"

    T["size"] = dict(
        title="p = 0.90·0.99에서 FT·Scratch는 seed 평균으로 학습 크기(24px)의 패치에 가장 크게 반응하고, LP는 패치가 클수록 더 반응한다",
        q="패치 크기를 바꾸면 어떻게 되나?",
        a=(f"p = 0.99에서 FT·Scratch의 PSR은 학습 크기 24px에서 가장 크고(한 변 12 → 24 → 48px: FT {pc('psr_s12.ft.99')} → "
           f"{pc('psr_s24.ft.99')} → {pc('psr_s48.ft.99')}, Scratch {pc('psr_s12.scratch.99')} → {pc('psr_s24.scratch.99')} → "
           f"{pc('psr_s48.scratch.99')}), LP는 패치가 클수록 크다({pc('psr_s12.lp.99')} → {pc('psr_s24.lp.99')} → "
           f"{pc('psr_s48.lp.99')})."),
        intro="", outro=(
            f"<p>p = 0.99에서 48px 패치에는 학습 크기 반응의 FT {size_ratio('ft', 48)}, Scratch {size_ratio('scratch', 48)}가 "
            f"남고, 12px에서는 {size_ratio('ft', 12)}·{size_ratio('scratch', 12)}로 준다. 해석(따로 확인하지 않았다): FT·Scratch의 "
            "PSR이 학습 크기에서 가장 큰 것은 학습 때의 패치 크기에 맞춘 특징을 만들었다는 쪽과 맞는다.</p>"
            "<p>p = 0.90·0.99에서 IG 패치 몫/넓이는 세 전략 모두 학습 크기에서 가장 크다. 그러나 LP의 PSR은 48px에서 가장 커서"
            f"(p = 0.99 {pc('psr_s48.lp.99')}), LP에서는 IG와 행동이 가리키는 크기가 다르다. 그래서 이 IG 결과는 위 해석의 근거가 "
            "되지 못한다.</p>"),
    )
    need(v("psr_occ_center.lp.10") > max(v("psr_occ_center.ft.10"), v("psr_occ_center.scratch.10"))
         and v("psr_occ_center.lp.99") < v("psr_occ_center.lp.10"), "가림: LP가 가장 많이 흔들림, p가 오르면 줄어듦")
    T["ctrl"] = dict(
        title="LP는 패치와 상관없는 회색 사각형에도 더 흔들린다(가림 약함)",
        q="패치를 떼거나, 패치는 두고 같은 넓이의 회색 사각형을 덧대면 어떻게 되나?",
        a=(f"p = 0.10에서 가운데에 회색 사각형을 덧대면 LP {pc('psr_occ_center.lp.10')}, Scratch {pc('psr_occ_center.scratch.10')}, "
           f"FT {pc('psr_occ_center.ft.10')}의 답이 바뀌어 LP가 가장 많이 흔들리고, 패치를 떼면 p = 0.99에서 LP {pc('psr_none.lp.99')}, "
           f"FT {pc('psr_none.ft.99')}, Scratch {pc('psr_none.scratch.99')}의 답이 바뀐다."),
        intro="", outro=(
            f"<p>p가 오르면 회색 가림 전환율은 줄어든다(LP {pc('psr_occ_center.lp.10')} → {pc('psr_occ_center.lp.99')}). "
            "맞는 색 패치가 답을 붙잡아 주는 것으로 보인다(해석, 따로 확인하지 않았다).</p>"),
    )
    acc16 = [x for st in STS for q in ("10", "90", "99") for x in vals(f"acc_none_b16.{st}.{q}")]
    need(v("follow_excess_b16.lp.99") < 0.5 < min(v("follow_excess_b16.ft.99"), v("follow_excess_b16.scratch.99")),
         "σ=16 따라가기(우연 보정): LP < 50%p < FT, Scratch")
    for q in ("90", "99"):
        for st in STS:
            need(all(a > b for a, b in zip(vals(f"follow_excess_b16.{st}.{q}"), vals(f"follow_excess_b0.{st}.{q}"))),
                 f"{st} p=0.{q}: 모델 하나하나에서 σ=16 따라가기 > 흐림 없음")
    T["blur"] = dict(
        title="p = 0.90·0.99 모델은 물체를 흐리면(σ = 0 → 16) 패치를 더 따라가지만, LP는 끝까지 절반 아래다",
        q="물체 정보를 줄이면(물체만 흐리게) 패치 의존이 커지나?",
        a=(f"커진다: p = 0.99에서 물체를 σ = 16으로 흐리면 패치 따라가기(우연 보정)가 FT {pcp('follow_excess_b16.ft.99')}, "
           f"Scratch {pcp('follow_excess_b16.scratch.99')}, LP {pcp('follow_excess_b16.lp.99')}가 된다."),
        intro="", outro=(
            f"<p>σ = 16에서는 패치 없는 사진의 정확도가 모델에 따라 {floor_fx(100 * min(acc16), 1)}~{ceil_fx(100 * max(acc16), 1)}%로 "
            "떨어진다(10클래스를 무작위로 고르면 10%). 물체를 알아볼 수 없을 만큼 흐려도 LP는 패치 색을 절반 아래로만 따라간다. "
            f"H2 절의 결과(LP의 패치 따라가기 증가가 FT의 {fx(100 * ratio, 0)}%)와 같은 방향이다.</p>"),
    )

    # ------------------------------------------------------------ IG
    for q in ("90", "99"):
        for st in STS:
            need(v(f"ig.rho_gray.{st}.{q}.lo") > 0, f"{st} p=0.{q}: 계획서의 ρ가 양수 (구간)")
        need(v(f"ig.rho_gray.lp.{q}") > max(v(f"ig.rho_gray.ft.{q}"), v(f"ig.rho_gray.scratch.{q}")), f"p=0.{q}: LP의 ρ가 가장 큼")
        need(excl0(f"igc.rho_gray_within.lp-ft.{q}") == 1 and same_sign(f"igc.rho_gray_within.lp-ft.{q}", 1),
             f"p=0.{q}: 클래스 안 ρ LP > FT")
        need(v(f"ig.rho_blur.scratch.{q}") < 0.1, f"Scratch p=0.{q}: 흐림 기준 ρ < 0.1")
    within_max = max(v(f"ig.rho_gray_within.{st}.{q}") for st in ("ft", "scratch") for q in ("90", "99"))
    within_lp_min = min(v(f"ig.rho_gray_within.lp.{q}") for q in ("90", "99"))
    abs_within_min = min(v(f"ig.rho_gray_abs_within.{st}.{q}") for st in STS for q in ("90", "99"))
    need(within_lp_min > 0.3 and within_max < 0.1 and abs_within_min > 0.2,
         "클래스 안 ρ: LP > 0.3, FT·Scratch < 0.1 (비율), 절대량은 모두 > 0.2")
    need(v("ig.rho_gray.lp.10") < min(v("ig.rho_gray.ft.10"), v("ig.rho_gray.scratch.10")) and v("ig.rho_gray.lp.10.lo") > 0,
         "p=0.10: LP의 ρ가 가장 작지만 0보다 큼")
    for q in ("90", "99"):
        cis = S["groups"][f"ft|0.{q}"]["ig"]["rho_blur_ci_per_seed"]
        need(all(lo > 0 for lo, _ in cis), f"FT p=0.{q}: 흐림 기준 ρ의 seed별 구간이 모두 0보다 큼")
    sc_cis = S["groups"]["scratch|0.99"]["ig"]["rho_gray_ci_per_seed"]
    zero_seeds = [s for s, (lo, hi) in enumerate(sc_cis) if lo <= 0 <= hi]
    need(len(zero_seeds) == 1, "Scratch p=0.99: seed 하나의 ρ 구간만 0을 포함")
    zs = zero_seeds[0]
    zrho = S["per_model"][f"scratch|0.99|{zs}"]["ig"]["rho_gray"]
    T["ig"] = dict(
        title="p = 0.90·0.99에서 IG 패치 몫은 세 전략 모두 패치 민감도와 같이 움직이고, 같은 클래스 안에서 비율로 재면 LP의 관계만 뚜렷하게 남는다",
        q="SQ2·H3: IG가 패치에 준 몫이 큰 사진일수록 실제로 패치 교체에 민감한가?",
        a=("p = 0.90·0.99에서 IG 패치 몫은 세 전략 모두 패치 민감도와 같이 움직이지만, 같은 클래스 안에서 비율로 재면 LP는 그 관계가 "
           f"남고(seed 평균 {num('ig.rho_gray_within.lp.90')}·{num('ig.rho_gray_within.lp.99')}) FT·Scratch는 seed 평균 "
           f"{ceil_fx(within_max, 2)} 이하로 약해진다."),
        intro=(
            "<ul class=\"tight\">"
            "<li><b>계획서의 통계</b>(1,000장 전체에서 구한 ρ(R_patch, |Δy|))는 p = 0.90·0.99에서 셋 다 양수이고 LP가 가장 크다"
            f"(LP {num('ig.rho_gray.lp.90')}·{num('ig.rho_gray.lp.99')}, FT {num('ig.rho_gray.ft.90')}·{num('ig.rho_gray.ft.99')}, "
            f"Scratch {num('ig.rho_gray.scratch.90')}·{num('ig.rho_gray.scratch.99')}). 다만 Scratch p = 0.99에서는 seed 하나의 구간이 "
            f"0을 포함한다(ρ = {fx(zrho, 2)} [{fx(sc_cis[zs][0], 2)}, {fx(sc_cis[zs][1], 2)}]).</li>"
            "<li><b>같은 클래스 사진끼리만 보면</b>(계획서에 없던 분석) LP는 남고, FT·Scratch는 seed 평균으로 "
            f"{ceil_fx(within_max, 2)} 이하로 준다('모델 안 상관' 절).</li>"
            "<li><b>패치 안 |A| 합으로 재면</b>(H3의 '패치 안 기여의 크기'에 더 가까운 측정) 세 전략 모두 클래스 안 ρ가 seed 평균으로 "
            f"{floor_fx(abs_within_min, 2)} 이상이다.</li>"
            f"<li><b>흐림 기준</b>으로는 Scratch의 ρ만 작아서(p = 0.90·0.99에서 {num('ig.rho_blur.scratch.90')}·"
            f"{num('ig.rho_blur.scratch.99')}), Scratch의 결론은 기준 이미지에도 달려 있다. FT는 흐림 기준에서도 seed별 구간이 모두 "
            "0보다 크다.</li>"
            f"<li><b>p = 0.10</b>에서는 LP가 가장 약하게만 따라간다(ρ: LP {num('ig.rho_gray.lp.10')}, FT {num('ig.rho_gray.ft.10')}, "
            f"Scratch {num('ig.rho_gray.scratch.10')}).</li></ul>"
            "<p>그래서 H3의 판정은 LP에서는 지지이고, FT에서는 측정 방식에 따라, Scratch에서는 측정 방식과 기준 이미지에 따라 갈린다.</p>"),
        outro="")
    cls_rho = {st: rank_rho(R[f"ig.r_norm_by_class.{st}.99"], R[f"ig.absdy_med_by_class.{st}.99"]) for st in STS}
    for st in ("ft", "scratch"):
        need(v(f"ig.rho_gray_within.{st}.99") < 0.5 * v(f"ig.rho_gray.{st}.99"), f"{st} p=0.99: 클래스 안 ρ < 전체 ρ의 절반")
        need(v(f"ig.rho_gray_abs_within.{st}.99") > v(f"ig.rho_gray_within.{st}.99"), f"{st} p=0.99: 절대량 클래스 안 ρ > 비율")
    need(v("ig.rho_gray_within.lp.99") > 0.6 * v("ig.rho_gray.lp.99"), "LP p=0.99: 클래스 안 ρ가 전체 ρ의 60% 넘게 남음")
    need(excl0("igc.rho_gray.lp-ft.99") == 1 and excl0("igc.rho_gray_abs.lp-ft.99") == 1
         and same_sign("igc.rho_gray_abs.lp-ft.99", 1), "p=0.99 LP−FT: 두 측정 모두 양수")
    need(all(x > 0 for x in cls_rho.values()), "클래스 평균끼리의 순위 상관: 세 전략 모두 양수")
    for st in ("ft", "scratch"):
        need(v(f"ig.rho_gray_within.{st}.90") < 0.5 * v(f"ig.rho_gray.{st}.90"), f"{st} p=0.90: 클래스 안 ρ < 전체 ρ의 절반")
    T["rho"] = dict(
        title="p = 0.90·0.99에서 클래스끼리의 차이로 생긴 상관을 빼면 FT·Scratch의 상관은 절반 아래로 줄고 LP의 상관은 남는다",
        q="한 모델 안에서 IG 패치 몫의 순위와 |Δy|의 순위는 얼마나 같이 움직이나?",
        a=(f"p = 0.99에서 1,000장 전체의 ρ는 LP가 가장 크고(LP {num('ig.rho_gray.lp.99')}, FT {num('ig.rho_gray.ft.99')}, "
           f"Scratch {num('ig.rho_gray.scratch.99')}), 같은 클래스 사진끼리만 보면 FT·Scratch는 절반 아래로 줄지만"
           f"(FT {num('ig.rho_gray_within.ft.99')}, Scratch {num('ig.rho_gray_within.scratch.99')}) LP는 남는다"
           f"(LP {num('ig.rho_gray_within.lp.99')})."),
        intro=(
            "<p>ρ(Spearman 순위 상관)는 테스트 1,000장에서 두 순위가 얼마나 같은지다(−1~1). 하나는 IG 패치 몫(R_patch)의 순위, "
            "다른 하나는 패치 민감도 |Δy|(패치 색을 바꿨을 때 정답 점수가 바뀐 크기)의 순위다.</p>"
            "<ul class=\"tight\"><li><b>클래스 안 ρ</b>: 1,000장은 클래스마다 100장이고 클래스마다 패치 색이 달라, IG 패치 몫과 "
            "|Δy|가 둘 다 클래스마다 다르다. 두 값의 클래스 평균이 같은 순서로 움직이면, 같은 클래스 안의 관계와 상관없이 1,000장 "
            f"전체의 ρ가 생긴다(p = 0.99, 클래스 10개 평균끼리의 순위 상관: LP {fx(cls_rho['lp'], 2)}, FT {fx(cls_rho['ft'], 2)}, "
            f"Scratch {fx(cls_rho['scratch'], 2)}). 클래스 안 ρ는 클래스 10개 각각에서 구한 ρ의 평균이라, 클래스끼리의 차이에서 "
            "생긴 ρ를 뺀다.</li>"
            "<li><b>패치 안 |A| 합</b>: R_patch는 패치 안 기여를 사진 전체의 기여로 나눈 비율이다. 나누지 않은 패치 안 기여 크기로 "
            "같은 ρ를 구해 측정 방식을 바꿔 본다.</li></ul>"),
        outro=(
            "<p>측정 방식을 바꾸면 FT·Scratch의 결론이 달라진다. 패치 안 |A| 합으로 재면 클래스 안 ρ가 비율로 잴 때보다 크다"
            f"(p = 0.99에서 FT {num('ig.rho_gray_abs_within.ft.99')} 대 {num('ig.rho_gray_within.ft.99')}, "
            f"Scratch {num('ig.rho_gray_abs_within.scratch.99')} 대 {num('ig.rho_gray_within.scratch.99')}). "
            "비율은 패치 안 기여를 사진 전체의 기여로 나누므로, 물체 쪽 기여가 큰 사진은 패치 기여가 같아도 몫이 작아진다. "
            "1,000장 전체의 ρ로 보면 LP − FT 차이는 두 측정 모두에서 양수지만 크기가 다르다"
            f"(p = 0.99, 비율 {ppx_r('igc.rho_gray.lp-ft.99')}, 패치 안 |A| 합 {ppx_r('igc.rho_gray_abs.lp-ft.99')}).</p>"
            "<p>LP p = 0.10의 ρ가 작은 것이 |Δy|의 잡음 때문이 아니라는 확인(seed 사이 ρ = "
            f"{num('ig.absdy_seed_rho.lp.10')})은 '한계' 절에 적었다. 왜 작은지는 확인하지 않았다.</p>"),
    )
    ml90 = R["ml.p90.rnorm_psr"]
    lpft90 = vals("ig.r_norm_mean.lp.90") + vals("ig.r_norm_mean.ft.90")
    need((S.get("model_level") or {}).get("cond") is not None and R["ml.cond.rnorm_psr"] > 0.8, "모델 단위: 조건 평균 ρ > 0.8")
    need(ml90 < -0.5, "모델 단위: p=0.90 안 ρ < −0.5")
    need(v("psr.scratch.90") > max(v("psr.lp.90"), v("psr.ft.90")), "p=0.90 PSR: Scratch가 가장 큼")
    need(max(vals("ig.r_norm_mean.scratch.90")) < min(lpft90), "p=0.90 몫/넓이: Scratch seed 모두가 LP·FT의 어느 seed보다 작음")
    need(v("follow_excess.ft.90") > v("follow_excess.lp.90") and v("ig.r_norm_mean.ft.90") > v("ig.r_norm_mean.lp.90"),
         "p=0.90: 따라가기(우연 보정)와 IG 몫 모두 FT > LP")
    ftv = vals("ig.r_norm_mean.ft.90")
    T["model"] = dict(
        title="모델 사이에서 IG 패치 몫은 p의 차이는 따라가지만, p = 0.90에서는 패치에 가장 많이 기댄 Scratch를 가장 낮게 매긴다",
        q="평균 IG 패치 몫이 큰 모델일수록 패치에 더 기대나?",
        a=(f"조건(전략 × p) {R['ml.cond.n']}개의 평균끼리는 같이 움직이지만(ρ = {fx(R['ml.cond.rnorm_psr'], 2)}), "
           f"p = 0.90 안에서는 반대로 움직인다(모델 {R['ml.p90.n']}개, ρ = {fx(ml90, 2)})."),
        intro=(
            f"<p>p = 0.90에서 PSR이 가장 큰 Scratch({pc('psr.scratch.90')})는 IG 패치 몫/넓이가 가장 작다"
            f"({num('ig.r_norm_mean.scratch.90')}배, seed 3개 모두 LP·FT의 어느 seed보다 작다). FT"
            f"({num('ig.r_norm_mean.ft.90')}배)와 LP({num('ig.r_norm_mean.lp.90')}배)의 차이는 FT의 seed 사이 폭"
            f"({fx(min(ftv), 2)}~{fx(max(ftv), 2)}배) 안이다. 패치 따라가기(우연 보정)로 재면 FT가 LP보다 더 기대서"
            f"({pcp('follow_excess.ft.90')} 대 {pcp('follow_excess.lp.90')}) IG의 순서와 맞는다.</p>"),
        outro=(
            "<p>조건 사이의 ρ가 큰 것은 p를 올리면 PSR과 IG 패치 몫이 함께 오르기 때문이다. 그래서 IG 패치 몫만 보고 같은 p의 "
            "모델 중 어느 쪽이 패치에 더 기대는지 고르면 틀릴 수 있다. p = 0.90의 Scratch가 그 예다.</p>"),
    )
    out_min = 1 - max(max(vals(f"ig.r_mean.{st}.{q}")) for st in STS for q in ("10", "90", "99"))
    need(v("ig.r_norm_mean.scratch.99") > v("ig.r_norm_mean.ft.99") > v("ig.r_norm_mean.lp.99"), "p=0.99 몫/넓이: Scratch > FT > LP")
    T["maps"] = dict(
        title="히트맵에서 패치 칸이 뚜렷하게 진한 것은 p = 0.99의 Scratch와 FT다",
        q="IG 히트맵으로 보면 모델들은 어디에 기여를 주나?",
        a=(f"p = 0.99에서 IG 패치 몫/넓이는 Scratch {num('ig.r_norm_mean.scratch.99', 1)}배, "
           f"FT {num('ig.r_norm_mean.ft.99', 1)}배, LP {num('ig.r_norm_mean.lp.99', 1)}배라, 히트맵에서 패치 칸이 뚜렷하게 진한 것은 "
           "Scratch와 FT다."),
        intro=f"<p>모든 모델에서 IG 기여의 {floor_fx(100 * out_min, 0)}% 이상은 패치 밖에 있다(모델마다 잰 값).</p>",
        outro="")

    cases = S.get("ig_cases") or {}
    sc = cases.get("scratch|0.99|0")
    need(sc is not None and len(cases) == 6, "사례 6개 모델")
    miss = [c for k, c in zip(sc["kind"], sc["classes"]) if k == "miss"]
    false = [c for k, c in zip(sc["kind"], sc["classes"]) if k == "false"]
    need(len(set(miss)) == 1 and len(set(false)) == 1 and miss[0] != false[0], "Scratch p=0.99: 놓침·헛짚음이 각각 한 클래스")
    ncls = {key: len(set(c["classes"])) for key, c in cases.items()}
    need(all(n >= 4 for key, n in ncls.items() if key != "scratch|0.99|0"), "다른 모델은 6장이 클래스 4개 이상")
    r0 = S["per_model"]["scratch|0.99|0"]["ig"]["r_norm_by_class"]
    order = list(np.argsort(r0))  # 작은 것부터
    need(order[-1] == false[0] and order.index(miss[0]) <= 2, "Scratch p=0.99 seed 0: 헛짚음 클래스의 몫이 가장 크고, 놓침 클래스는 아래 셋 안")
    names = {"lp": "LP", "ft": "FT", "scratch": "Scratch"}
    spread_txt = ", ".join(f"{names[k.split('|')[0]]} p = {k.split('|')[1]} {n}개"
                           for k, n in sorted(ncls.items(), key=lambda kv: (STS.index(kv[0].split('|')[0]), kv[0])))
    T["cases"] = dict(
        title="p = 0.99의 Scratch에서는 IG가 놓친 사진과 헛짚은 사진이 각각 한 클래스(패치 색)로 몰린다",
        q="IG 패치 몫과 패치 민감도가 크게 어긋나는 사진에는 공통점이 있나?",
        a=(f"p = 0.99의 Scratch에서 IG가 놓친 3장은 모두 {CLASS[miss[0]]}({COLOR[miss[0]]} 패치)이고, "
           f"헛짚은 3장은 모두 {CLASS[false[0]]}({COLOR[false[0]]} 패치)다."),
        intro=(
            "<p>고르는 규칙은 결과를 보기 전에 정했다(ig_cases.py). seed 0 모델마다 IG 1,000장에서 순위를 백분위로 바꾼 뒤 두 가지를 "
            "3장씩 골랐다.</p><ul class=\"tight\">"
            "<li><b>IG가 놓친 사진</b>: |Δy| 순위 상위 25% 안에서 'R_patch 순위가 |Δy| 순위보다 가장 많이 낮은' 사진. "
            "패치 교체에 크게 반응하는데 IG는 패치에 몫을 적게 줬다.</li>"
            "<li><b>IG가 헛짚은 사진</b>: R_patch 순위 상위 25% 안에서 반대로 고른 사진. IG는 패치에 몫을 많이 줬는데 "
            "패치를 바꿔도 반응이 작다.</li></ul>"
            f"<p>다른 모델은 덜 몰린다. 모델마다 고른 6장이 속한 클래스 수: {spread_txt}.</p>"),
        outro=(
            "<p>사례는 seed 0 모델의 1,000장 전체 순위로 고르므로, IG 패치 몫이 큰 클래스에서 헛짚은 사례가, 작은 클래스에서 놓친 "
            f"사례가 나오기 쉽다. Scratch p = 0.99(seed 0)에서 IG 패치 몫/넓이는 {josa(CLASS[false[0]], '이', '가')} "
            f"{fx(r0[false[0]], 1)}배로 클래스 중 가장 크고, {josa(CLASS[miss[0]], '은', '는')} {fx(r0[miss[0]], 1)}배로 아래에서 "
            f"{order.index(miss[0]) + 1}번째다. "
            "클래스 안 ρ는 클래스마다 따로 구한 값의 평균이라, 이런 클래스끼리의 차이로는 작아지지 않는다.</p>"),
    )

    # ------------------------------------------------------------ 학습
    ft_sc_min = min(x for st in ("ft", "scratch") for q in ("10", "90", "99") for x in vals(f"final_train_acc.{st}.{q}"))
    ep_total = (S.get("tuned") or {}).get("lp", {}).get("epochs")
    lp_eps = [int(x) for q in ("10", "90", "99") for x in vals(f"best_epoch.lp.{q}")]
    grid = (S.get("tuned") or {}).get("lp", {}).get("grid") or []
    best = {}
    for r in grid:
        best[r["lr"]] = max(best.get(r["lr"], 0), r["best_val_acc"])
    lrs = sorted(best)
    need(ft_sc_min >= 0.995, "FT·Scratch 모델 하나하나의 마지막 학습 정확도 ≥ 99.5%")
    need(ep_total is not None and max(lp_eps) < ep_total, "LP의 고른 에폭이 모두 마지막 에폭 전")
    need(len(lrs) >= 2 and 0 < best[lrs[-1]] - best[lrs[-2]] < 0.005, "LP 학습률 맨 위 두 칸 차이 < 0.5%p")
    need(v("final_train_acc.lp.99") < 0.97, "LP p=0.99 학습 정확도 < 97%")
    need(v("acc_id.lp.99") > v("acc_id.lp.10"), "LP ID 정확도는 p를 올리면 오른다")
    drops = [x for q in ("10", "90", "99") for x in vals(f"val_drop_last.lp.{q}")]
    # 후속 실험(fc·tu_lp·times·kk·ks는 Main RQ 절 위에서 정했다)
    need(v("final_train_acc.lp.99") < v("final_train_acc.lplong.99") < fc["p"], "후속 LP 학습 정확도: 본 LP보다 높고 p보다 낮다")
    need(v("train_acc_gain_last20.lplong.99") > 0, "후속 LP 학습 정확도가 마지막 20에폭에도 올랐다")
    for k, sign in (("psr", 1), ("follow_excess", 1), ("dprob", 1), ("acc_conf", -1)):
        need(excl0(f"c.{k}.lplong-lp.99") == sign and same_sign(f"c.{k}.lplong-lp.99", sign), f"후속 LP − 본 LP {k}: 패치를 더 쓴 쪽")
    need(excl0("c.follow_excess.lplong-ft.99") == -1 and same_sign("c.follow_excess.lplong-ft.99", -1), "후속 LP − FT 따라가기 < 0")
    need(excl0("c.psr_none.lplong-ft.99") == 1 and same_sign("c.psr_none.lplong-ft.99", 1), "후속 LP − FT 패치 제거 전환율 > 0")
    T["train"] = dict(
        title=f"FT·Scratch는 충분히 학습됐고, LP는 head를 {times:g}배 오래 학습해도 FT보다 틀린 색 패치에 훨씬 덜 흔들린다",
        q="덜 학습된 모델을 '지름길에 강하다'고 잘못 읽은 것은 아닌가?",
        a=(f"FT·Scratch는 아니고(18개 모델 모두 마지막 에폭 학습 정확도 {floor_fx(100 * ft_sc_min, 1)}% 이상), p = 0.99의 LP는 "
           f"head를 {times:g}배 오래, weight decay 없이 학습해도 FT와의 PSR 차이가 {gap('c.psr.lp-ft.99')}에서 "
           f"{gap('c.psr.lplong-ft.99')}로 줄었을 뿐이다."),
        intro=(
            "<p>p = 0.99의 학습 데이터는 패치만 봐도 99%를 맞힐 수 있다. 그런데 본 실험의 LP는 학습 정확도 "
            f"{pc('final_train_acc.lp.99')}에서 멈췄다. 본 실험만으로는 LP의 학습이 모자랐는지 아래 두 가지 때문에 가를 수 없었다.</p>"
            "<ul class=\"tight\">"
            f"<li>LP가 고른 에폭은 {ep_total}에폭 중 {min(lp_eps)}~{max(lp_eps)}번째지만, 마지막 에폭과의 검증 정확도 차이가 "
            f"{fx(100 * min(drops), 2)}~{fx(100 * max(drops), 2)}%p라, 학습률이 0으로 줄며 생긴 평탄 구간과 구별되지 않는다.</li>"
            f"<li>튜닝의 학습률 비교({lrs[-2]:g} → {lrs[-1]:g}에서 {fx(100 * (best[lrs[-1]] - best[lrs[-2]]), 2, True)}%p)는 "
            "p = 0.10에서 한 것이라, p = 0.99의 head가 패치를 덜 쓴 까닭은 가르지 못한다.</li></ul>"
            "<p>그래서 결과를 본 뒤 후속 실험을 정했다. p = 0.99의 LP를 본 실험과 같은 seed"
            f"({'·'.join(str(s) for s in S['followup']['seeds'])}, 같은 학습 패치 배정)와 같은 학습률({fc['lr']:g})로 다시 학습하면서 "
            f"에폭({tu_lp['epochs']} → {fc['epochs']})과 weight decay({tu_lp['wd']:g} → {fc['wd']:g})만 바꿨다. "
            "평가는 본 실험과 같고, IG는 하지 않았다. 아래 마지막 표와 그림이 그 결과다.</p>"),
        outro=(
            "<p>LP가 패치를 쓸 만큼은 학습됐다는 것은 ID 정확도로 확인된다. p를 올리면 LP의 ID 정확도가 "
            f"{pc('acc_id.lp.10')} → {pc('acc_id.lp.99')}로 오른다.</p>"
            f"<p><b>후속 LP는 더 학습됐지만 {100 * fc['p']:g}%에 닿지 않았다.</b> 마지막 에폭 학습 정확도는 "
            f"{pc('final_train_acc.lp.99')} → {pc('final_train_acc.lplong.99')}로 "
            f"{fx(100 * (v('final_train_acc.lplong.99') - v('final_train_acc.lp.99')), 1)}%p 올랐고, 마지막 20에폭 동안 오른 폭은 "
            f"{pcp('train_acc_gain_last20.lplong.99')}다. FT는 {pc('final_train_acc.ft.99', 2)}다.</p>"
            "<p><b>패치 색을 바꾸는 지표는 모두 후속 LP가 본 LP보다 패치에 더 기댔다고 말한다(seed 3개 모두 같은 방향).</b> "
            f"후속 LP − 본 LP: PSR {pp('c.psr.lplong-lp.99')}, 패치 따라가기(우연 보정) {pp('c.follow_excess.lplong-lp.99')}, "
            f"정답 확률 변화 {pp('c.dprob.lplong-lp.99')}, 충돌 OOD 정확도 {pp('c.acc_conf.lplong-lp.99')}.</p>"
            "<p><b>그래도 FT와의 PSR 차이는 4분의 3 넘게 남는다.</b> 남은 몫(후속 LP − FT를 본 LP − FT로 나눈 값)은 "
            f"{pc(kk)} [{fx(100 * v(kk + '.lo'), 1)}, {fx(100 * v(kk + '.hi'), 1)}], seed별 "
            f"{' · '.join(fx(100 * x, 1) for x in ks)}%다. 후속 LP − FT는 PSR "
            f"{pp('c.psr.lplong-ft.99')}(본 LP − FT {ppx('c.psr.lp-ft.99')}), 패치 따라가기(우연 보정) "
            f"{pp('c.follow_excess.lplong-ft.99')}(본 LP − FT {ppx('c.follow_excess.lp-ft.99')})다. 패치를 떼는 대조군에서는 "
            f"후속 LP도 FT보다 더 흔들린다(패치 제거 전환율 후속 LP − FT {pp('c.psr_none.lplong-ft.99')}, 본 LP − FT "
            f"{ppx('c.psr_none.lp-ft.99')}).</p>"
            "<p><b>해석(따로 확인하지 않았다).</b> 얼린 특징에 가중치를 곱해 더하는 head는 에폭을 "
            f"{times:g}배로 늘리고 weight decay를 없애도 학습 데이터를 패치만 보는 규칙({100 * fc['p']:g}%)만큼 맞히지 못했다. "
            "얼린 특징만으로는 FT처럼 패치 색을 직접 읽기 어렵다는 해석(Main RQ 절)과 맞는다. 다만 학습률"
            f"({fc['lr']:g})은 바꾸지 않았으므로, 학습률을 바꿨을 때 LP가 패치를 더 쓰는지는 이 실험으로 알 수 없다.</p>"),
    )

    # ------------------------------------------------------------ 정밀도
    prec = S.get("precision") or {}
    need(len(prec) == 6, "정밀도 확인 모델 6개")
    rho_min = min(x["rho_tf32_vs_ieee"] for x in prec.values())
    drho_max = max(abs(x["rho_absdy_tf32"] - x["rho_absdy_ieee"]) for x in prec.values())
    rel_lo = min(x["rel_diff_median"] for x in prec.values())
    rel_hi = max(x["rel_diff_median"] for x in prec.values())
    need(rho_min > 0.999 and drho_max < 0.01, "TF32 vs IEEE: 순위 상관 > 0.999, ρ 차이 < 0.01")
    T["precision"] = dict(
        title="IG를 TF32로 계산해도 R_patch의 순위와 IG 상관은 IEEE fp32와 같다",
        q="IG를 빠른 TF32 convolution으로 계산해도 결론이 같나?",
        a=(f"같다: seed 0 모델 6개에서 같은 사진 200장의 R_patch 순위 상관은 모두 {floor_fx(rho_min, 3)} 이상이고, "
           f"ρ(R_patch, |Δy|)는 두 방식에서 최대 {ceil_fx(drho_max, 3)} 차이다."),
        intro="", outro=(
            f"<p>R_patch 값 자체의 상대 차이는 모델마다 사진 200장의 중간값으로 {fx(100 * rel_lo, 2)}~{fx(100 * rel_hi, 1)}%다. "
            "순위와 상관이 그대로라 이 보고서의 IG 결론에는 영향이 없다.</p>"),
    )

    # ------------------------------------------------------------ 한눈에 보기 · 핵심
    T["glance"] = [
        ("Main RQ: LP가 FT보다 패치에 덜 기대나?", "mixed", "p와 대조군에 따라 다름",
         "p = 0.99에서 틀린 색 패치에는 훨씬 덜 흔들린다. 패치를 떼었을 때와 p = 0.90의 주 지표로는 아니다.",
         f"PSR LP − FT: p = 0.99 {ppx('c.psr.lp-ft.99')}, p = 0.90 {ppx('c.psr.lp-ft.90')}; "
         f"패치 제거 전환율 p = 0.99 {ppx('c.psr_none.lp-ft.99')}"),
        ("H1: p가 높을 때 LP가 Scratch보다 패치 교체에 덜 흔들리나?", "yes", "지지",
         "그렇다. 차이는 p = 0.99에서 크고 p = 0.90에서 작다.",
         f"PSR LP − Scratch: p = 0.90 {ppx('c.psr.lp-scratch.90')}, p = 0.99 {ppx('c.psr.lp-scratch.99')}"),
        ("H2: p = 0.99에서 LP도 패치를 쓰나?", "yes", "지지",
         f"쓴다. 패치 따라가기(우연 보정) 증가는 FT의 {fx(100 * ratio, 0)}%다. 그러나 p = 0.10 모델보다 줄어든 패치 없음 정확도는 "
         f"LP {gap('pe.acc_none.lp.99')}, FT {gap('pe.acc_none.ft.99')}로 비슷하다.",
         f"패치 따라가기(우연 보정)의 p = 0.10 → 0.99 변화: LP {ppx('pe.follow_excess.lp.99')}, FT {ppx('pe.follow_excess.ft.99')}"),
        ("SQ1: p가 커지면 전략 사이의 차이가 어떻게 바뀌나?", "mixed", "비교 상대에 따라 다름",
         "FT와 비교한 PSR 이점은 p = 0.99에서만, Scratch와 비교한 이점은 p = 0.90부터 나타난다.",
         f"PSR 차이의 차이: LP − FT 0.90 {ppx('did.psr.lp-ft.90')}, 0.99 {ppx('did.psr.lp-ft.99')}; "
         f"LP − Scratch 0.90 {ppx('did.psr.lp-scratch.90')}"),
        ("SQ2·H3: IG 패치 몫이 패치 민감도를 따라가나?", "mixed", "측정에 따라 다름",
         "p = 0.90·0.99에서 셋 다 양의 상관이고 LP가 가장 크다. 같은 클래스 안에서 IG 패치 몫으로 재면 LP는 남고 FT·Scratch는 "
         "크게 약해지며, 패치 안 |A| 합으로 재면 셋 다 남는다.",
         f"p = 0.99 ρ, 전체: LP {num('ig.rho_gray.lp.99')}, FT {num('ig.rho_gray.ft.99')}, Scratch {num('ig.rho_gray.scratch.99')}; "
         f"클래스 안: LP {num('ig.rho_gray_within.lp.99')}, FT {num('ig.rho_gray_within.ft.99')}, "
         f"Scratch {num('ig.rho_gray_within.scratch.99')}; 패치 안 |A| 합, 클래스 안: LP {num('ig.rho_gray_abs_within.lp.99')}, "
         f"FT {num('ig.rho_gray_abs_within.ft.99')}, Scratch {num('ig.rho_gray_abs_within.scratch.99')}"),
    ]
    T["core"] = "".join(f"<li>{x}</li>" for x in [
        f"p = 0.99에서 LP는 FT보다 틀린 색 패치에 훨씬 덜 흔들렸다(PSR {pc('psr.lp.99')} 대 {pc('psr.ft.99')}; head를 "
        f"{times:g}배 오래, weight decay 없이 학습한 후속 실험에서도 {pc('psr.lplong.99')} 대 {pc('psr.ft.99')}). "
        "패치를 떼었을 때와 p = 0.90에서는 그렇지 않았다.",
        f"LP도 p = 0.99에서 패치를 썼다. 틀린 색 패치를 붙인 사진의 정확도가 p = 0.10 모델 {pc('acc_conf.lp.10')}에서 "
        f"p = 0.99 모델 {pc('acc_conf.lp.99')}로 낮아졌다.",
        f"Scratch는 p = 0.99에서 패치 색만 바꿔도 {pc('psr.scratch.99')}의 답이 바뀌어 세 전략 중 가장 크게 기댔다.",
        "p = 0.90·0.99에서 IG 패치 몫은 세 전략 모두 패치 민감도와 같이 움직였지만, 같은 클래스 사진끼리 비교하면 LP의 관계는 "
        "남고 FT·Scratch의 관계는 크게 약해졌다(패치 안 |A| 합으로 재면 셋 다 남는다).",
    ])
    T["limits_extra"] = (
        f"<li>p = 0.90의 LP − FT 판정('아니다')은 미리 정한 주 지표 PSR({ppx('c.psr.lp-ft.90')})과 충돌 OOD 정확도"
        f"({ppx('c.acc_conf.lp-ft.90')})에 따른 것이다. 패치 따라가기(우연 보정)로만 LP가 {gap('c.follow_excess.lp-ft.90')} "
        "덜 따라간다.</li>"
        "<li>IG가 어긋난 사례는 seed 0 모델에서 모델마다 6장씩만 골랐다.</li>")
    return T
