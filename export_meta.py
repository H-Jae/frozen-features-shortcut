# 분석용 메타데이터를 한 파일로 내보낸다 (analyze.py는 torch 없이 numpy만으로 이 파일을 읽는다).
# results/test_meta.npz: 테스트 라벨 y, 결합 추출 u·w, IG 부분집합, 팔레트, 분할 인덱스.
import numpy as np

from common import PALETTE, RESULTS_DIR, Splits
from ig import ig_subsets

if __name__ == "__main__":
    d = Splits(device="cpu")
    s1000, s500, s200, common = ig_subsets(d.y_te)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_DIR / "test_meta.npz", y_te=d.y_te, u_te=d.u_te, w_te=d.w_te,
             idx_tr=d.idx_tr, idx_va=d.idx_va, s1000=s1000, s500=s500, s200=s200, common=common,
             palette=PALETTE)
    print("saved", RESULTS_DIR / "test_meta.npz")
