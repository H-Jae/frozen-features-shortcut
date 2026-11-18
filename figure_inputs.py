# 보고서 첫 그림: 실험에 쓰는 입력 이미지의 예 (학습 배치, 테스트 변형). CPU로 돈다.
# 실행: CUDA_VISIBLE_DEVICES=-1 ..\..\.venv\Scripts\python.exe figure_inputs.py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import CLASSES, MEAN, PALETTE, RESULTS_DIR, STD, Splits, blur224, make_train_batch, upsample
from evaluate import VARIANTS, build_variant

SHOW = ["match", "wrong", "none", "occ_center", "match_tl", "match_s12", "match_s48", "match_blur8"]
LABEL = {"match": "x+ (matching color)", "wrong": "x- (wrong color)", "none": "no patch (removal)",
         "occ_center": "x+ + gray occluder", "match_tl": "x+ moved to top-left", "match_s12": "x+ size 12",
         "match_s48": "x+ size 48", "match_blur8": "object blurred σ=8 + x+"}


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    d = Splits(device="cpu")
    tile, pad, head = 168, 8, 34
    f_small, f_big = font(15), font(17)

    # (a) 팔레트
    pal = Image.new("RGB", (10 * (tile // 2 + pad) + pad, tile // 2 + head + pad), "#fcfcfb")
    dr = ImageDraw.Draw(pal)
    for k in range(10):
        x0 = pad + k * (tile // 2 + pad)
        dr.rectangle([x0, head, x0 + tile // 2, head + tile // 2 - 10], fill=tuple(int(c) for c in PALETTE[k]),
                     outline="#52514e")
        dr.text((x0, 8), f"{k} {CLASSES[k]}", fill="#0b0b0b", font=f_small)

    # (b) 테스트 변형: 이미지 2장
    idx = torch.tensor([3, 7])
    clean = upsample(d.x_te[idx].float() / 255.0)
    bases = {"clean": clean}
    for b in (4, 8, 16):
        bases[f"blur{b}"] = blur224(clean, b)
    y = torch.from_numpy(d.y_te)[idx]
    w = torch.from_numpy(d.w_te)[idx]
    specs = {v[0]: v for v in VARIANTS}
    grid = Image.new("RGB", (len(SHOW) * (tile + pad) + pad, 2 * (tile + pad) + head + pad), "#fcfcfb")
    dr = ImageDraw.Draw(grid)
    for c, name in enumerate(SHOW):
        x = build_variant(specs[name], bases, y, w).clamp(0, 1)
        arr = (x.permute(0, 2, 3, 1).numpy() * 255 + 0.5).astype(np.uint8)
        x0 = pad + c * (tile + pad)
        dr.text((x0, 8), LABEL[name], fill="#0b0b0b", font=f_small)
        for r in range(2):
            grid.paste(Image.fromarray(arr[r]).resize((tile, tile), Image.BILINEAR), (x0, head + r * (tile + pad)))

    # (c) 학습 배치 (p=0.99, seed 0): crop+flip 뒤에 패치
    g = torch.Generator()
    g.manual_seed(0)
    s = torch.from_numpy(d.train_patches(0.99, 0))
    xb = make_train_batch(d.x_tr[:8], s[:8], g)
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    sd = torch.tensor(STD).view(1, 3, 1, 1)
    arr = ((xb * sd + m).clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255 + 0.5).astype(np.uint8)
    tb = Image.new("RGB", (8 * (tile + pad) + pad, tile + head + pad), "#fcfcfb")
    dr = ImageDraw.Draw(tb)
    for j in range(8):
        x0 = pad + j * (tile + pad)
        dr.text((x0, 8), f"{CLASSES[int(d.y_tr[j])]} / patch {int(s[j])}", fill="#0b0b0b", font=f_small)
        tb.paste(Image.fromarray(arr[j]).resize((tile, tile), Image.BILINEAR), (x0, head))

    width = max(pal.width, grid.width, tb.width)
    out = Image.new("RGB", (width, pal.height + grid.height + tb.height + 3 * head), "#fcfcfb")
    dr = ImageDraw.Draw(out)
    yy = 0
    for title, im in (("(a) palette: class k <-> color k", pal),
                      ("(b) test variants of the same two test images", grid),
                      ("(c) training batch at p = 0.99 (random crop + flip, then patch)", tb)):
        dr.text((pad, yy + 8), title, fill="#0b0b0b", font=f_big)
        out.paste(im, (0, yy + head))
        yy += im.height + head
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    out.save(RESULTS_DIR / "figures" / "fig0_inputs.png")
    print("saved", RESULTS_DIR / "figures" / "fig0_inputs.png")
