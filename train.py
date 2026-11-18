# 학습 1회: 전략 하나 x p 하나 x seed 하나 x (lr, wd) 하나.
# 결과 폴더: config.json, log.json(에폭별 곡선), best.pt(ID 검증 정확도가 가장 높은 에폭), done.json
# - LP: backbone 전체를 얼리고 eval 모드로 둔다(BatchNorm 통계도 고정). head(fc)만 학습한다.
# - FT: 같은 ImageNet 체크포인트에서 시작해 전부 학습한다. Scratch: 무작위 초기화에서 전부 학습한다.
# - 체크포인트 선택은 학습과 같은 p로 패치를 붙인 검증 5,000장의 정확도로만 한다 (동률이면 뒤 에폭).
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, Splits, backbone_features, build_model, make_eval_batch,
                    make_train_batch, now)

torch.backends.cudnn.benchmark = True
CL = torch.channels_last


@torch.no_grad()
def val_metrics(model, data, s_va, y_va, batch=500):
    model.eval()
    correct = torch.zeros((), device=DEV)
    loss = torch.zeros((), device=DEV)
    for i in range(0, len(y_va), batch):
        x = make_eval_batch(data.x_va[i:i + batch], s_va[i:i + batch]).contiguous(memory_format=CL)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        y = y_va[i:i + batch]
        loss += F.cross_entropy(logits, y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
    n = len(y_va)
    return correct.item() / n, loss.item() / n


def train(strategy, p, seed, lr, wd, epochs, out_dir, data=None, batch=256, limit=None, quiet=False):
    out_dir = Path(out_dir)
    if (out_dir / "done.json").exists():
        old = json.loads((out_dir / "config.json").read_text())
        want = dict(strategy=strategy, p=p, seed=seed, lr=lr, wd=wd, epochs=epochs, batch=batch, limit=limit)
        bad = {k: (old.get(k), v) for k, v in want.items() if old.get(k) != v}
        if bad:  # 끝난 실행을 다른 설정으로 재사용하지 않는다 (예: tuned.json이 바뀐 뒤 final을 다시 돌릴 때)
            raise ValueError(f"{out_dir.name}: 이미 끝난 실행과 설정이 다르다 {bad}")
        return json.loads((out_dir / "done.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    data = data or Splits()
    cfg = dict(strategy=strategy, p=p, seed=seed, lr=lr, wd=wd, epochs=epochs, batch=batch,
               optimizer="SGD(momentum=0.9, nesterov=True)", schedule="1-epoch linear warmup + cosine to 0",
               precision="bf16 autocast", limit=limit, started=now(),
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))

    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(data.y_tr) if limit is None else limit
    s_tr = torch.from_numpy(data.train_patches(p, seed)).to(DEV)
    s_va = torch.from_numpy(data.val_patches(p, seed)).to(DEV)
    y_tr = torch.from_numpy(data.y_tr).to(DEV)
    y_va = torch.from_numpy(data.y_va).to(DEV)

    model = build_model(strategy, seed).to(DEV).to(memory_format=CL)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = math.ceil(n / batch)
    total, warm = epochs * steps, steps

    def lr_at(t):
        if t < warm:
            return (t + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)

    log, best, best_ep, t_start = [], -1.0, 0, time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train(strategy != "lp")
        perm = torch.randperm(n, device=DEV, generator=gen)
        tl = torch.zeros((), device=DEV)
        tc = torch.zeros((), device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x = make_train_batch(data.x_tr[idx], s_tr[idx], gen).contiguous(memory_format=CL)
            y = y_tr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if strategy == "lp":
                    with torch.no_grad():
                        feats = backbone_features(model, x)
                    logits = model.fc(feats)
                else:
                    logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.detach() * len(y)
            tc += (logits.detach().argmax(1) == y).sum()
        va_acc, va_loss = val_metrics(model, data, s_va, y_va)
        row = dict(epoch=ep, train_loss=tl.item() / n, train_acc=tc.item() / n, val_acc=va_acc,
                   val_loss=va_loss, lr_end=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1))
        log.append(row)
        if va_acc >= best:
            best, best_ep = va_acc, ep
            state = model.fc.state_dict() if strategy == "lp" else model.state_dict()
            torch.save({k: v.detach().cpu() for k, v in state.items()}, out_dir / "best.pt")
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))
        if not quiet:
            print(f"[{now()}] {out_dir.name} ep{ep}/{epochs} train_loss={row['train_loss']:.4f} "
                  f"train_acc={row['train_acc']:.4f} val_acc={va_acc:.4f} ({row['sec']}s)", flush=True)

    done = dict(best_val_acc=best, best_epoch=best_ep, epochs=epochs,
                total_sec=round(time.time() - t_start, 1), finished=now())
    (out_dir / "done.json").write_text(json.dumps(done, indent=1))
    del model, opt
    torch.cuda.empty_cache()
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["lp", "ft", "scratch"], required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(train(a.strategy, a.p, a.seed, a.lr, a.wd, a.epochs, a.out, limit=a.limit))
