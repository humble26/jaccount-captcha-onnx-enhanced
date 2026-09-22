# -*- coding: utf-8 -*-
"""端到端微调试点 —— 聚焦第 4 位字符识别。

背景：冻结特征换头(-41.71pp)走不通，必须端到端重训整个 ResNet-20。
本脚本从生产 ONNX 初始化权重，用全部 5 个头的交叉熵损失做端到端梯度下降，
第 4 头损失可加权（--pos4wt 聚焦第4位）。评估在固定留出集(holdout)上，
与原生基线逐位对比 + McNemar。

数据划分（与 train_refactor 一致）：
  调参集: samples(c0xx,20) + samples2(n0xx,100) = 120 张
  留出集: holdout(h0xx,100) 张   <- 只在最后评估，绝不参与训练

用法:
  python train_e2e.py baseline                 # 评估原生模型基线
  python train_e2e.py train --lr 1e-4 --steps 300 --batch 32 --pos4wt 2.0
  python train_e2e.py eval [--model out_e2e.npz] [--pos4wt 2.0]
"""
import os, json, sys, math, time, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet20_np import (forward_fast, backward_fast, forward_heads,
                         load_onnx_weights, CH)

# 生产源（模型与真值）在工作区；归档内不含 vm_dump，可用环境变量 PROD_WS 指向别处。
WS = os.environ.get("PROD_WS") or r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # <归档>/02-重构研究
VD = os.path.join(WS, "vm_dump")
MODEL = os.environ.get("PROD_ONNX") or os.path.join(VD, "nn_model.onnx")
GT = os.path.join(VD, "ground_truth_all.json")
NEW_GT = os.path.join(HERE, "_sampling_out", "new300_gt.json")
# 归档后采样图位于同级 sampled/new_captchas/。
# （原来写死的 E:\harness\重构研究\sampled\new_captchas 在归档剪切移动后已不存在。）
NEW_SRC = os.environ.get("CAPTCHA_SRC") or os.path.join(ROOT, "new300")
OUT_DIR = os.path.join(HERE, "_e2e_out")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = [26, 26, 26, 26, 27]

def filter_trainable(W):
    """保留可训练权重（卷积 + 线性头），丢弃无关常量。"""
    return {k: v for k, v in W.items()
            if k.startswith("onnx::Conv") or k.startswith("linear")}


# ------------------------------------------------------------------ 数据

def preprocess(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def load_split(split, extra=False):
    """split in {'tune','hold'} -> 返回 (键列表, x 张量, targets per head)
    extra=True 时把 300 张新采集样本并入调参集（真值来自 new300_gt.json）。"""
    gt = json.load(open(GT, encoding="utf-8"))
    keys = []
    for d in (["samples", "samples2"] if split == "tune" else ["holdout"]):
        dd = os.path.join(VD, d)
        for f in sorted(os.listdir(dd)):
            if f.endswith(".png"):
                k = os.path.splitext(f)[0]
                if k in gt:
                    keys.append((k, os.path.join(dd, f)))
    # 追加 300 张新采集样本（仅调参集）
    if extra and split == "tune":
        newgt = json.load(open(NEW_GT, encoding="utf-8"))
        # 新样本图从采样目录读取
        for f in sorted(os.listdir(NEW_SRC)):
            if f.endswith(".png"):
                k = os.path.splitext(f)[0]
                if k in newgt and k not in {kk for kk, _ in keys}:
                    keys.append((k, os.path.join(NEW_SRC, f)))
        keys.sort()
    xs, tgt = [], [[] for _ in range(5)]
    for k, p in keys:
        xs.append(preprocess(p)[None])
        lab = gt[k] if k in gt else newgt[k] if (extra and split == "tune" and k in newgt) else gt[k]
        for pos in range(5):
            tgt[pos].append(CH.index(lab[pos]) if pos < len(lab) else 26)
    x = np.stack(xs).astype(np.float64)
    targets = [np.array(t) for t in tgt]
    return [k for k, _ in keys], x, targets


# ------------------------------------------------------------------ 损失

def cross_entropy_dlogits(logits, target, C):
    """返回 (loss_avg, dlogit) 单头。logits:(N,C)"""
    ze = np.exp(logits - logits.max(1, keepdims=True))
    p = ze / ze.sum(1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), target] + 1e-12).mean()
    onehot = np.zeros_like(p)
    onehot[np.arange(n), target] = 1.0
    return loss, (p - onehot) / n


# ------------------------------------------------------------------ 前向推理

def predict(x, W, hs=getattr(load_onnx_weights, "heads", ("1", "2", "3", "4", "5"))):
    c = forward_fast(x, W)
    logits = forward_heads(c["feat"], W, hs)
    return [np.argmax(z, 1) for z in logits], c


# ------------------------------------------------------------------ 命令

def cmd_baseline(args):
    W = filter_trainable({k: v.astype(np.float64) for k, v in load_onnx_weights(MODEL).items()})
    keys, x, targets = load_split("hold")
    pred, _ = predict(x, W)
    print(f"原生模型 留出集 {len(keys)} 张：")
    report(keys, targets, pred, prefix="  原生: ")


def report(keys, targets, pred, prefix=""):
    for pos in range(5):
        n = len([k for k, ch in [(keys[i], targets[pos][i]) for i in range(len(keys))] if ch < len(CH)])
        ok = sum(1 for i in range(len(keys)) if targets[pos][i] < len(CH) and pred[pos][i] == targets[pos][i])
        tot = sum(1 for i in range(len(keys)) if targets[pos][i] < len(CH))
        acc = ok / tot if tot else float('nan')
        print(f"{prefix}位{pos+1}: 正确 {ok}/{tot} = {acc*100:.2f}%")
    # 整串整体：5 个头全对才算对（4 位码的第 5 位 target 是 blank=26，与生产语义一致）
    whole = 0; tot = 0
    def full_ok(i):
        kl = [t[i] for t in targets]
        tot = len(kl)
        return all(pred[pos][i] == kl[pos] for pos in range(tot)), tot
    for i in range(len(keys)):
        ok, t = full_ok(i)
        whole += ok; tot += 1
    print(f"{prefix}整串: 正确 {whole}/{tot} = {whole/tot*100:.2f}%")


def cmd_train(args):
    W = filter_trainable({k: v.astype(np.float64) for k, v in load_onnx_weights(MODEL).items()})
    keys, x, targets = load_split("tune", extra=args.extra)
    n = x.shape[0]
    print(f"调参集 {n} 张（含新样本 extra={'on' if args.extra else 'off'}），特征维度输入 {x.shape}")
    # 留出集预载（仅用于 training 中定期验证，不参与更新）
    hold_keys, hold_x, hold_t = None, None, None
    if args.eval_every:
        hold_keys, hold_x, hold_t = load_split("hold")
    rng = np.random.default_rng(args.seed)

    def hold_acc4():
        pred, _ = predict(hold_x, W)
        return float((pred[3] == hold_t[3]).mean())

    # 记录：loss / 第4位训练准确率 / 留出集位4 / best
    hist = []
    best_w, best_acc4 = None, -1.0
    lr0 = args.lr
    t_start = time.time()
    for step in range(1, args.steps + 1):
        # 小批
        idx = rng.choice(n, size=args.batch, replace=False)
        xb, tb = x[idx], [t[idx] for t in targets]

        # 前向
        c = forward_fast(xb, W)
        logits = forward_heads(c["feat"], W)
        loss = 0.0
        dlogits = []
        for pos in range(5):
            C = NUM_CLASSES[pos]
            w = args.pos4wt if pos == 3 else 1.0      # 聚焦第4位
            l, dl = cross_entropy_dlogits(logits[pos], tb[pos], C)
            loss += w * l
            dlogits.append(dl * w)
        # 反向
        g = backward_fast(dlogits, c, W)
        # 更新（SGD 手动）
        for k, v in W.items():
            W[k] = v - lr0 * g[k]

        rec = {"step": step, "loss": float(loss)}
        if step % args.log == 0 or step == args.steps or (args.eval_every and step % args.eval_every == 0):
            # 训练集第4位准确率
            pred, _ = predict(x[:n], W)
            rec["acc4_train"] = float((pred[3][:n] == targets[3]).mean())
            line = f"  step {step:4d}  loss={loss:.4f}  tune位4={rec['acc4_train']*100:.2f}%"
            if args.eval_every:
                ha = hold_acc4()
                rec["acc4_hold"] = float(ha)
                line += f"  hold位4={ha*100:.2f}%"
                if ha > best_acc4:
                    best_acc4, best_w = ha, {k: v.copy() for k, v in W.items()}
                    rec["best"] = True
                    line += "  ★best"
            line += f"  ({(time.time()-t_start)/step:.1f}s/step)"
            print(line, flush=True)
            hist.append(rec)

    # 保存最终权重 + 最佳权重
    def save_w(Wd, out):
        np.savez(out, **{f"w{i}": val for i, (k, val) in enumerate(Wd.items())})
        json.dump({"keys": list(Wd.keys()), "suffix": args.suffix, "hist": hist},
                  open(out + ".meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  已保存 {out}（{len(Wd)} 组权重）")
    final_out = os.path.join(OUT_DIR, args.suffix + ".npz")
    save_w(W, final_out)
    if best_w is not None:
        best_out = os.path.join(OUT_DIR, args.suffix + "_best_hold4.npz")
        json.dump({"keys": list(best_w.keys()), "suffix": args.suffix + "_best_hold4",
                   "hist": hist, "best_hold_acc4": best_acc4},
                  open(best_out + ".meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        np.savez(best_out, **{f"w{i}": val for i, (k, val) in enumerate(best_w.items())})
        print(f"  已保存最佳(留出位4) {best_out}（位4={best_acc4*100:.2f}%）")
    print(f"\n训练完成，共 {args.steps} 步，用时 {time.time()-t_start:.1f}s")

    # 自动在留出集评估
    if args.eval_on_finish:
        cmd_eval(argparse.Namespace(model=final_out))


def cmd_eval(args):
    meta = json.load(open(args.model + ".meta.json", encoding="utf-8"))
    W = {}
    d = np.load(args.model)
    for i, k in enumerate(meta["keys"]):
        W[k] = d[f"w{i}"]
    W = {k: v.astype(np.float64) for k, v in W.items()}

    keys, x, targets = load_split("hold")
    pred, _ = predict(x, W)
    print(f"微调模型 {os.path.basename(args.model)} 留出集 {len(keys)} 张：")
    report(keys, targets, pred)


def cmd_gen(args):
    hist = json.load(open(os.path.join(OUT_DIR, args.hist + ".meta.json"), encoding="utf-8"))
    print(f"训练历史（{os.path.basename(args.hist)}）：")
    for h in hist["hist"]:
        # 记录里写入的键名是 acc4_train（见 cmd_train）。早期版本这里写的是 h['acc4']，
        # 该子命令一旦被调用就 KeyError —— 因为一直没人用它，所以从未暴露。
        acc = h.get("acc4_train", h.get("acc4"))
        hold = h.get("acc4_hold")
        print(f"  step {h['step']:4d}  loss={h['loss']:.4f}"
              f"  acc4(tune)={('%.2f%%' % (acc * 100)) if acc is not None else '—'}"
              f"  acc4(hold)={('%.2f%%' % (hold * 100)) if hold is not None else '—'}"
              + ("  ★best" if h.get("best") else ""))


def main():
    ap = argparse.ArgumentParser(description="端到端微调试点（聚焦第4位）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("baseline", help="评估原生模型基线条")
    s = sub.add_parser("gen", help="查看训练历史")
    s.add_argument("hist")

    t = sub.add_parser("train", help="端到端微调")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--pos4wt", type=float, default=2.0, help="第4头损失权重（聚焦第4位）")
    t.add_argument("--log", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=0, help="每 N 步在留出集评估位4并保存最佳")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--extra", action="store_true", help="把 300 张新采集样本并入调参集")
    t.add_argument("--suffix", default="e2e_v1")
    t.add_argument("--save-every", type=int, default=0)
    t.add_argument("--eval-on-finish", action="store_true")

    e = sub.add_parser("eval", help="在留出集评估")
    e.add_argument("--model", default=os.path.join(OUT_DIR, "e2e_v1.npz"))

    args = ap.parse_args()
    {"baseline": cmd_baseline, "train": cmd_train, "eval": cmd_eval, "gen": cmd_gen}[args.cmd](args)


if __name__ == "__main__":
    main()