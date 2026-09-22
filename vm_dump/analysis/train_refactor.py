import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""架构重构训练脚本 —— 「加宽全局向量 + 加深输出头 + 数据增广 + 正则」

设计依据（来自 220 张实测分析）：
  * 当前架构：backbone -> AveragePool -> Reshape(1,64) -> 5 个单层 Gemm 头
  * 实测瓶颈：64 维特征"过度专属"（跨集迁移仅 32~74%），MLP 头优于线性头 +3~8pp
  * 实测上限：改架构天花板约 98.5%~99.0%（受 1~2px 笔画物理限制）
  * 明确排除：灰度输入（实测更差）、纯扩容量（会加剧过拟合，必须配合增广/正则）

本脚本提供三个可独立运行的子命令：
  extract   —— 从现有 ONNX 抽出 backbone 特征，缓成 npz（避免每次重训都跑推理）
  train     —— 训练新的分类头（可调宽度/深度/Dropout/权重衰减/增广）
  eval      —— 在固定划分（调参集/留出集）上评估，并与基线做 McNemar 对比

依赖：numpy（必需）；onnxruntime + Pillow（仅 extract 需要）

用法：
  python train_refactor.py extract
  python train_refactor.py train --width 128 --hidden 64 --epochs 8000 --aug
  python train_refactor.py eval
"""
import os, json, argparse, math
import numpy as np

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"
FEAT_NPZ = os.path.join(VD, "backbone_feats.npz")
HEAD_NPZ = os.path.join(VD, "refactor_head.npz")
EXP_JSON = os.path.join(VD, "refactor_eval.json")


# ------------------------------------------------------------------ 数据

def find_images():
    paths = {}
    for d in ["samples", "samples2", "holdout"]:
        dd = os.path.join(VD, d)
        if os.path.isdir(dd):
            for f in os.listdir(dd):
                if f.endswith(".png"):
                    paths[os.path.splitext(f)[0]] = os.path.join(dd, f)
    return paths


def preprocess_py(path):
    """Python 侧预处理：与脚本实现严格一致（ITU-R 601-2 luma + round + >=156）"""
    from PIL import Image
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (np.round(g) >= 156).astype(np.float32)


def extract():
    """抽出 backbone 特征（AveragePool 前的全局池化输出）并缓存"""
    import onnxruntime as rt
    import onnx

    gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
    paths = find_images()
    keys = sorted(k for k in gt if k in paths)

    model = os.path.join(VD, "nn_model.onnx")
    m = onnx.load(model)
    FEAT = "/Reshape_output_0"
    if not any(o.name == FEAT for o in m.graph.output):
        vi = onnx.helper.ValueInfoProto(); vi.name = FEAT
        m.graph.output.append(vi)
    tmp = os.path.join(VD, "_tmp_extract.onnx")
    onnx.save(m, tmp)

    so = rt.SessionOptions(); so.log_severity_level = 3
    sess = rt.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]
    fi = onames.index(FEAT)

    F, LAB, LOGITS = [], [], []
    for k in keys:
        x = preprocess_py(paths[k])[None, None]
        r = sess.run(onames, {iname: x})
        F.append(r[fi][0])
        # 注意：第 5 个头是 27 类，前 4 个是 26 类，shape 不一致，
        # 不能 np.stack 成一个数组。存成 object 数组保持原样。
        LOGITS.append(np.array([o[0] for o in r[:5]], dtype=object))
        LAB.append(gt[k])
    os.remove(tmp)

    np.savez_compressed(FEAT_NPZ,
                        keys=np.array(keys),
                        feats=np.array(F, dtype=np.float64),
                        logits=np.array(LOGITS, dtype=object),
                        labels=np.array(LAB))
    print(f"已缓存 {len(keys)} 张的 backbone 特征 -> {FEAT_NPZ}")
    print(f"  特征维度 {np.array(F).shape}")
    return FEAT_NPZ


def load_feats():
    if not os.path.exists(FEAT_NPZ):
        raise SystemExit("缺少特征缓存，请先运行：python train_refactor.py extract")
    d = np.load(FEAT_NPZ, allow_pickle=True)
    return list(d["keys"]), d["feats"], list(d["logits"]), list(d["labels"])


# ------------------------------------------------------------- 增广

def augment(x, rng):
    """在特征空间的轻量增广。

    为什么不在像素空间增广：提取特征后已无法回到像素。
    这里用的是特征空间噪声（等价于对 backbone 输出做抖动），
    配合 Dropout 一起抑制"样本专属化"。

    注意：这是权宜之计。真正的像素级增广（平移/缩放/弹性形变）
    需要在 extract 阶段对图像做变换，效果更好但要多跑几遍推理。
    """
    scale = rng.normal(1.0, 0.05, size=x.shape)
    return x * scale


# ------------------------------------------------------------- 模型

def init_params(D, width, hidden, C=26, seed=0):
    """初始化：先投影到 width 维（模拟"加宽全局向量"），再走 hidden 层"""
    r = np.random.default_rng(seed)
    P = {
        "Wp": r.normal(0, np.sqrt(2.0 / D), (D, width)), "bp": np.zeros(width),
        "W1": r.normal(0, np.sqrt(2.0 / max(width, 1)), (width, hidden)), "b1": np.zeros(hidden),
        "W2": r.normal(0, np.sqrt(2.0 / max(hidden, 1)), (hidden, C)), "b2": np.zeros(C),
    }
    return P


def forward(P, X, dropout=0.0, rng=None, training=False):
    Hp = np.maximum(0, X @ P["Wp"] + P["bp"])
    if training and dropout > 0 and rng is not None:
        mask = (rng.random(Hp.shape) > dropout) / (1.0 - dropout)
        Hp = Hp * mask
    H1 = np.maximum(0, Hp @ P["W1"] + P["b1"])
    Z = H1 @ P["W2"] + P["b2"]
    return Z, (Hp, H1)


def softmax(Z):
    Ze = np.exp(Z - Z.max(1, keepdims=True))
    return Ze / Ze.sum(1, keepdims=True)


def train(Xtr, Ytr, width=128, hidden=64, epochs=6000, lr=0.05,
          wd=1e-3, dropout=0.2, aug=False, seed=0, C=26, verbose=False):
    rng = np.random.default_rng(seed)
    D = Xtr.shape[1]
    P = init_params(D, width, hidden, C, seed)
    Yoh = np.eye(C)[Ytr]
    n = len(Xtr)
    best = None
    for ep in range(epochs):
        Xb = augment(Xtr, rng) if aug else Xtr
        Z, (Hp, H1) = forward(P, Xb, dropout, rng, training=dropout > 0)
        Pp = softmax(Z)
        dZ = (Pp - Yoh) / n
        gW2 = H1.T @ dZ + wd * P["W2"]; gb2 = dZ.sum(0)
        dH1 = dZ @ P["W2"].T; dH1[H1 <= 0] = 0
        gW1 = Hp.T @ dH1 + wd * P["W1"]; gb1 = dH1.sum(0)
        dHp = dH1 @ P["W1"].T; dHp[Hp <= 0] = 0
        gWp = Xb.T @ dHp + wd * P["Wp"]; gbp = dHp.sum(0)
        P["W2"] -= lr * gW2; P["b2"] -= lr * gb2
        P["W1"] -= lr * gW1; P["b1"] -= lr * gb1
        P["Wp"] -= lr * gWp; P["bp"] -= lr * gbp
        if verbose and (ep + 1) % 2000 == 0:
            acc = (softmax(forward(P, Xtr)[0]).argmax(1) == Ytr).mean()
            print(f"    ep{ep+1:5d} 训练集准确率 {acc*100:.2f}%")
    return P


def predict(P, X):
    return softmax(forward(P, X)[0]).argmax(1)


# ------------------------------------------------------------- 评估

def mcnemar(a_ok, b_ok):
    """McNemar 精确检验：返回 (改进数, 退化数, p 值)"""
    A = sum(1 for i in range(len(a_ok)) if a_ok[i] and not b_ok[i])   # a 对 b 错
    B = sum(1 for i in range(len(a_ok)) if not a_ok[i] and b_ok[i])   # a 错 b 对
    n = A + B
    if n == 0:
        return A, B, 1.0
    # 双尾精确检验
    p = 2 * sum(math.comb(n, k) * (0.5 ** n) for k in range(0, min(A, B) + 1))
    return A, B, min(1.0, p)


def cmd_train(args):
    keys, F, LOGITS, LBL = load_feats()
    print(f"载入 {len(keys)} 张特征，维度 {F.shape}")

    tune = [i for i, k in enumerate(keys) if not k.startswith("h")]
    hold = [i for i, k in enumerate(keys) if k.startswith("h")]
    print(f"调参集 {len(tune)} / 留出集 {len(hold)}")

    results = {}
    # 对每个位置分别训练（与原模型 5 头结构对位）
    for pos in range(5):
        tr = [i for i in tune if len(LBL[i]) > pos]
        te = [i for i in hold if len(LBL[i]) > pos]
        if len(tr) < 40 or len(te) < 20:
            continue
        Xtr = F[tr]; Ytr = np.array([CH.index(LBL[i][pos]) for i in tr])
        Xte = F[te]; Yte = np.array([CH.index(LBL[i][pos]) for i in te])

        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtrn, Xten = (Xtr - mu) / sd, (Xte - mu) / sd

        # 基线：原生头
        nat = []
        for i in te:
            v = LOGITS[i][pos]
            j = int(np.argmax(v))
            nat.append(j < 26 and CH[j] == LBL[i][pos])
        nat_acc = sum(nat) / len(nat)

        # 新架构
        P = train(Xtrn, Ytr, width=args.width, hidden=args.hidden,
                  epochs=args.epochs, lr=args.lr, wd=args.wd,
                  dropout=args.dropout, aug=args.aug, seed=args.seed)
        pr = predict(P, Xten)
        new_ok = [int(pr[i]) == Yte[i] for i in range(len(Yte))]
        new_acc = sum(new_ok) / len(new_ok)

        A, B, p = mcnemar(new_ok, nat)
        results[pos] = {"native": nat_acc, "new": new_acc, "A": A, "B": B, "p": p}
        star = " ★" if new_acc > nat_acc else ""
        print(f"  位{pos+1}: 原生 {nat_acc*100:6.2f}%  新架构 {new_acc*100:6.2f}%{star}"
              f"  改进{A} 退化{B} p={p:.3f}")

    json.dump(results, open(EXP_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n结果写入 {EXP_JSON}")
    if results:
        nn = np.mean([v["native"] for v in results.values()])
        nw = np.mean([v["new"] for v in results.values()])
        print(f"平均：原生 {nn*100:.2f}% -> 新架构 {nw*100:.2f}%  ({'+' if nw>nn else ''}{(nw-nn)*100:.2f}pp)")


def cmd_eval(args):
    if not os.path.exists(EXP_JSON):
        raise SystemExit("没有评估结果，请先运行 train")
    r = json.load(open(EXP_JSON, encoding="utf-8"))
    print(f"{'位置':>6s} {'原生':>10s} {'新架构':>10s} {'改进':>6s} {'退化':>6s} {'p':>8s}")
    for pos in sorted(r, key=lambda x: int(x)):
        v = r[pos]
        print(f"  位{int(pos)+1:>2d} {v['native']*100:9.2f}% {v['new']*100:9.2f}% "
              f"{v['A']:6d} {v['B']:6d} {v['p']:8.3f}")


def cmd_extract(args):
    extract()


def main():
    ap = argparse.ArgumentParser(description="架构重构：训练/评估")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract", help="抽取并缓存 backbone 特征")
    sub.add_parser("eval", help="查看评估结果")

    t = sub.add_parser("train", help="训练新分类头")
    t.add_argument("--width", type=int, default=128, help="投影宽度（模拟加宽全局向量）")
    t.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    t.add_argument("--epochs", type=int, default=6000)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--wd", type=float, default=1e-3, help="权重衰减（正则）")
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--aug", action="store_true", help="启用特征空间增广")
    t.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
