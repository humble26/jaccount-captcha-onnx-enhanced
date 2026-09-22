import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""定向采样器 —— 为「重构」采集有针对性训练数据

设计依据（来自 220 张实测分析）：
  * 错误 100% 集中在第 4 个字符位
  * 5 组混淆对：x/o, c/o, y/u, w/g, g/z
  * 现有数据里这些对照各只有 8~13 例 —— 光加量不够，必须"定向加权"

本脚本做两件事：
  1. 对**已有** 220 张数据做标签级统计，算出每类字符、每组混淆对的缺口
  2. 生成一个**目标配额表**，指导后续采集/筛选应补哪些样本

用法：
  python targeted_sampling.py                 # 只做缺口分析
  python targeted_sampling.py --plan 2000     # 按 2000 张目标生成配额表
"""
import os, json, argparse, itertools
from collections import Counter, defaultdict

WS = _REPO
VD = os.path.join(WS, "vm_dump")
CH = "abcdefghijklmnopqrstuvwxyz"

# 实测得出的 5 组高危混淆对（真值 -> 被误判成）
CONFUSION_PAIRS = [("x", "o"), ("c", "o"), ("y", "u"), ("w", "g"), ("g", "z")]
# 第 4 位是高危位置
HOT_POS = 3


def load():
    gt = json.load(open(os.path.join(VD, "ground_truth_all.json"), encoding="utf-8"))
    paths = {}
    for d in ["samples", "samples2", "holdout"]:
        dd = os.path.join(VD, d)
        if os.path.isdir(dd):
            for f in os.listdir(dd):
                if f.endswith(".png"):
                    paths[os.path.splitext(f)[0]] = os.path.join(dd, f)
    keys = [k for k in gt if k in paths]
    return gt, keys


def analyze(gt, keys):
    total = len(keys)
    print("=" * 82)
    print(f"现有数据总览：{total} 张")
    print("=" * 82)

    # 1) 字符总体分布
    all_ch = Counter()
    for k in keys:
        all_ch.update(gt[k])
    n_char = sum(all_ch.values())
    print(f"\n字符总数 {n_char}，平均每字符 {n_char/26:.1f} 次")
    print("最少的 8 个字符（越少越容易欠拟合）：")
    for ch, c in all_ch.most_common()[-8:]:
        print(f"    {ch}: {c:4d} 次")
    print("最多的 8 个字符：")
    for ch, c in all_ch.most_common(8):
        print(f"    {ch}: {c:4d} 次")

    # 2) 第 4 位的字符分布（高危位置）
    pos4 = Counter()
    n_pos4 = 0
    for k in keys:
        if len(gt[k]) > HOT_POS:
            pos4[gt[k][HOT_POS]] += 1
            n_pos4 += 1
    print(f"\n第 4 位字符分布（共 {n_pos4} 张）—— 高危位置")
    print(f"    最少的 10 个: " + ", ".join(f"{c}={n}" for c, n in pos4.most_common()[-10:]))

    # 3) 混淆对现状
    print("\n" + "=" * 82)
    print("5 组高危混淆对现状")
    print("=" * 82)
    print(f"{'混淆对':>10s} {'真值总次数':>10s} {'在第4位':>9s} {'误判目标次数':>12s} {'缺口评估':>10s}")
    gaps = {}
    for a, b in CONFUSION_PAIRS:
        ca_total = all_ch.get(a, 0)
        # 真值 a 出现在第 4 位的次数
        ca_pos4 = sum(1 for k in keys if len(gt[k]) > HOT_POS and gt[k][HOT_POS] == a)
        cb_total = all_ch.get(b, 0)
        # 经验值：每组至少需要 ~60 例才有希望学到字形差别（当前 8~13，严重不足）
        need = max(0, 60 - ca_pos4)
        gaps[(a, b)] = need
        print(f"    {a}->{b:>4s} {ca_total:10d} {ca_pos4:9d} {cb_total:12d} {need:>10d}")

    return all_ch, pos4, gaps


def build_plan(all_ch, pos4, gaps, target_n):
    """按目标总量生成采样配额表"""
    print("\n" + "=" * 82)
    print(f"目标配额表（目标样本量 {target_n} 张）")
    print("=" * 82)

    # 每张验证码 4~5 个字符，平均约 4.5
    chars_needed = int(target_n * 4.5)
    per_char = chars_needed // 26
    print(f"共需字符位约 {chars_needed} 个，平均每字符 {per_char} 次")

    plan = {"total": target_n, "per_char_baseline": per_char, "boost": []}

    # 基础配额
    print(f"\n[基础] 每字符至少 {per_char} 次")
    # 高危字符额外加权
    hot_chars = set()
    for a, b in CONFUSION_PAIRS:
        hot_chars.add(a)
        hot_chars.add(b)
    boost = max(20, per_char // 2)
    print(f"[加权] 高危字符（{len(hot_chars)} 个：{''.join(sorted(hot_chars))}）额外 +{boost} 次")

    final = {}
    for ch in CH:
        base = per_char
        if ch in hot_chars:
            base += boost
        final[ch] = base

    print(f"\n{'字符':>4s} {'现有':>6s} {'目标':>6s} {'需补':>6s} {'是否高危':>8s}")
    shortfall = {}
    for ch in CH:
        cur = all_ch.get(ch, 0)
        tgt = final[ch]
        need = max(0, tgt - cur)
        shortfall[ch] = need
        print(f"  {ch:>2s} {cur:6d} {tgt:6d} {need:6d} {'★' if ch in hot_chars else '':>8s}")

    # 第 4 位专项配额
    print(f"\n[专项] 第 4 位：每组混淆对至少需要 60 例真值样本")
    for (a, b), need in gaps.items():
        cur = sum(1 for k in [] )  # 占位，实际次数在 analyze 里算过
        print(f"    {a} 在第 4 位需再补约 {need} 例（真值={a}，期望被识别为 {a} 而非 {b}）")

    plan["char_quota"] = final
    plan["char_shortfall"] = shortfall
    plan["hot_chars"] = sorted(hot_chars)
    plan["confusion_pairs"] = CONFUSION_PAIRS
    plan["hot_position"] = HOT_POS

    out = os.path.join(VD, "sampling_plan.json")
    json.dump(plan, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n配额表已写入 {out}")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=int, default=0, help="目标总样本量，用于生成配额表")
    args = ap.parse_args()

    gt, keys = load()
    all_ch, pos4, gaps = analyze(gt, keys)

    total_short = 0
    print("\n" + "=" * 82)
    print("缺口汇总")
    print("=" * 82)
    hot = set()
    for a, b in CONFUSION_PAIRS:
        hot.add(a); hot.add(b)
    for ch in sorted(hot):
        c = all_ch.get(ch, 0)
        print(f"  高危字符 {ch}: 现有 {c:4d} 次")
        total_short += max(0, 60 - c)

    print(f"\n粗略估计：要让 8 个高危字符各达到 60 次，还需补约 {total_short} 个字符位")
    print(f"（按每张 4.5 字符算，约 {int(total_short/4.5)} 张验证码）")

    if args.plan:
        build_plan(all_ch, pos4, gaps, args.plan)


if __name__ == "__main__":
    main()
