"""汇总 120 张样本上四个引擎的表现，并做配对显著性检验（McNemar 精确检验）。"""
import json, os, math

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"

GT = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))
logit = {r["file"].replace(".png", ""): r for r in json.load(open(os.path.join(W, "logit_check.json"), encoding="utf-8"))}
res2 = {r["id"]: r for r in json.load(open(os.path.join(W, "samples2_resnet.json"), encoding="utf-8"))}
tess = json.load(open(os.path.join(W, "tess_120.json"), encoding="utf-8"))

P = {}
for sid in GT:
    if sid in logit:
        P[sid] = {"resnet_new": logit[sid]["new"], "resnet_old": logit[sid]["old_final"]}
    else:
        P[sid] = {"resnet_new": res2[sid]["resnet_new"], "resnet_old": res2[sid]["resnet_old"]}
    P[sid]["tess_now"] = tess["A"].get(sid, {}).get("pred", "")
    P[sid]["tess_new"] = tess["B"].get(sid, {}).get("pred", "")

IDS = sorted(GT)
ENGINES = [
    ("tess_now",  "你现在跑的（Tesseract 原图直喂）"),
    ("tess_new",  "新脚本兜底（Tesseract 调优）"),
    ("resnet_old","原版 ONNX 后处理（含启发式）"),
    ("resnet_new","新脚本主路径（ResNet ONNX）"),
]

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))

def binom_two_sided(b, c):
    """McNemar 精确检验：b 次在 (b+c) 中服从 p=0.5 的二项分布"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

hits = {}
print("=" * 78)
print(f"{'引擎':<34}{'命中':>8}{'准确率':>9}   95% Wilson 区间")
print("=" * 78)
for key, label in ENGINES:
    k = sum(1 for i in IDS if P[i][key] == GT[i])
    hits[key] = k
    lo, hi = wilson(k, len(IDS))
    print(f"{label:<34}{k:>4}/{len(IDS):<4}{k/len(IDS)*100:>8.1f}%   [{lo*100:.1f}%, {hi*100:.1f}%]")

print("\n" + "=" * 78)
print("按验证码长度拆分")
print("=" * 78)
print(f"{'引擎':<34}{'4 位':>12}{'5 位':>12}")
for key, label in ENGINES:
    i4 = [i for i in IDS if len(GT[i]) == 4]
    i5 = [i for i in IDS if len(GT[i]) == 5]
    k4 = sum(1 for i in i4 if P[i][key] == GT[i])
    k5 = sum(1 for i in i5 if P[i][key] == GT[i])
    print(f"{label:<34}{f'{k4}/{len(i4)} ({k4/len(i4)*100:.0f}%)':>12}{f'{k5}/{len(i5)} ({k5/len(i5)*100:.0f}%)':>12}")

print("\n" + "=" * 78)
print("配对比较（McNemar 精确检验，同一批样本上的成对结果）")
print("=" * 78)
pairs = [("resnet_new", "tess_now"), ("resnet_new", "tess_new"),
         ("resnet_new", "resnet_old"), ("resnet_old", "tess_now")]
lab = dict(ENGINES)
for a, b in pairs:
    b_only = sum(1 for i in IDS if P[i][a] != GT[i] and P[i][b] == GT[i])
    a_only = sum(1 for i in IDS if P[i][a] == GT[i] and P[i][b] != GT[i])
    both_wrong = sum(1 for i in IDS if P[i][a] != GT[i] and P[i][b] != GT[i])
    p = binom_two_sided(b_only, a_only)
    verdict = "显著 (p<0.05)" if p < 0.05 else "不显著"
    print(f"\n{lab[a]}  vs  {lab[b]}")
    print(f"  前者对后者错: {a_only}   后者对前者错: {b_only}   两者都错: {both_wrong}")
    print(f"  McNemar 精确 p = {p:.4f}  ->  {verdict}")

# 错例明细
print("\n" + "=" * 78)
print("错例明细")
print("=" * 78)
for key, label in ENGINES:
    wrong = [(i, GT[i], P[i][key]) for i in IDS if P[i][key] != GT[i]]
    print(f"\n{label}  错 {len(wrong)} 张:")
    for i, t, pv in wrong:
        print(f"   {i}  真值={t:<6} 结果={pv or '∅'}")

json.dump({"hits": hits, "n": len(IDS),
           "detail": {i: {"truth": GT[i], **P[i]} for i in IDS}},
          open(os.path.join(W, "final_120.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n已写入 final_120.json")
