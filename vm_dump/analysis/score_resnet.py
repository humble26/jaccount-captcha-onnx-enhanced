import json, os

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"

# 真值：由人肉眼逐张辨认（见 blind_sheet.png，盲标不参考任何模型输出）
TRUTH = {
    "c00": "yvcew", "c01": "fcvtf", "c02": "ymoad", "c03": "qgjnf", "c04": "txju",
    "c05": "fomr",  "c06": "bbmbh", "c07": "wxmh",  "c08": "kwwkc", "c09": "afex",
    "c10": "fsyeu", "c11": "jgvc",  "c12": "riixo", "c13": "oanu",  "c14": "ugal",
    "c15": "odeb",  "c16": "mutww", "c17": "ohlf",  "c18": "rbrmw", "c19": "whto",
}
json.dump(TRUTH, open(os.path.join(W, "ground_truth.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

rows = json.load(open(os.path.join(W, "logit_check.json"), encoding="utf-8"))

def score(pred, truth):
    if not pred:
        return False, "空结果"
    if pred == truth:
        return True, ""
    if len(pred) != len(truth):
        return False, f"长度 {len(pred)}≠{len(truth)}"
    return False, f"错位 {sum(1 for a,b in zip(pred,truth) if a!=b)} 个字符"

print("=== ResNet(ONNX) vs 真值 ===")
new_ok = old_ok = 0
for r in rows:
    cid = r["file"].replace(".png", "")
    t = TRUTH[cid]
    a1, w1 = score(r["new"], t)
    a2, w2 = score(r["old_final"], t)
    new_ok += a1
    old_ok += a2
    print(f"  {cid}  真值={t:<6} 新版={r['new']:<7}{'✓' if a1 else '✗ ' + w1:<12}"
          f" 旧版(含启发式)={r['old_final']:<7}{'✓' if a2 else '✗ ' + w2}")

n = len(rows)
print(f"\nResNet 新版(按 27 类 + blank 跳过) : {new_ok}/{n} = {new_ok/n*100:.1f}%")
print(f"ResNet 旧版(前26类 + logit启发式)  : {old_ok}/{n} = {old_ok/n*100:.1f}%")
print(f"其中 4 位样本 {sum(1 for r in rows if len(TRUTH[r['file'].replace('.png','')])==4)} 张,"
      f" 5 位样本 {sum(1 for r in rows if len(TRUTH[r['file'].replace('.png','')])==5)} 张")

json.dump({"resnet_new": new_ok, "resnet_old": old_ok, "n": n},
          open(os.path.join(W, "resnet_score.json"), "w", encoding="utf-8"), indent=1)
