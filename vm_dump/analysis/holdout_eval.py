import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""1) 写入留出集真值  2) 修正 n062 错标  3) 用字形几何特征系统审计 i/j 标注
   4) 在留出集上评估 基线 vs TTA"""
import os, json, math, warnings
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)
W = _VD

HOLDOUT = {
"h000":"ekttd","h001":"aqjuk","h002":"mpgt","h003":"orpk","h004":"lfbt","h005":"iwyvl",
"h006":"axcid","h007":"ymge","h008":"nxpk","h009":"ptrnv","h010":"ziopk","h011":"apup",
"h012":"jixn","h013":"rnxmj","h014":"iyub","h015":"xotw","h016":"dktj","h017":"xyes",
"h018":"ingo","h019":"snzut","h020":"lgat","h021":"xpyx","h022":"dcrdf","h023":"ybubw",
"h024":"behio","h025":"dtvnj","h026":"vtmr","h027":"hyhdr","h028":"ifee","h029":"yhvi",
"h030":"ijtwg","h031":"ydmd","h032":"eweo","h033":"yxfxk","h034":"mrcbd","h035":"qqyhz",
"h036":"iynn","h037":"rfajp","h038":"njxla","h039":"kklnq","h040":"ifjgz","h041":"cwgu",
"h042":"cnich","h043":"fkkrr","h044":"eedw","h045":"arzx","h046":"vved","h047":"fbvd",
"h048":"ozqpm","h049":"hnnlj","h050":"knvr","h051":"igzc","h052":"hyysl","h053":"osefq",
"h054":"enyg","h055":"tjmut","h056":"ttjq","h057":"lyuj","h058":"bzbw","h059":"oqemm",
"h060":"bcis","h061":"tpol","h062":"ayffi","h063":"wzty","h064":"zmkcz","h065":"wask",
"h066":"wvojj","h067":"daxeo","h068":"rbel","h069":"nkzxf","h070":"kucdp","h071":"hfgq",
"h072":"prpx","h073":"sgnyx","h074":"ffjl","h075":"agfx","h076":"vgji","h077":"umdu",
"h078":"zyvo","h079":"ytvgb","h080":"hurc","h081":"nnstw","h082":"viku","h083":"rkfsf",
"h084":"xbjs","h085":"bibub","h086":"yywwi","h087":"myuy","h088":"qsda","h089":"szrgq",
"h090":"vajbg","h091":"tkbp","h092":"ekznr","h093":"wylqz","h094":"hyjl","h095":"izha",
"h096":"lwnmq","h097":"iwiw","h098":"qxfm","h099":"yfpm",
}

gt = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))
fixed = []
if gt.get("n062") != "jyrv":
    fixed.append(f"n062: {gt.get('n062')} -> jyrv")
    gt["n062"] = "jyrv"
gt.update(HOLDOUT)
json.dump(gt, open(os.path.join(W, "ground_truth_all.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
json.dump(HOLDOUT, open(os.path.join(W, "ground_truth_holdout.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("修正的错标:", fixed or "无")

TUNE = [i for i in gt if i[0] in "cn"] and sorted([i for i in gt if i.startswith(("c", "n")) and not i.startswith("n1") or i.startswith("n0") or i.startswith("n1")])
TUNE = sorted([i for i in gt if not i.startswith("h")])
HOLD = sorted([i for i in gt if i.startswith("h")])
print(f"调参集 {len(TUNE)} 张 / 留出集 {len(HOLD)} 张")

# ---------------- 字形几何审计：i 高约 15px，j 高约 19px ----------------
def path_of(sid):
    for sub in ("samples", "samples2", "holdout"):
        p = os.path.join(W, sub, sid + ".png")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(sid)


def glyph_shape(sid, idx):
    g = np.asarray(Image.open(path_of(sid)).convert("L"))
    ink = g < 156
    cols = ink.any(axis=0)
    segs, st = [], None
    for c, v in enumerate(cols):
        if v and st is None:
            st = c
        elif not v and st is not None:
            if c - st >= 2:
                segs.append((st, c))
            st = None
    if st is not None and len(cols) - st >= 2:
        segs.append((st, len(cols)))
    if idx >= len(segs):
        return None
    x0, x1 = segs[idx]
    ys = np.where(ink[:, x0:x1].any(axis=1))[0]
    top, bot = int(ys.min()), int(ys.max())
    # 顶部独立像素块 = 字母上方的点
    row_ink = ink[:, x0:x1].any(axis=1)
    has_dot = False
    gap_rows = [y for y in range(top, bot + 1) if not row_ink[y]]
    if gap_rows and gap_rows[0] - top <= 5:
        has_dot = True
    return {"w": x1 - x0, "h": bot - top + 1, "top": top, "bot": bot, "dot": has_dot}


print("\n=== 字形审计：标注为 i 或 j 的字符，其高度是否符合几何特征 ===")
suspects = []
for sid in sorted(gt):
    for k, ch in enumerate(gt[sid]):
        if ch not in "ij":
            continue
        sh = glyph_shape(sid, k)
        if not sh:
            continue
        # 基线 = 该样本所有字符的中位底边
        if ch == "i" and sh["h"] >= 18:
            suspects.append((sid, k, ch, sh["h"], "标了 i 但高度像 j"))
        if ch == "j" and sh["h"] <= 16:
            suspects.append((sid, k, ch, sh["h"], "标了 j 但高度像 i"))
if suspects:
    for s in suspects:
        print(f"  ⚠ {s[0]} 第{s[1]}位 标注={s[2]} 高{s[3]}px  {s[4]}")
else:
    print("  未发现 i/j 标注与字形高度矛盾的情况")
print(f"  共检查 {sum(1 for s in gt for c in gt[s] if c in 'ij')} 个 i/j 字符")

# ---------------- ResNet: 基线 vs TTA ----------------
sess = rt.InferenceSession(os.path.join(W, "nn_model.onnx"), providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name


def base_input(sid, thr=156):
    g = np.asarray(Image.open(path_of(sid)).convert("L"), dtype=np.float32)
    return (g >= thr).astype(np.float32)


def shift(img, dx, dy, fill=1.0):
    out = np.full_like(img, fill)
    h, w = img.shape
    out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
        img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def decode(outs):
    txt = ""
    for t in outs:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            continue
        txt += chr(ord("a") + a)
    return txt


def run(arr):
    return sess.run(None, {IN: arr[None, None, ...]})


PRED = {"base": {}, "tta1x3": {}, "tta3x3": {}}
for sid in sorted(gt):
    b = base_input(sid)
    PRED["base"][sid] = decode(run(b))
    for name, dxs, dys in (("tta1x3", [-1, 0, 1], [0]), ("tta3x3", [-1, 0, 1], [-1, 0, 1])):
        acc = None
        for dy in dys:
            for dx in dxs:
                o = run(shift(b, dx, dy))
                if acc is None:
                    acc = [x.astype(np.float32).copy() for x in o]
                else:
                    for k2, x in enumerate(o):
                        acc[k2] += x
        PRED[name][sid] = decode(acc)

print("\n=== 准确率（已修正 n062 错标）===")
print(f"{'方案':<12}{'调参集 120':>16}{'留出集 100':>16}")


def score(name, ids):
    return sum(1 for i in ids if PRED[name][i] == gt[i])


for name in ("base", "tta1x3", "tta3x3"):
    a, b = score(name, TUNE), score(name, HOLD)
    print(f"{name:<12}{f'{a}/{len(TUNE)} ({a/len(TUNE)*100:.1f}%)':>16}"
          f"{f'{b}/{len(HOLD)} ({b/len(HOLD)*100:.1f}%)':>16}")


def mcnemar(name_a, name_b, ids):
    A = sum(1 for i in ids if PRED[name_a][i] == gt[i] and PRED[name_b][i] != gt[i])
    B = sum(1 for i in ids if PRED[name_a][i] != gt[i] and PRED[name_b][i] == gt[i])
    n = A + B
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(A, B) + 1)) / 2 ** n)
    return A, B, p


print("\n=== 留出集上 TTA 的配对检验 ===")
for name in ("tta1x3", "tta3x3"):
    A, B, p = mcnemar("base", name, HOLD)
    print(f"  base vs {name}: base对/TTA错={A}  base错/TTA对={B}  p={p:.4f} "
          f"-> {'显著' if p < 0.05 else '不显著'}")

print("\n=== 留出集上的错例 ===")
for name in ("base", "tta1x3", "tta3x3"):
    errs = [(i, gt[i], PRED[name][i]) for i in HOLD if PRED[name][i] != gt[i]]
    print(f"  {name} 错 {len(errs)} 张: {[(a,b,c) for a,b,c in errs]}")

json.dump({"pred": PRED, "gt": gt}, open(os.path.join(W, "holdout_eval.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n已写入 holdout_eval.json")
