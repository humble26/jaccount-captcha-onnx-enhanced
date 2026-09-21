"""写入 100 张新样本的人工真值（来源：blind2/sheet_*.png 肉眼辨认），并为全部 120 张
生成两种 Tesseract 输入形态：
  A = 原尺寸 110x40 JPEG（≈ canvas.toDataURL('image/jpeg')）—— 用户当前脚本的喂法
  B = 原尺寸 156 阈值二值化 PNG —— 新脚本兜底路径的喂法
"""
import json, os
from PIL import Image

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
GT2 = {
"n000":"ngfym","n001":"rxdx","n002":"ybun","n003":"bkpek","n004":"ywys","n005":"ppdgv",
"n006":"vxzq","n007":"volq","n008":"gnhri","n009":"lnwif","n010":"gzpet","n011":"ejnpk",
"n012":"ddch","n013":"czyu","n014":"dpssc","n015":"azjqc","n016":"gvmqx","n017":"hopl",
"n018":"naxc","n019":"kgji","n020":"magf","n021":"osos","n022":"dnpgk","n023":"locki",
"n024":"gdlha","n025":"fdxk","n026":"ycnjb","n027":"saqvi","n028":"rudih","n029":"odzzl",
"n030":"mrmia","n031":"lxqp","n032":"fskz","n033":"pfex","n034":"ljqwv","n035":"sokzn",
"n036":"fnsn","n037":"utih","n038":"menyj","n039":"ofts","n040":"kdqo","n041":"tkmje",
"n042":"idad","n043":"ylyo","n044":"huaa","n045":"bhbu","n046":"sqvoj","n047":"aouim",
"n048":"zclwe","n049":"qzxo","n050":"lhvq","n051":"ffryu","n052":"qhxj","n053":"hosy",
"n054":"btgs","n055":"tdiyu","n056":"pfnbb","n057":"heoog","n058":"xvhsa","n059":"hokco",
"n060":"qivcj","n061":"klham","n062":"iyrv","n063":"vjpdd","n064":"cdfs","n065":"ptouc",
"n066":"xykl","n067":"xnaw","n068":"vlpc","n069":"mhnnb","n070":"kpkbq","n071":"keail",
"n072":"tics","n073":"bpucn","n074":"rjun","n075":"drxgp","n076":"sbhwg","n077":"susi",
"n078":"xsxh","n079":"kvxfl","n080":"zmbrw","n081":"jezd","n082":"avdk","n083":"jlzdd",
"n084":"zryw","n085":"zugpb","n086":"ugxq","n087":"dgsb","n088":"mwoc","n089":"xeys",
"n090":"ggamk","n091":"qjxkp","n092":"qrvsp","n093":"mpzw","n094":"lelcj","n095":"bjqf",
"n096":"luqxp","n097":"vsziu","n098":"ttgq","n099":"ikrg",
}

gt1 = json.load(open(os.path.join(W, "ground_truth.json"), encoding="utf-8"))
allgt = dict(gt1)
allgt.update(GT2)
json.dump(allgt, open(os.path.join(W, "ground_truth_all.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# 真值长度分布核对
from collections import Counter
print("真值长度分布:", dict(Counter(len(v) for v in allgt.values())), " 共", len(allgt), "张")
bad = [k for k, v in allgt.items() if len(v) not in (4, 5) or not v.isalpha() or not v.islower()]
print("异常真值:", bad)

A = os.path.join(W, "all_A_original"); B = os.path.join(W, "all_B_bin156")
os.makedirs(A, exist_ok=True); os.makedirs(B, exist_ok=True)
LUT = [0] * 156 + [255] * 100
n = 0
for sid in allgt:
    src = os.path.join(W, "samples", sid + ".png")
    if not os.path.exists(src):
        src = os.path.join(W, "samples2", sid + ".png")
    im = Image.open(src).convert("L")
    im.convert("RGB").save(os.path.join(A, sid + ".jpg"), quality=92, subsampling=2)
    im.point(LUT, "L").save(os.path.join(B, sid + ".png"))
    n += 1
print(f"A 组 {len(os.listdir(A))} 张 / B 组 {len(os.listdir(B))} 张（应为 {n}）")
