import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import json, os, base64

W = _VD
OUT = _os.path.join(_REPO, "验证码识别实测报告.html")

truth = json.load(open(os.path.join(W, "ground_truth.json"), encoding="utf-8"))
logit = {r["file"].replace(".png", ""): r for r in json.load(open(os.path.join(W, "logit_check.json"), encoding="utf-8"))}
tessA = {r["id"]: r for r in json.load(open(os.path.join(W, "tesseract_result.json"), encoding="utf-8"))["A"]}
tessF = json.load(open(os.path.join(W, "tess_final_result.json"), encoding="utf-8"))
tessFm = {r["id"]: r for r in tessF["rows"]}
ablation = json.load(open(os.path.join(W, "ablation_result.json"), encoding="utf-8"))

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()

ids = sorted(truth.keys())
n = len(ids)

resnet_ok = sum(1 for i in ids if logit[i]["new"] == truth[i])
oldonnx_ok = sum(1 for i in ids if logit[i]["old_final"] == truth[i])
tessnow_ok = sum(1 for i in ids if tessA[i]["pred"] == truth[i])
tessnew_ok = tessF["hit"]

# 按“用户现在实际跑的引擎”排序：先列错的，方便核对
ids_sorted = sorted(ids, key=lambda i: (tessA[i]["pred"] == truth[i], logit[i]["new"] == truth[i]))

cards = []
for i in ids_sorted:
    t = truth[i]
    r_new = logit[i]["new"]
    r_old = logit[i]["old_final"]
    tn = tessA[i]["pred"]
    tf = tessFm[i]["pred"]
    img = b64(os.path.join(W, "samples", f"{i}.png"))
    def cell(v, ok):
        return f'<span class="{"ok" if ok else "bad"}">{v or "∅"}</span>'
    cards.append(f"""<figure>
<img src="data:image/png;base64,{img}" alt="{i}">
<figcaption>
<div class="truth">真值 <b>{t}</b></div>
<div class="row"><span class="lbl">ResNet 新版</span>{cell(r_new, r_new==t)}</div>
<div class="row"><span class="lbl">ResNet 原版</span>{cell(r_old, r_old==t)}</div>
<div class="row"><span class="lbl">Tess 你现在</span>{cell(tn, tn==t)}</div>
<div class="row"><span class="lbl">Tess 新版</span>{cell(tf, tf==t)}</div>
</figcaption></figure>""")

ab_rows = "".join(
    f"<tr><td>{r['params']}</td><td>{r['variant']}</td>"
    f"<td class='num'>{r['hit']}/{r['n']}</td>"
    f"<td class='num {'good' if r['acc']>=0.85 else ('mid' if r['acc']>=0.8 else 'poor')}'>"
    f"{r['acc']*100:.0f}%</td><td class='num'>{r['avgMs']:.0f} ms</td></tr>"
    for r in sorted(ablation, key=lambda x: -x["acc"]))

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>jAccount 验证码识别 · 实测报告</title>
<style>
body{{font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;background:#f6f7f9;color:#1f2328;margin:0;padding:32px 36px;line-height:1.65}}
h1{{font-size:21px;margin:0 0 6px;font-weight:600}}
h2{{font-size:15px;margin:34px 0 12px;font-weight:600;padding-bottom:7px;border-bottom:1px solid #e3e6ea}}
.sub{{color:#656d76;font-size:13px;margin-bottom:22px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.stat{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px 18px}}
.stat .v{{font-size:25px;font-weight:600;line-height:1.15}}
.stat .l{{color:#656d76;font-size:12px;margin-top:2px}}
.stat.hi .v{{color:#1a7f37}} .stat.lo .v{{color:#b35900}} .stat.base .v{{color:#6e7781}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));gap:14px}}
figure{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;margin:0;padding:10px}}
figure img{{width:100%;border:1px solid #eef0f2;border-radius:6px;display:block;background:#fff}}
figcaption{{font-size:12px;margin-top:9px;line-height:1.85}}
.truth{{font-size:13px;margin-bottom:5px;color:#1f2328}} .truth b{{font-family:ui-monospace,Consolas,monospace;font-size:15px;letter-spacing:1px}}
.row{{display:flex;justify-content:space-between;gap:8px}}
.lbl{{color:#8c959f;font-size:11.5px}}
.ok,.bad{{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;font-weight:600}}
.ok{{color:#1a7f37}} .bad{{color:#cf222e}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden;font-size:12.5px}}
th,td{{padding:7px 12px;text-align:left;border-bottom:1px solid #eef0f2}}
th{{background:#f6f8fa;font-weight:500;color:#57606a}}
tr:last-child td{{border-bottom:none}}
td.num{{text-align:right;font-family:ui-monospace,Consolas,monospace}}
.good{{color:#1a7f37;font-weight:600}} .mid{{color:#9a6700;font-weight:600}} .poor{{color:#cf222e;font-weight:600}}
.note{{margin-top:26px;padding:15px 18px;background:#fff;border:1px solid #e3e6ea;border-left:3px solid #0969da;border-radius:8px;font-size:13px}}
.note.warn{{border-left-color:#d4920b}}
.note li{{margin:5px 0}}
code{{background:#f0f2f4;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body>

<h1>jAccount 验证码识别 · 实测报告</h1>
<div class="sub">{n} 张真实抓取的 jAccount 验证码（110×40，含 10 张 4 位 / 10 张 5 位）·
真值由人工逐张辨认 · 所有引擎跑的是同一组图</div>

<div class="stats">
  <div class="stat base"><div class="v">{tessnow_ok}/{n}</div><div class="l">你现在的脚本（Tesseract 原图直喂）</div></div>
  <div class="stat hi"><div class="v">{resnet_ok}/{n}</div><div class="l">新版主路径（ResNet ONNX）</div></div>
  <div class="stat lo"><div class="v">{oldonnx_ok}/{n}</div><div class="l">原版 ONNX 后处理（含启发式）</div></div>
  <div class="stat hi"><div class="v">{tessnew_ok}/{n}</div><div class="l">新版兜底（Tesseract 调优后）</div></div>
</div>

<h2>逐张核对</h2>
<div class="sub">按"你现在的脚本是否识别正确"排序 —— 错的排在最前面，方便你直接核对图片。</div>
<div class="grid">{''.join(cards)}</div>

<h2>Tesseract 参数消融（6 种图像预处理 × 4 种参数集）</h2>
<div class="sub">worker 复用、同一进程连续跑，因此耗时可比。这里推翻了一个常见直觉：<b>堆预处理没有用，PSM 选错才是灾难。</b></div>
<table>
<tr><th>参数集</th><th>图像预处理</th><th>命中</th><th>准确率</th><th>单张耗时</th></tr>
{ab_rows}
</table>

<div class="note warn">
<b>结论与保留意见</b>
<ul>
<li>真正带来提升的不是 Tesseract 调参（80% → 85%，边际），而是<b>把一直没被调用的 ResNet 路径接上</b>（80% → 95%）。</li>
<li>Tesseract 在本类验证码上约 <b>85% 就见顶</b>：换任何预处理组合都上不去；而 <code>PSM 8</code> 会把准确率打到 45%，务必避开。</li>
<li>样本量只有 {n} 张，单个样本即 5%。ResNet 唯一那张错例是 <code>riixo</code> 被读成 <code>riioo</code>（x→o）。
上游项目标称 ResNet 约 98~99%、Tesseract 约 90%，因此 <b>95% 与 85% 的差距方向可信，但具体数值受样本量限制，不宜当成精确值</b>。</li>
<li>原版 ONNX 后处理其实并非不可用（{oldonnx_ok}/{n}）。它的失败方向是把 5 位码误截成 4 位（<code>kwwkc → kwwk</code>），
这正是它需要用置信度阈值"猜"4/5 位的代价。</li>
</ul>
</div>

</body></html>"""

open(OUT, "w", encoding="utf-8").write(html)
print("report ->", OUT)
print(f"你现在: {tessnow_ok}/{n} | ResNet新版: {resnet_ok}/{n} | 原版ONNX: {oldonnx_ok}/{n} | Tesseract调优: {tessnew_ok}/{n}")
