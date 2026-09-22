import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import json, os, base64, math

W = _VD
OUT = _os.path.join(_REPO, "验证码识别实测报告.html")

D = json.load(open(os.path.join(W, "final_120.json"), encoding="utf-8"))
detail = D["detail"]
IDS = sorted(detail)
n = len(IDS)

COL = [("tess_now", "Tesseract 原图直喂", "你原来跑的"),
       ("tess_new", "Tesseract 调优", "新脚本兜底"),
       ("resnet_old", "ONNX 前26类+启发式", "原版后处理"),
       ("resnet_new", "ONNX 27类+blank跳过", "新脚本主路径")]

def acc(key, ids=None):
    ids = ids or IDS
    return sum(1 for i in ids if detail[i][key] == detail[i]["truth"])

def wilson(k, m, z=1.96):
    if not m: return (0, 0)
    p = k / m; d = 1 + z*z/m
    c = (p + z*z/(2*m))/d
    h = z*math.sqrt(p*(1-p)/m + z*z/(4*m*m))/d
    return (max(0, c-h), min(1, c+h))

def binom2(b, c):
    m = b + c
    if not m: return 1.0
    k = min(b, c)
    return min(1.0, 2*sum(math.comb(m, i) for i in range(k+1)) / 2**m)

def src(sid):
    p = os.path.join(W, "samples", sid + ".png")
    if not os.path.exists(p): p = os.path.join(W, "samples2", sid + ".png")
    return base64.b64encode(open(p, "rb").read()).decode()

# 卡片
cells = []
for sid in IDS:
    d = detail[sid]; t = d["truth"]
    img = src(sid)
    dots = "".join(
        f'<span class="d {"y" if d[k]==t else "x"}" title="{lab}">{lab[0]}</span>'
        for k, lab, _ in COL)
    cells.append(f'<div class="cell"><img src="data:image/png;base64,{img}" alt="{sid}">'
                 f'<div class="cid">{sid} <b>{t}</b></div><div class="dots">{dots}</div></div>')

# 统计表
rows = "".join(
    f'<tr><td>{desc}<div class="mut">{lab}</div></td>'
    f'<td class="num">{acc(k)}/{n}</td>'
    f'<td class="num {"good" if acc(k)/n>=0.9 else ("warn2" if acc(k)/n>=0.8 else "poor")}">{acc(k)/n*100:.1f}%</td>'
    f'<td class="num mut">[{wilson(acc(k),n)[0]*100:.1f}%, {wilson(acc(k),n)[1]*100:.1f}%]</td></tr>'
    for k, lab, desc in reversed(COL))

i4 = [i for i in IDS if len(detail[i]["truth"]) == 4]
i5 = [i for i in IDS if len(detail[i]["truth"]) == 5]
len_rows = "".join(
    f'<tr><td>{desc}</td><td class="num">{acc(k,i4)}/{len(i4)} ({acc(k,i4)/len(i4)*100:.0f}%)</td>'
    f'<td class="num">{acc(k,i5)}/{len(i5)} ({acc(k,i5)/len(i5)*100:.0f}%)</td></tr>'
    for k, lab, desc in reversed(COL))

def mcnemar(a, b):
    ao = sum(1 for i in IDS if detail[i][a] == detail[i]["truth"] and detail[i][b] != detail[i]["truth"])
    bo = sum(1 for i in IDS if detail[i][a] != detail[i]["truth"] and detail[i][b] == detail[i]["truth"])
    both = sum(1 for i in IDS if detail[i][a] != detail[i]["truth"] and detail[i][b] != detail[i]["truth"])
    return ao, bo, both, binom2(bo, ao)

pairs = [("resnet_new", "tess_now"), ("resnet_new", "tess_new"), ("resnet_new", "resnet_old")]
lab_of = {k: desc for k, l, desc in COL}
mc = ""
for a, b in pairs:
    ao, bo, both, p = mcnemar(a, b)
    sig = "显著" if p < 0.05 else "不显著"
    cls = "ok" if p < 0.05 else "warn2"
    mc += (f'<tr><td>{lab_of[a]}<div class="mut">vs {lab_of[b]}</div></td>'
           f'<td class="num">{ao}</td><td class="num">{bo}</td><td class="num">{both}</td>'
           f'<td class="num"><span class="{cls}">p = {p:.4f} · {sig}</span></td></tr>')

err_html = ""
for k, labl, desc in reversed(COL):
    ws = [(i, detail[i]["truth"], detail[i][k]) for i in IDS if detail[i][k] != detail[i]["truth"]]
    items = "".join(f'<li><code>{i}</code> 真值 <b>{t}</b> → 识别 <span class="bad">{v or "∅"}</span></li>' for i, t, v in ws)
    err_html += f'<details><summary>{desc} — 错 {len(ws)} 张</summary><ul class="errs">{items}</ul></details>'

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>jAccount 验证码识别 · 120 张实测报告</title>
<style>
body{{font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;background:#f6f7f9;color:#1f2328;margin:0;padding:32px 36px;line-height:1.65;font-size:14px}}
h1{{font-size:21px;margin:0 0 6px;font-weight:600}}
h2{{font-size:15px;margin:34px 0 12px;font-weight:600;padding-bottom:7px;border-bottom:1px solid #e3e6ea}}
.sub{{color:#656d76;font-size:13px;margin-bottom:22px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}}
.stat{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px 18px}}
.stat .v{{font-size:25px;font-weight:600;line-height:1.15}} .stat .l{{color:#656d76;font-size:12px;margin-top:3px}}
.stat .s{{color:#8c959f;font-size:11px}}
.stat.hi .v{{color:#1a7f37}} .stat.mid .v{{color:#9a6700}} .stat.lo .v{{color:#cf222e}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden;font-size:13px}}
th,td{{padding:8px 13px;text-align:left;border-bottom:1px solid #eef0f2}}
th{{background:#f6f8fa;font-weight:500;color:#57606a;font-size:12px}}
tr:last-child td{{border-bottom:none}}
td.num{{text-align:right;font-family:ui-monospace,Consolas,monospace;white-space:nowrap}}
.mut{{color:#8c959f;font-size:11.5px;font-family:inherit}}
.good{{color:#1a7f37;font-weight:600}} .warn2{{color:#9a6700;font-weight:600}} .poor{{color:#cf222e;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px}}
.cell{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:7px;text-align:center}}
.cell img{{width:100%;border-radius:4px;background:#fff}}
.cid{{font-size:11px;color:#8c959f;margin-top:5px;font-family:ui-monospace,Consolas,monospace}}
.cid b{{color:#1f2328;font-size:12.5px;letter-spacing:.5px}}
.dots{{display:flex;gap:3px;justify-content:center;margin-top:4px}}
.d{{width:15px;height:15px;line-height:15px;border-radius:4px;font-size:9.5px;font-weight:600;color:#fff;font-family:ui-monospace,monospace}}
.d.y{{background:#1a7f37}} .d.x{{background:#e5534b}}
.note{{margin-top:26px;padding:15px 18px;background:#fff;border:1px solid #e3e6ea;border-left:3px solid #0969da;border-radius:8px;font-size:13px}}
.note.warn{{border-left-color:#d4920b}} .note li{{margin:6px 0}}
code{{background:#f0f2f4;padding:1px 5px;border-radius:4px;font-size:12px;font-family:ui-monospace,Consolas,monospace}}
details{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:9px 14px;margin-bottom:8px;font-size:13px}}
summary{{cursor:pointer;font-weight:500}} .errs{{margin:8px 0 2px;padding-left:20px;color:#3d444d}}
.bad{{color:#cf222e;font-family:ui-monospace,monospace}}
.legend{{color:#656d76;font-size:12px;margin:10px 0 16px}}
</style></head><body>

<h1>jAccount 验证码识别 · 120 张实测报告</h1>
<div class="sub">{n} 张真实抓取的验证码（110×40，4 位 {len(i4)} 张 / 5 位 {len(i5)} 张）·
真值全部由人工逐张辨认，辨认时未参考任何模型输出 · 四个引擎跑的是同一批图</div>

<div class="stats">
  <div class="stat lo"><div class="v">{acc('tess_now')/n*100:.1f}%</div><div class="l">Tesseract 原图直喂</div><div class="s">你原来跑的 {acc('tess_now')}/{n}</div></div>
  <div class="stat mid"><div class="v">{acc('tess_new')/n*100:.1f}%</div><div class="l">Tesseract 调优</div><div class="s">新脚本兜底 {acc('tess_new')}/{n}</div></div>
  <div class="stat hi"><div class="v">{acc('resnet_old')/n*100:.1f}%</div><div class="l">ONNX 原版后处理</div><div class="s">{acc('resnet_old')}/{n}</div></div>
  <div class="stat hi"><div class="v">{acc('resnet_new')/n*100:.1f}%</div><div class="l">新脚本主路径</div><div class="s">{acc('resnet_new')}/{n}</div></div>
</div>

<h2>准确率与 95% 置信区间</h2>
<table><tr><th>引擎</th><th>命中</th><th>准确率</th><th>95% Wilson 区间</th></tr>{rows}</table>

<h2>按验证码长度拆分</h2>
<table><tr><th>引擎</th><th>4 位</th><th>5 位</th></tr>{len_rows}</table>

<h2>配对显著性检验（McNemar 精确检验）</h2>
<div class="sub">同一批样本上的成对结果，只看两个引擎判断不一致的样本。</div>
<table><tr><th>比较</th><th>前者对/后者错</th><th>后者对/前者错</th><th>都错</th><th>结论</th></tr>{mc}</table>

<div class="note warn">
<b>这份报告推翻了我此前两个说法，先说清楚：</b>
<ul>
<li><b>Tesseract「调优」其实没用。</b>20 张时看起来是 80% → 85%，扩到 120 张后两者<b>完全打平，都是 89/120 = 74.2%</b>。
之前的提升是小样本噪声。这项改动真正的价值只剩<b>速度</b>：worker 复用把单张 169ms 压到 12ms（约 13.6 倍）。</li>
<li><b>「27 类 + blank 跳过」也不是准确率提升。</b>120 张上原版启发式 114/120，新版 116/120，
McNemar p = 0.5，<b>不显著</b>。它修的是一个确定性缺陷（原版会把个别 5 位码误截成 4 位），
方向对，但不能算作可测量的准确率增益。</li>
</ul>
<b>唯一统计显著的提升只有一个：把一直没有被调用的 ResNet 路径接上。</b>
74.2% → 96.7%，前者对后者错 29 例、后者对前者错 2 例，p &lt; 0.0001。
</div>

<h2>逐张核对（全部 {n} 张）</h2>
<div class="legend">小方块依次代表四个引擎：<span class="d y" style="display:inline-block">T</span> Tesseract原图 ·
<span class="d y" style="display:inline-block">T</span> Tesseract调优 ·
<span class="d y" style="display:inline-block">O</span> ONNX原版 ·
<span class="d y" style="display:inline-block">O</span> ONNX新版。绿色=与真值一致，红色=错误。</div>
<div class="grid">{''.join(cells)}</div>

<h2>错例明细</h2>
{err_html}

<div class="note">
<b>结论</b>
<ul>
<li>ResNet 路径 96.7%（95% CI 91.7%–98.7%）与上游项目标称的 98~99% 区间相容，未发现异常。</li>
<li>新版主路径的 4 例错误中有 3 例是 <code>o↔x</code>、<code>r→u</code> 这类形近字混淆，
且 softmax 置信度仍很高 —— 靠置信度阈值无法筛出，这一点已在脚本注释里写明。</li>
<li>120 张样本里单张 = 0.83%，比之前 20 张时的 5% 精细得多，因此上面的显著性结论可信。</li>
<li>完整逐样本数据见 <code>vm_dump/final_120.json</code>，真值见 <code>vm_dump/ground_truth_all.json</code>。</li>
</ul>
</div>

</body></html>"""

open(OUT, "w", encoding="utf-8").write(html)
print("report ->", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")
for k, l, d in reversed(COL):
    print(f"  {d:<16} {acc(k)}/{n} = {acc(k)/n*100:.1f}%")
