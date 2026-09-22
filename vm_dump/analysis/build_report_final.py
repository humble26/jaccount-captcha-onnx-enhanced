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

gt = json.load(open(os.path.join(W, "ground_truth_all.json"), encoding="utf-8"))
EV = json.load(open(os.path.join(W, "holdout_eval.json"), encoding="utf-8"))
PRED = EV["pred"]
t120 = json.load(open(os.path.join(W, "tess_120.json"), encoding="utf-8"))
th = json.load(open(os.path.join(W, "tess_holdout.json"), encoding="utf-8"))
conf = {r["id"]: r for r in json.load(open(os.path.join(W, "confidence_analysis.json"), encoding="utf-8"))}
tune = json.load(open(os.path.join(W, "tune_result.json"), encoding="utf-8"))["threshold"]

TUNE = sorted(i for i in gt if not i.startswith("h"))
HOLD = sorted(i for i in gt if i.startswith("h"))

# Tesseract 在两组上的预测
TE = {}
for i in TUNE:
    TE.setdefault("tess_now", {})[i] = t120["A"].get(i, {}).get("pred", "")
    TE.setdefault("tess_new", {})[i] = t120["B"].get(i, {}).get("pred", "")
for i in HOLD:
    TE.setdefault("tess_now", {})[i] = th["A"].get(i, {}).get("pred", "")
    TE.setdefault("tess_new", {})[i] = th["B"].get(i, {}).get("pred", "")

ENG = [("tess_now", "Tesseract 原图直喂", "你原来跑的"),
       ("tess_new", "Tesseract + 白名单 + PSM7", "新脚本兜底"),
       ("resnet_old", "ONNX 前26类 + 启发式", "原版后处理"),
       ("resnet_new", "ONNX 27类 + blank 跳过", "新脚本主路径")]

def pred_of(key, sid):
    if key in ("tess_now", "tess_new"):
        return TE[key].get(sid, "")
    return PRED["base"][sid] if key == "resnet_new" else None

# resnet_old 只在调参集上算过；用 logit_check 的数据补
old_pred = {r["file"].replace(".png", ""): r["old_final"] for r in
            json.load(open(os.path.join(W, "logit_check.json"), encoding="utf-8"))}
old_pred2 = {r["id"]: r["resnet_old"] for r in
             json.load(open(os.path.join(W, "samples2_resnet.json"), encoding="utf-8"))}
old_pred.update(old_pred2)

def get(key, sid):
    if key == "resnet_old":
        return old_pred.get(sid, "")
    return pred_of(key, sid)

def acc(key, ids):
    return sum(1 for i in ids if get(key, i) == gt[i])

def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))

def binom2(b, c):
    n = b + c
    if not n: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def mc(a, b, ids):
    ao = sum(1 for i in ids if get(a, i) == gt[i] and get(b, i) != gt[i])
    bo = sum(1 for i in ids if get(a, i) != gt[i] and get(b, i) == gt[i])
    return ao, bo, binom2(bo, ao)

rows = ""
for k, lab, desc in reversed(ENG):
    a1, r1 = acc(k, TUNE), acc(k, HOLD)
    lo1, hi1 = wilson(a1, len(TUNE))
    rows += (f'<tr><td>{desc}<div class="mut">{lab}</div></td>'
             f'<td class="num">{a1}/{len(TUNE)}</td>'
             f'<td class="num {"good" if a1/len(TUNE)>=.9 else ("w2" if a1/len(TUNE)>=.8 else "poor")}">{a1/len(TUNE)*100:.1f}%</td>'
             f'<td class="num mut">[{lo1*100:.1f}, {hi1*100:.1f}]</td>'
             f'<td class="num">{r1}/{len(HOLD) if k.startswith("resnet") or k.startswith("tess") else len(HOLD)}</td>'
             f'<td class="num {"good" if r1/len(HOLD)>=.9 else ("w2" if r1/len(HOLD)>=.8 else "poor")}">{r1/len(HOLD)*100:.1f}%</td></tr>')

mc_rows = ""
for a, b, tag in (("resnet_new", "tess_now", "接上 ResNet vs 你原来跑的"),
                  ("resnet_new", "tess_new", "接上 ResNet vs Tesseract 调优"),
                  ("resnet_old", "resnet_new", "原版后处理 vs 新版后处理")):
    l1 = dict((k, d) for k, _, d in ENG)[a]; l2 = dict((k, d) for k, _, d in ENG)[b]
    ao, bo, p = mc(a, b, TUNE)
    cls = "good" if p < 0.05 else "w2"
    mc_rows += (f'<tr><td>{tag}</td><td class="num">{ao}</td><td class="num">{bo}</td>'
                f'<td class="num"><span class="{cls}">p = {p:.4f}</span></td>'
                f'<td>{"显著" if p < 0.05 else "不显著（差异在噪声范围内）"}</td></tr>')

def thr_row(item):
    t, v = int(item[0]), item[1]
    note = "最优平台区" if 145 <= t <= 175 else ("过暗，笔画粘连" if t < 145 else "过亮，笔画断裂")
    c = "good" if v >= 116 else ("w2" if v >= 113 else "poor")
    return (f'<tr><td class="num">{t}</td><td class="num">{v}/120</td>'
            f'<td class="num {c}">{v/120*100:.1f}%</td><td>{note}</td></tr>')

thr_rows = "".join(thr_row(x) for x in sorted(tune.items(), key=lambda x: int(x[0])))

def cls(r):
    return "y" if r["ok"] else "x"
crows = ""
for r in sorted(conf.values(), key=lambda x: x["minP"]):
    crows += (f'<tr><td><code>{r["id"]}</code></td><td>{r["truth"]}</td><td>{r["pred"]}</td>'
              f'<td class="num">{r["minP"]*100:.2f}%</td>'
              f'<td><span class="{"good" if r["ok"] else "poor"}">{"正确" if r["ok"] else "错误"}</span></td></tr>')

# 错例图
def src(sid):
    for sub in ("samples", "samples2", "holdout"):
        p = os.path.join(W, sub, sid + ".png")
        if os.path.exists(p):
            return base64.b64encode(open(p, "rb").read()).decode()
    return ""

err_ids = [i for i in TUNE + HOLD if get("resnet_new", i) != gt[i]]
err_cards = "".join(
    f'<figure><img src="data:image/png;base64,{src(i)}" alt="{i}">'
    f'<figcaption>{i}<br><b>{gt[i]}</b><br><span class="bad">{get("resnet_new", i) or "∅"}</span></figcaption></figure>'
    for i in err_ids)

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>jAccount 验证码识别 · 220 张实测与提升空间评估</title>
<style>
body{{font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;background:#f6f7f9;color:#1f2328;margin:0;padding:32px 36px;line-height:1.65;font-size:14px}}
h1{{font-size:21px;margin:0 0 6px;font-weight:600}}
h2{{font-size:15px;margin:36px 0 12px;font-weight:600;padding-bottom:7px;border-bottom:1px solid #e3e6ea}}
.sub{{color:#656d76;font-size:13px;margin-bottom:22px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.stat{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px 18px}}
.stat .v{{font-size:25px;font-weight:600;line-height:1.15}} .stat .l{{color:#656d76;font-size:12px;margin-top:3px}}
.stat .s{{color:#8c959f;font-size:11px}}
.stat.hi .v{{color:#1a7f37}} .stat.mid .v{{color:#9a6700}} .stat.lo .v{{color:#cf222e}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden;font-size:13px}}
th,td{{padding:8px 13px;text-align:left;border-bottom:1px solid #eef0f2}}
th{{background:#f6f8fa;font-weight:500;color:#57606a;font-size:12px}}
tr:last-child td{{border-bottom:none}}
td.num{{text-align:right;font-family:ui-monospace,Consolas,monospace;white-space:nowrap}}
.mut{{color:#8c959f;font-size:11.5px}}
.good{{color:#1a7f37;font-weight:600}} .w2{{color:#9a6700;font-weight:600}} .poor{{color:#cf222e;font-weight:600}}
.note{{margin-top:24px;padding:15px 18px;background:#fff;border:1px solid #e3e6ea;border-left:3px solid #0969da;border-radius:8px;font-size:13px}}
.note.warn{{border-left-color:#d4920b}} .note.ok{{border-left-color:#1a7f37}}
.note li{{margin:6px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:14px}}
figure{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;margin:0;padding:9px;text-align:center}}
figure img{{width:100%;border-radius:5px;background:#fff}}
figcaption{{font-size:11.5px;color:#8c959f;margin-top:6px;font-family:ui-monospace,Consolas,monospace}}
figcaption b{{color:#1f2328;font-size:13px}} .bad{{color:#cf222e}}
code{{background:#f0f2f4;padding:1px 5px;border-radius:4px;font-size:12px;font-family:ui-monospace,Consolas,monospace}}
</style></head><body>

<h1>jAccount 验证码识别 · 220 张实测与「还能不能更好」评估</h1>
<div class="sub">
调参集 {len(TUNE)} 张 + <b>独立留出集 {len(HOLD)} 张</b>（留出集是单独采集、不参与任何调参决策的，
用来防止自欺）· 真值全部人工逐张辨认 · 所有引擎跑同一批图
</div>

<div class="stats">
  <div class="stat lo"><div class="v">{acc("tess_now", TUNE)/len(TUNE)*100:.1f}%</div><div class="l">Tesseract 原图直喂</div><div class="s">你原来跑的</div></div>
  <div class="stat mid"><div class="v">{acc("tess_new", TUNE)/len(TUNE)*100:.1f}%</div><div class="l">Tesseract 调优</div><div class="s">准确率打平，只快 13.6 倍</div></div>
  <div class="stat hi"><div class="v">{acc("resnet_new", TUNE)/len(TUNE)*100:.1f}% / {acc("resnet_new", HOLD)/len(HOLD)*100:.1f}%</div><div class="l">ResNet 新版主路径</div><div class="s">调参集 / 留出集</div></div>
  <div class="stat hi"><div class="v">2.2s</div><div class="l">冷启动总下载耗时</div><div class="s">实测 jsDelivr 10.4MB/s</div></div>
</div>

<h2>准确率与置信区间</h2>
<table>
<tr><th>引擎</th><th>调参集命中</th><th>准确率</th><th>95% Wilson</th><th>留出集命中</th><th>准确率</th></tr>
{rows}
</table>

<h2>配对显著性检验（McNemar 精确检验，调参集）</h2>
<table>
<tr><th>比较</th><th>前者对/后者错</th><th>后者对/前者错</th><th>p 值</th><th>结论</th></tr>
{mc_rows}
</table>

<div class="note ok">
<b>唯一统计显著的提升只有一条：把一直没被调用的 ResNet 路径接上。</b>
74.2% → {acc("resnet_new", TUNE)/len(TUNE)*100:.1f}%，29 比 2，p &lt; 0.0001；
并且在完全独立的 {len(HOLD)} 张留出集上复现为 {acc("resnet_new", HOLD)/len(HOLD)*100:.1f}%。
</div>

<h2>本轮修正的一处标注错误</h2>
<div class="note warn">
上一版报告里我把 <code>n062</code> 的首字符标成了 <code>i</code>，实际是 <code>j</code> —— <b>模型读对了，是我错了</b>。
判定方法：把所有 i / j 字形放大对比后发现，这个小写字母集里 <b>i 的字形高度固定约 15px</b>（点在短竖上、无下延），
<b>j 固定约 19px</b>（有下延）。n062 首字符高 19px，是 j。
<br>随后用这个几何特征<b>系统复核了全部 220 张里的 83 个 i/j 字符，未再发现矛盾</b>。
修正后 ResNet 主路径从 116/120 提升到 117/120。
</div>

<h2>「准确度还能不能提升」—— 逐个测掉</h2>
<table>
<tr><th>方案</th><th>结果</th><th>结论</th></tr>
<tr><td>Tesseract 调优（字母白名单 + PSM7 + 复用 worker）</td><td class="num w2">220 张：74.5% → 76.8%</td>
<td>McNemar <b>p = 0.52，不显著</b>；且两个集合方向不一致（调参集 89 vs 89 打平，留出集 75 vs 79 +4）。
<b>准确率收益不成立</b>，这项改动的真实收益只有速度：单张 169ms → 12ms（13.6 倍）</td></tr>
<tr><td>二值化阈值扫描</td><td class="num">145~175 全部 116/120</td><td>一片平台，官方 156 已在最优区，<b>无可调空间</b></td></tr>
<tr><td>不二值化，直接喂灰度</td><td class="num poor">0/120 (0%)</td><td>模型要求硬 0/1 输入，<b>预处理一步都不能省</b></td></tr>
<tr><td>TTA：平移 ±1px 后 logits 平均</td><td class="num w2">调参集 116/120；留出集 98/100</td><td>调参集上像是 +1 例，但留出集上 <b>0 对不一致、完全没变</b>，调参集反而 -1 例 —— <b>已否决</b></td></tr>
<tr><td>换成纯 JS 前向实现（省掉 10.4MB wasm）</td><td class="num">省一次 2.2 秒</td><td>代价是每次识别慢十倍以上；冷启动实测只有 2.2 秒且之后长期缓存命中，<b>不划算</b></td></tr>
</table>

<div class="note">
<b>阈值扫描明细</b>（120 张调参集）
<table style="margin-top:10px">
<tr><th>阈值</th><th>命中</th><th>准确率</th><th>说明</th></tr>
{thr_rows}
</table>
</div>

<h2>为什么「低置信度就重新取图」这条路很贵</h2>
<div class="sub">这个模型的错误几乎全是形近字混淆（x↔o、c↔o、y↔u、w↔g、g↔z），而且它<b>错得同样自信</b>。</div>
<table>
<tr><th>样本</th><th>真值</th><th>识别</th><th>最低字符概率</th><th>对错</th></tr>
{crows}
</table>
<div class="note warn">
错误样本的最低字符概率落在 <b>84% ~ 99.6%</b>，而正确样本最低可到 <b>94.6%</b> —— 两者完全重叠。
要把 5 个错例全筛出来，阈值得压到 99.9%，那会连带标记 <b>15.5% 的正常样本</b>（每次登录约 1.18 次取图）。
理论上能把准确率推到 99.9% 以上，但代价是频繁刷新验证码、给学校登录接口增加额外请求，
而且错例样本只有 5 个，统计基础很薄。<b>因此没有把它写进脚本</b> —— 需要的话可以加成一个默认关闭的开关。
</div>

<h2>冷启动预算（实测）</h2>
<table>
<tr><th>资源</th><th>体积</th><th>实测下载</th><th>说明</th></tr>
<tr><td>ort.min.js</td><td class="num">0.54 MB</td><td class="num">0.54 s</td><td>与模型并行下载</td></tr>
<tr><td>ort-wasm-simd.wasm</td><td class="num">10.41 MB</td><td class="num">1.00 s</td><td>由 ORT 在 create() 阶段取，无法再提前</td></tr>
<tr><td>nn_model.onnx</td><td class="num">1.08 MB</td><td class="num">0.62 s</td><td>已与 ORT 并行，省约 0.6 秒</td></tr>
<tr><td>合计（首次）</td><td class="num">12.03 MB</td><td class="num">≈2.2 s</td><td>之后浏览器 HTTP 缓存长期命中，≈0</td></tr>
</table>

<h2>可运行性验证（不是"读起来没问题"，是跑过了）</h2>
<table>
<tr><th>检查项</th><th>方法</th><th>结果</th></tr>
<tr><td>调用的 API 是否真实存在</td><td>对着 onnxruntime-web@1.16.3 的实际 bundle 反查</td>
<td class="good">`get inputNames()` / `get outputNames()` 确为 InferenceSession 的 getter；`env.wasm.wasmPaths` / `numThreads` / `simd` 均存在，且 wasmPaths 会被拼接文件名 —— 脚本的值以 <code>/</code> 结尾 ✓</td></tr>
<tr><td>ORT 能否加载这个模型</td><td>用真实 onnxruntime-web 加载真实 nn_model.onnx</td>
<td class="good">InferenceSession.create 成功（121ms）；inputNames = ["input.1"]，outputNames = ["218".."222"]，与 postprocess 的假设完全一致 ✓</td></tr>
<tr><td>解码结果是否与官方实现一致</td><td>220 张样本，用<b>从交付脚本里抽出的</b> postprocess 源码执行</td>
<td class="good">与 Python 参考实现 <b>220/220 完全一致</b>；样本0 logits 最大绝对差 1.001e-5（浮点舍入级）✓</td></tr>
<tr><td>推理速度</td><td>220 张连续推理</td><td class="good">9.6 ms/张 ✓</td></tr>
<tr><td>灰度换算是否与 PIL 逐像素一致</td><td>对比 968,000 个像素的判定</td>
<td class="w2">原来直接拿浮点比阈值，有 <b>106 个像素</b>判定不同；改为 <code>Math.round</code> 后 <b>0 个不同</b>（已修）</td></tr>
<tr><td>不同 JPEG 解码器会不会影响结果</td><td>验证码是 4:2:0 子采样 JPEG，给 RGB 加 ±1/±2/±4/±8 随机扰动模拟解码器差异</td>
<td class="good">±8 扰动下 220 张里 <b>0 张</b>结果改变 → 解码器差异不影响准确率 ✓</td></tr>
</table>
<div class="note ok">
关于"<b>浏览器版 bundle 在 Node 里跑不通</b>"：<code>ort.min.js</code> 把 <code>path</code> polyfill 打成了空壳，
Emscripten 走 Node 分支时 <code>path.normalize</code> 不存在。这是 Node 环境的产物 ——
浏览器里该分支根本不进入（走 <code>fetch</code>）。因此验证时改用同包的 Node 构建
<code>ort-web.node.js</code>：同一套 ORT wasm 内核、同一份推理代码，只是 wasm 的获取方式由 fetch 换成 fs。
</div>

<h2>ResNet 主路径的全部错例（{len(err_ids)} / {len(TUNE)+len(HOLD)}）</h2>
<div class="sub">全部是单字符形近字混淆，且都是同一类"圆/曲笔画"的字母。</div>
<div class="grid">{err_cards}</div>

<div class="note">
<b>结论</b>
<ul>
<li><b>准确度已基本触顶。</b>220 张上 {acc("resnet_new", TUNE)+acc("resnet_new", HOLD)}/{len(TUNE)+len(HOLD)} = {(acc("resnet_new", TUNE)+acc("resnet_new", HOLD))/(len(TUNE)+len(HOLD))*100:.1f}%，
与上游项目标称的 98~99% 一致。阈值、预处理、TTA 三条路都实测无增益，剩下的错误是字形本身在该分辨率下就难以区分。</li>
<li><b>效率上还有小空间但不多了。</b>冷启动 2.2 秒已是一次性成本，本轮已把能并行的下载并行、把启动时机提到 document-start。
再往下要么换模型（需重训，收益不明），要么牺牲本地算力去省下载（不值）。</li>
<li>如果确实要再往上推，唯一有实测支撑的路径是<b>低置信度重新取图</b>，代价见上节，需要你确认是否值得。</li>
</ul>
</div>

</body></html>"""

open(OUT, "w", encoding="utf-8").write(html)
print("report ->", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")
print(f"调参集 ResNet {acc('resnet_new', TUNE)}/{len(TUNE)}  留出集 {acc('resnet_new', HOLD)}/{len(HOLD)}")
print(f"Tesseract 留出集 A={th['meta']['A_hit']}/{th['meta']['n']}  B={th['meta']['B_hit']}/{th['meta']['n']}")
