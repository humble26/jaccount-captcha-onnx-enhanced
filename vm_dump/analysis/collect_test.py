import os, time, base64, urllib.request, socket, sys, warnings
import numpy as np
import onnxruntime as rt
from PIL import Image

warnings.filterwarnings("ignore")
rt.set_default_logger_severity(4)          # 0=VERBOSE .. 4=FATAL，压掉 initializer 警告

socket.setdefaulttimeout(10)
W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
MODEL = os.path.join(W, "nn_model.onnx")
SAMPLES = os.path.join(W, "samples")
os.makedirs(SAMPLES, exist_ok=True)

sess = rt.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
print("[1] 模型全部输入:", [i.name for i in sess.get_inputs()])
print("    模型全部输出:", [o.name for o in sess.get_outputs()])

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
       "Referer": "https://jaccount.sjtu.edu.cn/jaccount/jalogin",
       "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}

LUT = [0] * 156 + [1] * 100

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def predict(pil):
    arr = np.array(pil.convert("L").point(LUT, "1"), dtype=np.float32)
    out = sess.run(None, {in_name: arr[None, None, ...]})

    new_txt, confs, blk = "", [], False
    for t in out:
        n = t.shape[1]
        a = int(np.argmax(t, 1)[0])
        if a >= 26:
            blk = True
            continue
        confs.append(float(softmax(t[0][:n])[a]))
        new_txt += chr(ord("a") + a)

    old_txt = "".join(chr(ord("a") + int(np.argmax(t[0][:26]))) for t in out)
    return new_txt, old_txt, (min(confs) if confs else 0.0), blk

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
for i in range(N):
    try:
        req = urllib.request.Request(f"https://jaccount.sjtu.edu.cn/jaccount/captcha?_={int(time.time()*1000)}", headers=HDR)
        with urllib.request.urlopen(req, timeout=10) as r:
            b = r.read()
        if not (b[:2] == b"\xff\xd8" or b[:8] == b"\x89PNG\r\n\x1a\n"):
            print(f"  [{i}] 非图片响应，跳过"); continue
        p = os.path.join(SAMPLES, f"c{i:02d}.png")
        open(p, "wb").write(b)
        pil = Image.open(p)
        new_txt, old_txt, conf, blk = predict(pil)
        rows.append({"file": f"c{i:02d}.png", "size": pil.size, "new": new_txt,
                     "old": old_txt, "conf": conf, "blk": blk, "bytes": len(b)})
        print(f"  [{i:02d}] {pil.size}  新版={new_txt:<6} 旧版={old_txt:<6} 置信={conf*100:5.1f}%  命中blank={blk}")
    except Exception as e:
        print(f"  [{i:02d}] 失败 {type(e).__name__}: {e}")
    time.sleep(0.35)

print(f"\n[2] 共取得 {len(rows)} 张真实验证码")
if rows:
    n4 = sum(1 for r in rows if len(r["new"]) == 4)
    n5 = sum(1 for r in rows if len(r["new"]) == 5)
    diff = sum(1 for r in rows if r["new"] != r["old"])
    lo = [r for r in rows if r["conf"] < 0.6]
    print(f"    新版长度分布: 4位={n4}  5位={n5}")
    print(f"    新版 vs 旧版结果不同的样本数: {diff} / {len(rows)}")
    print(f"    低置信(<60%)样本数: {len(lo)}")
    print(f"    旧版长度恒为 5 的比例: {sum(1 for r in rows if len(r['old']) == 5)}/{len(rows)}")

    cards = "\n".join(
        f'<figure><img src="data:image/png;base64,{base64.b64encode(open(os.path.join(SAMPLES, r["file"]), "rb").read()).decode()}">'
        f'<figcaption><b>{r["new"]}</b><br><span class="c">{r["conf"]*100:.0f}%</span>'
        f'<br><span class="o">旧:{r["old"]}</span></figcaption></figure>'
        for r in rows)

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>jAccount 验证码识别实测（{len(rows)} 张真实样本）</title>
<style>
body{{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;background:#f7f8fa;color:#1f2328;margin:0;padding:28px 32px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#656d76;font-size:13px;margin-bottom:20px}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:22px}}
.stat{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:12px 18px;min-width:130px}}
.stat b{{display:block;font-size:22px;line-height:1.2}} .stat span{{color:#656d76;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}}
figure{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;margin:0;padding:10px;text-align:center}}
figure img{{width:100%;max-width:132px;image-rendering:auto;border:1px solid #eef0f2;border-radius:6px;background:#fff}}
figcaption{{font-size:13px;margin-top:8px;line-height:1.7}}
figcaption b{{font-family:ui-monospace,Consolas,monospace;font-size:16px;color:#1a7f37;letter-spacing:1px}}
.c{{color:#656d76;font-size:12px}} .o{{color:#8c959f;font-size:11px;font-family:ui-monospace,monospace}}
.note{{margin-top:24px;padding:14px 16px;background:#fff;border:1px solid #e3e6ea;border-left:3px solid #d4920b;border-radius:8px;font-size:13px;line-height:1.75;color:#3d444d}}
</style></head><body>
<h1>jAccount 验证码识别实测</h1>
<div class="sub">ResNet-20 (ONNX) 本地推理 · {len(rows)} 张真实抓取样本 · 绿色大字为新版脚本识别结果</div>
<div class="stats">
  <div class="stat"><b>{len(rows)}</b><span>样本数</span></div>
  <div class="stat"><b>{n4} / {n5}</b><span>4 位 / 5 位</span></div>
  <div class="stat"><b>{diff}</b><span>新旧逻辑结果不同</span></div>
  <div class="stat"><b>{len(lo)}</b><span>低置信样本 (&lt;60%)</span></div>
</div>
<div class="grid">{cards}</div>
<div class="note"><b>怎么读这张表：</b>每张卡片下方绿色大字是新版脚本的识别结果，灰色 "旧:" 是原版脚本会填进去的值。<br>
旧版对 5 个输出头一律只读前 26 类，而第 5 个头实际有 27 类（多一个 blank 占位），所以它<b>永远不可能输出 4 位验证码</b>，只能靠置信度阈值硬猜。<br>
如果上面的绿色结果与图片内容一致，说明准确率符合预期（官方标称约 98~99%）。</div>
</body></html>"""
    hp = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\验证码识别实测报告.html"
    open(hp, "w", encoding="utf-8").write(html)
    print("    报告已生成:", hp)
