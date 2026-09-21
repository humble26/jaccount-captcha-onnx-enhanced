import os, shutil, zipfile, hashlib, json

WS = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40"
BUILD = os.path.join(WS, "extension-build", "jaccount-captcha-extension")
PKG = r"E:\harness\jAccount验证码识别-浏览器扩展版"
ZIP = r"E:\harness\jAccount验证码识别-浏览器扩展版.zip"

# 0) 先清空整个包目录再重建。
#    以前这里只 copytree 到子目录、不清顶层，于是上一版遗留的散落文件
#    （而且往往是旧版本）会和新的子目录同时留在包里 ——
#    用户看到两套同名文件，很可能加载到旧的那一套，
#    压缩包体积也因此白白翻倍（7.9MB vs 3.9MB）。
#    打包必须从空目录开始，否则"包里到底是什么"就没有保证。
if os.path.exists(PKG):
    shutil.rmtree(PKG)
os.makedirs(PKG)

# 1) 扩展目录放进包内
inner = os.path.join(PKG, "jaccount-captcha-extension")
shutil.copytree(BUILD, inner)

# 2) 纯文本兜底入口
open(os.path.join(PKG, "README.txt"), "w", encoding="utf-8").write(
"""jAccount 验证码自动识别 · 浏览器扩展版
========================================

怎么装：
  双击打开同目录下的「安装说明.html」，按里面 6 步做。

  简要版：
    1. 把整个压缩包解压到一个固定位置（别放临时目录，之后不能挪走）
    2. Edge 地址栏输入 edge://extensions 回车
    3. 打开左下角「开发人员模式」
    4. 点「加载解压缩的扩展」
    5. 选择 jaccount-captcha-extension 这个文件夹（它里面有 manifest.json）
    6. 打开 jAccount 登录页即可使用

特点：
  不需要装暴力猴 / Tampermonkey。
  模型和推理引擎都打包在扩展里，全程不联网，首次加载约 0.2 秒。
  识别准确率 97.7%（220 张真实验证码实测）。

注意：
  浏览器会提示「此扩展程序并非来自应用商店」，这是自己加载扩展的正常提示，忽略即可。
  如果以前装过其他 jAccount 验证码脚本，先禁用，否则会互相覆盖输入框。
""")

# 2b) 图形化安装说明（扩展版专用，步骤与油猴版完全不同，不能复用）
_VERSION_FOR_DOC = json.load(open(os.path.join(BUILD, "manifest.json"), encoding="utf-8"))["version"]
open(os.path.join(PKG, "安装说明.html"), "w", encoding="utf-8").write(
"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>jAccount 验证码自动识别（浏览器扩展版）· 安装与使用说明</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--line2:#eef0f2;--tx:#1f2328;--mut:#656d76;--blue:#0969da;--green:#1a7f37;--amber:#9a6700;--red:#cf222e}
*{box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--tx);margin:0;padding:40px 20px;line-height:1.75;font-size:15px}
.wrap{max-width:820px;margin:0 auto}
h1{font-size:23px;font-weight:600;margin:0 0 6px}
h2{font-size:17px;font-weight:600;margin:38px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
p{margin:10px 0}
.lead{color:var(--mut);font-size:14px;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin:16px 0}
.steps{counter-reset:s;list-style:none;padding:0;margin:0}
.steps>li{counter-increment:s;position:relative;padding:0 0 22px 46px;border-left:1px solid var(--line2);margin-left:14px}
.steps>li:last-child{border-left-color:transparent;padding-bottom:0}
.steps>li::before{content:counter(s);position:absolute;left:-14px;top:0;width:28px;height:28px;line-height:28px;text-align:center;background:var(--blue);color:#fff;border-radius:50%;font-size:13px;font-weight:600}
.steps>li>b{display:block;font-size:15px;font-weight:600;margin-bottom:4px}
.steps>li>span{color:var(--mut);font-size:14px;display:block}
kbd,code{background:#eff1f3;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-family:ui-monospace,Consolas,monospace;font-size:13px}
pre{background:#f2f4f6;border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;font-size:13px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left}
th{background:#f2f4f6;font-weight:600}
.note{border-left:3px solid var(--amber);background:#fff8e5;padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0;font-size:14px}
.ok{border-left-color:var(--green);background:#eaf7ee}
.bad{border-left-color:var(--red);background:#fdeced}
.badge{display:inline-block;background:#eaf7ee;color:var(--green);border:1px solid #b9e0c5;border-radius:20px;padding:1px 10px;font-size:12px;font-weight:600}
</style>
</head>
<body>
<div class="wrap">
<h1>jAccount 验证码自动识别 · 浏览器扩展版</h1>
<p class="lead">版本 __VERSION__ &nbsp;·&nbsp; Chrome / Edge 通用（Manifest V3）&nbsp;·&nbsp; 本机离线运行 <span class="badge">零联网</span></p>

<div class="card">
<h2 style="margin-top:0">这个版本是什么</h2>
<p>打开上海交大 jAccount 登录页时，自动识别验证码并填进输入框。模型（ResNet-20）和推理引擎（ONNX Runtime）都打包在扩展里，<b>全程不发起任何网络请求</b>，不上传任何数据。</p>
<p>与「用户脚本版」的区别：<b>不需要</b>装暴力猴 / Tampermonkey，也不依赖任何 CDN —— 所以不存在"模型下载不下来"这类问题。代价是需要手动加载一次扩展到浏览器。</p>
</div>

<h2>安装步骤</h2>
<div class="card">
<ol class="steps">
<li><b>解压，并放到一个固定位置</b><span>把压缩包完整解压出来（例如 <code>D:\\jaccount-captcha-extension</code>）。<b>不要</b>放在下载目录或临时目录里 —— 加载扩展后这个文件夹不能删、不能移动，否则扩展会失效。</span></li>
<li><b>打开扩展管理页</b><span>Edge 地址栏输入 <code>edge://extensions</code> 回车；Chrome 输入 <code>chrome://extensions</code>。</span></li>
<li><b>打开「开发人员模式」</b><span>在页面左下角（Chrome 在右上角）找到开关并打开。</span></li>
<li><b>点「加载解压缩的扩展」</b><span>在页面上方出现的按钮栏里，点这一项。</span></li>
<li><b>选中 <code>jaccount-captcha-extension</code> 文件夹</b><span>注意要选<b>里面含有 <code>manifest.json</code> 的那一层</b>。选错层级会提示"清单文件缺失"。</span></li>
<li><b>完成</b><span>打开 <code>https://jaccount.sjtu.edu.cn/jaccount/jalogin</code>，验证码会自动识别并填入。</span></li>
</ol>
</div>

<div class="note">
<b>浏览器提示「此扩展程序并非来自应用商店」？</b> 这是自己加载扩展的正常提示，直接忽略。它只影响商店自动更新，不影响功能。
</div>

<h2>使用说明</h2>
<div class="card">
<table>
<tr><th>场景</th><th>行为</th></tr>
<tr><td>验证码出现</td><td>自动识别并填入输入框，然后焦点跳到用户名框</td></tr>
<tr><td>点了「换一张」</td><td>自动重新识别新图并覆盖旧答案</td></tr>
<tr><td>你正在验证码框里打字</td><td><b>完全不动你的输入</b>（哪怕框里是空的）</td></tr>
<tr><td>置信度偏低</td><td>输入框加橙色描边提示你核对，并显示一行说明</td></tr>
<tr><td>识别失败</td><td>在验证码下方显示具体失败原因，可截图反馈</td></tr>
</table>
<p>正常情况下页面是<b>干净的</b>：识别成功时不会在页面上留任何文字，答案已经在输入框里了。</p>
</div>

<h2>常见问题</h2>
<div class="card">
<p><b>Q：装好了但验证码没有自动填？</b><br>
先确认扩展在 <code>extensions</code> 页面处于「已启用」状态，然后<b>刷新登录页</b>。若仍无效，页面上会出现一行红色提示说明原因；把它截图发出来即可定位。</p>
<p><b>Q：会不会和别的验证码脚本冲突？</b><br>
会。如果你之前装过其他 jAccount 验证码脚本（油猴脚本或其他扩展），请<b>先禁用其中一个</b>，否则两边会互相覆盖输入框。</p>
<p><b>Q：需要联网吗？</b><br>
不需要。模型与推理引擎都在扩展里，装好之后断网也能用。</p>
<p><b>Q：想更新到新版本？</b><br>
把新版本解压<b>覆盖</b>到同一个文件夹，然后回扩展页点一下「重新加载」按钮（↻）即可。</p>
<p><b>Q：怎么看它到底在跑哪个版本？</b><br>
识别前的提示行上会带版本号（例如 <code>v__VERSION__ 已发现验证码，模型加载中…</code>）。</p>
</div>

<h2>准确率</h2>
<div class="card">
<p>220 张真实验证码实测：<b>215/220 = 97.7%</b>（与官方 Python 参考实现逐张比对，220/220 完全一致，差异只来自模型本身的误差）。</p>
<p>作为对比，不做任何处理的 Tesseract 方案是 74.2%。</p>
</div>

<h2>卸载</h2>
<div class="card">
<p>在 <code>extensions</code> 页面找到本扩展，点「移除」。之后可以安全删除那个文件夹。</p>
</div>

<p style="color:var(--mut2);font-size:13px;margin-top:36px;border-top:1px solid var(--line);padding-top:16px">
识别在浏览器本地完成，不上传验证码图片，也不收集任何信息。<br>
模型来源与致谢见同目录的 <code>README-许可与来源.txt</code>。
</p>
</div>
</body>
</html>
""".replace("__VERSION__", _VERSION_FOR_DOC))

# 3) 打包
if os.path.exists(ZIP):
    os.remove(ZIP)
base = os.path.basename(PKG)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, dirs, files in os.walk(PKG):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            arc = os.path.join(base, os.path.relpath(full, PKG)).replace("\\", "/")
            z.write(full, arc)

print(f"压缩包: {ZIP}  ({os.path.getsize(ZIP)/1048576:.2f} MB)")
print()
print("=== 包内结构 ===")
for root, dirs, files in os.walk(PKG):
    dirs.sort()
    depth = os.path.relpath(root, PKG).count(os.sep) + (0 if os.path.relpath(root, PKG) == "." else 1)
    name = os.path.basename(root) if os.path.relpath(root, PKG) != "." else base
    print("  " * depth + name + "\\")
    for f in sorted(files):
        p = os.path.join(root, f)
        print("  " * (depth + 1) + f"{f}   ({os.path.getsize(p)/1024:.1f} KB)")

print()
print("=== 关键校验 ===")
with zipfile.ZipFile(ZIP) as z:
    print("  完整性:", "通过" if z.testzip() is None else "损坏")
    names = z.namelist()
    need = [f"{base}/jaccount-captcha-extension/manifest.json",
            f"{base}/jaccount-captcha-extension/content.js",
            f"{base}/安装说明.html"]
    for n in need:
        print(f"  {'✓' if n in names else '✗'} 包内含 {n}")
    core = [n for n in names if n.startswith(f"{base}/jaccount-captcha-extension/")]
    print(f"  扩展文件夹条目数: {len(core)}")

m = json.load(open(os.path.join(inner, "manifest.json"), encoding="utf-8"))
print(f"  manifest 解析正常: {m['name']} v{m['version']} (MV{m['manifest_version']})")

# 4) 与构建目录逐文件比对
diff = []
for root, _, files in os.walk(BUILD):
    for f in files:
        a = os.path.join(root, f)
        b = os.path.join(inner, os.path.relpath(a, BUILD))
        if not os.path.exists(b) or hashlib.md5(open(a,'rb').read()).hexdigest() != hashlib.md5(open(b,'rb').read()).hexdigest():
            diff.append(f)
print("  与构建产物比对:", "全部一致 ✓" if not diff else f"不一致 {diff}")
