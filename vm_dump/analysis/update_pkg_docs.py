import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""更新油猴交付包内的说明文档（README.txt / 安装说明.html），补充新版功能"""
import os, re

PKG = r"E:\harness\jAccount验证码识别-ResNet增强版"
US = _os.path.join(_REPO, "jaccount-captcha-onnx-enhanced.user.js")

ver = re.search(r"@version\s+([\d.]+)", open(US, encoding="utf-8").read()).group(1)
print("当前脚本版本:", ver)

# ---------------- README.txt ----------------
readme = f"""jAccount 验证码自动识别 · ResNet(ONNX) 增强版  v{ver}
=================================================

怎么装：
  双击打开同目录下的「安装说明.html」，按里面的 5 步做即可。

  简要版：
    1. 浏览器装一个用户脚本管理器（推荐 Violentmonkey）
    2. 点管理器图标 →「打开控制台」→ 左上角「+」→「从文件安装」
    3. 选中 jaccount-captcha-onnx-enhanced.user.js → 点「安装」
    4. 打开 jAccount 登录页就能用了

本版新增（v{ver}）：
  · 低置信自动换图重试
    识别置信度不足时，脚本会自动点「换一张」并重新识别（最多 3 次），
    尽量换到一张能高置信识别的图，而不是把不确定的答案直接填进去。
  · 低置信阈值从 0.60 提到 0.999
    220 张实测显示，5 个识别错误样本的最低置信都在 84%~99.6% 之间，
    原来的 0.60 阈值永远够不到、等于从未生效。
    现在按 99.9% 判定：约 84.5% 的图一次给出 100% 正确的答案，
    其余不猜、改走换图重试。

实测数据（220 张真实验证码）：
  整体准确率        97.7%（留出集 98.0%）
  对比：原 Tesseract 方案 74.2%
  长度判定（4/5位） 100% 正确
  推理速度          约 9.5ms/张

注意：
  如果你以前装过其他 jAccount 验证码脚本，先把旧的禁用掉，
  两个同时跑会互相覆盖输入框。

首次加载约 12MB / 2 秒，之后走浏览器缓存。
全部在本机推理，不上传任何数据。

想关闭自动换图：编辑脚本，把 maxRefreshRetries 改成 0
（那就只做低置信标注，不再自动换图）。
"""
p = os.path.join(PKG, "README.txt")
open(p, "w", encoding="utf-8").write(readme)
print(f"已更新 {p} ({os.path.getsize(p)} 字节)")

# ---------------- 安装说明.html：在功能说明处追加新版内容 ----------------
hp = os.path.join(PKG, "安装说明.html")
html = open(hp, encoding="utf-8").read()
before = len(html)

NEW_BLOCK = """<div class="card" style="border-left:4px solid #1a7f37">
<h2 style="margin-top:0;border:0;padding:0">本版新增：低置信自动换图重试</h2>
<p>识别置信度不足时，脚本会自动点「换一张」并重新识别（最多 3 次），
尽量换到一张能高置信识别的图，而不是把不确定的答案直接填进去。</p>
<p>配套地，低置信阈值从原来的 <code>0.60</code> 提高到 <code>0.999</code>。
220 张实测显示，5 个识别错误样本的最低置信都在 84%~99.6% 之间 ——
原来的 0.60 阈值永远够不到，等于该功能从未生效。</p>
<p>现在按 99.9% 判定：约 <b>84.5%</b> 的图一次就能给出 <b>100% 正确</b>的答案，
其余不猜、改走换图重试。若重试用尽仍偏低，会填入置信最高的结果并加橙色描边，
提醒你核对。</p>
<p style="color:#656d76;font-size:14px">想关闭自动换图：编辑脚本，把 <code>maxRefreshRetries</code>
改成 <code>0</code>，就只做低置信标注、不再自动换图。</p>
</div>
"""

# 插到第一个 </body> 前；若没有 body 则直接追加
if "</body>" in html:
    html = html.replace("</body>", NEW_BLOCK + "</body>", 1)
else:
    html += NEW_BLOCK

open(hp, "w", encoding="utf-8").write(html)
print(f"已更新 {hp} ({before} -> {len(html)} 字节)")
