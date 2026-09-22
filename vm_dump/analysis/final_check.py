import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""交付前总校验 —— 确认两个安装包内容正确、版本一致、关键逻辑都在"""
import os, re, json, zipfile, hashlib, sys

WS = _REPO
MK_ZIP = r"E:\harness\jAccount验证码识别-ResNet增强版.zip"
EX_ZIP = r"E:\harness\jAccount验证码识别浏览器扩展版.zip"
EX_ZIP2 = r"E:\harness\jAccount验证码识别-浏览器扩展版.zip"

ok = []
def chk(cond, msg):
    ok.append(bool(cond))
    print(f"  {'✓' if cond else '✗'} {msg}")

print("=" * 78)
print("1. 源文件版本")
print("=" * 78)
us = open(os.path.join(WS, "jaccount-captcha-onnx-enhanced.user.js"), encoding="utf-8").read()
ex = open(os.path.join(WS, "extension-src", "app.js"), encoding="utf-8").read()
mf = json.load(open(os.path.join(WS, "extension-src", "manifest.json"), encoding="utf-8"))
uv = re.search(r"@version\s+([\d.]+)", us).group(1)
ev = re.search(r"const VERSION = '([\d.]+)'", ex).group(1)
print(f"  油猴 {uv} / 扩展 {ev} / manifest {mf['version']}")
# 期望值一律从源码读，绝不写字面量。
# 这里曾经写死 "4.5.0"（manifest 那侧也写死过 "1.0.5"），于是每次升版都会
# 报两条假失败，逼得人去"修"一个根本没坏的东西 —— 检查脚本比被测代码还脆。
uv_const = re.search(r"const VERSION = '([\d.]+)'", us).group(1)
chk(uv == uv_const, f"油猴 @version 与内部 VERSION 常量一致（{uv}）")
chk(ev == mf["version"], "扩展 app.js 与 manifest 版本一致")

print()
print("=" * 78)
print("2. 油猴安装包")
print("=" * 78)
chk(os.path.exists(MK_ZIP), f"压缩包存在")
if os.path.exists(MK_ZIP):
    z = zipfile.ZipFile(MK_ZIP)
    chk(z.testzip() is None, "完整性通过")
    names = z.namelist()
    cjs = [n for n in names if n.endswith(".user.js")]
    chk(len(cjs) == 1, f"含 1 个 .user.js（实际 {len(cjs)}）")
    if cjs:
        inner = z.read(cjs[0]).decode("utf-8")
        iv = re.search(r"@version\s+([\d.]+)", inner).group(1)
        # 与工作区源文件比对，不写死版本号字面量（理由同上）
        chk(iv == uv, f"包内脚本版本与源文件一致（{iv}）")
        # 关键逻辑必须在
        for sym in ["findRefreshButton", "refreshCaptcha", "maxRefreshRetries", "0.999",
                    "lowMargin", "minMargin"]:
            chk(sym in inner, f"包内脚本含 {sym}")
        chk("0.60" not in inner or "0.60" in "".join(re.findall(r"lowConfidence:\s*[\d.]+", inner)) is False
            or "lowConfidence: 0.60" not in inner, "包内脚本无旧的 0.60 阈值")
        d = hashlib.md5(z.read(cjs[0])).hexdigest()[:12]
        print(f"    包内脚本 md5[:12] = {d}")
    chk(any(n.endswith("安装说明.html") for n in names), "含安装说明.html")
    chk(any(n.endswith("README.txt") for n in names), "含 README.txt")
    chk(any("test_retry.js" in n for n in names), "含测试脚本 test_retry.js")
    chk(any("consistency_check.py" in n for n in names), "含测试脚本 consistency_check.py")
    print(f"    压缩包大小 {os.path.getsize(MK_ZIP)/1024:.1f} KB，{len(names)} 个条目")

print()
print("=" * 78)
print("3. 扩展安装包")
print("=" * 78)
EZIP = EX_ZIP if os.path.exists(EX_ZIP) else EX_ZIP2
chk(os.path.exists(EZIP), f"压缩包存在：{os.path.basename(EZIP)}")
if os.path.exists(EZIP):
    z = zipfile.ZipFile(EZIP)
    chk(z.testzip() is None, "完整性通过")
    names = z.namelist()
    mf_in = [n for n in names if n.endswith("manifest.json")]
    chk(len(mf_in) == 1, f"含 1 个 manifest.json")
    if mf_in:
        m = json.loads(z.read(mf_in[0]).decode("utf-8"))
        # 期望版本从源码 manifest 读，不写死常量。
        # 之前写死 "1.0.5"，每次升版都得手动改这一行；忘了改就会在打包
        # 明明成功的情况下报一个假失败 —— 校验脚本自己成了维护负担。
        src_mf = json.load(open(os.path.join(WS, "extension-src", "manifest.json"),
                               encoding="utf-8"))
        chk(m["version"] == src_mf["version"],
            f"包内 manifest 版本 = {m['version']}（源码 {src_mf['version']}）")
    cj = [n for n in names if n.endswith("content.js")]
    chk(len(cj) == 1, "含 content.js")
    if cj:
        c = z.read(cj[0]).decode("utf-8", errors="replace")
        for sym in ["findRefreshButton", "refreshCaptcha", "maxRefreshRetries",
                    "lowMargin", "minMargin"]:
            chk(sym in c, f"包内 content.js 含 {sym}")
    chk(any(n.endswith("nn_model.onnx") for n in names), "含模型 nn_model.onnx")
    chk(any(n.endswith(".wasm") for n in names), "含 wasm 推理引擎")
    chk(any(n.endswith("安装说明.html") for n in names), "含安装说明.html")
    print(f"    压缩包大小 {os.path.getsize(EZIP)/1024/1024:.2f} MB，{len(names)} 个条目")

print()
print("=" * 78)
print(f"总结果：{'全部通过 ✓' if all(ok) else '存在失败 ✗'}  ({sum(ok)}/{len(ok)})")
print("=" * 78)
sys.exit(0 if all(ok) else 1)
