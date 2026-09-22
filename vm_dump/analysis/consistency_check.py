import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""两个交付物（油猴脚本 / 扩展）的现成解配置一致性校验

目的：同一套策略必须在两个交付物里完全一致，否则用户装了哪版行为不同，
      而这类"行为差异"在真机上极难排查。
"""
import os, re, sys, json

WS = _REPO
US = os.path.join(WS, "jaccount-captcha-onnx-enhanced.user.js")
EX = os.path.join(WS, "extension-src", "app.js")
MF = os.path.join(WS, "extension-src", "manifest.json")

us = open(US, encoding="utf-8").read()
ex = open(EX, encoding="utf-8").read()
mf = json.load(open(MF, encoding="utf-8"))

checks = []

print("=" * 78)
print("1. 版本号")
print("=" * 78)
uv = re.search(r"@version\s+([\d.]+)", us).group(1)
# 油猴脚本里还有一份 VERSION 常量用于状态条显示 —— 必须与 @version 一致。
# 这正是本轮抓到的缺陷：@version 已经写成 4.5.0，而 VERSION 还停在 4.4.4，
# 于是状态条上报的是旧版本号，用户按提示来排查会被误导。
uv_const = re.search(r"const VERSION = '([\d.]+)'", us)
uv_const = uv_const.group(1) if uv_const else None
ev = re.search(r"const VERSION = '([\d.]+)'", ex).group(1)
mv = mf["version"]
print(f"  油猴脚本 @version      : {uv}")
print(f"  油猴脚本 VERSION 常量  : {uv_const}")
print(f"  扩展 app.js VERSION    : {ev}")
print(f"  扩展 manifest version  : {mv}")

ok_us_v = (uv_const is not None and uv_const == uv)
print(f"  {'✓' if ok_us_v else '✗'} 油猴脚本内部两个版本号一致")
ok_ex_v = (ev == mv)
print(f"  {'✓' if ok_ex_v else '✗'} 扩展内部两个版本号一致")


def semver_ok(v):
    return bool(re.fullmatch(r"\d+\.\d+\.\d+", v or ""))


# 两版的版本序列不需要相等（4.x 与 1.x 是不同产品线），但都要是合法的三段式
ok_sem = semver_ok(uv) and semver_ok(uv_const or "") and semver_ok(ev) and semver_ok(mv)
print(f"  {'✓' if ok_sem else '✗'} 四个版本号都是合法的 x.y.z 形式")
checks.extend([ok_us_v, ok_ex_v, ok_sem])

print()
print("=" * 78)
print("2. 关键配置一致性")
print("=" * 78)


def num(src, key):
    m = re.search(key + r"\s*:\s*([0-9.]+)", src)
    return float(m.group(1)) if m else None


for key, label in [("lowMargin", "决策间隔阈值（主判据）"),
                   ("lowConfidence", "置信度阈值（仅显示用）"),
                   ("maxRefreshRetries", "最大换图重试次数")]:
    a, b = num(us, key), num(ex, key)
    same = (a is not None and a == b)
    checks.append(same)
    print(f"  {label:16s}: 油猴={a}  扩展={b}  {'✓ 一致' if same else '✗ 不一致'}")


def sels(src):
    """
    提取 refreshSelectors 里的全部选择器。

    这里踩过一个坑：原先写的是 re.search(r'refreshSelectors:\\s*\\[(.*?)\\]', src, re.S)，
    非贪婪匹配会在**第一个 `]`** 处就停下 —— 而选择器本身含方括号
    （'img[id*="captcha" i] + a'），于是列表被截成前 5 个，后 4 个从未参与比较。
    两侧同样被截断，所以这项检查一直"通过"，却什么都没真正守住。
    改为按**方括号配平**取整个数组。
    """
    i = src.index("refreshSelectors")
    j = src.index("[", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "[":
            depth += 1
        elif src[k] == "]":
            depth -= 1
            if depth == 0:
                return re.findall(r"'([^']+)'", src[j:k + 1])
    return []


sa, sb = sels(us), sels(ex)
same_s = (sa == sb and len(sa) > 0)
checks.append(same_s)
print(f"  换图选择器        : 油猴 {len(sa)} 个 / 扩展 {len(sb)} 个  {'✓ 一致' if same_s else '✗ 不一致'}")
if not same_s:
    print(f"     油猴: {sa}")
    print(f"     扩展: {sb}")

# 关键函数是否两边都有
print()
print("=" * 78)
print("3. 关键函数/标识符存在性")
print("=" * 78)
syms = ["findRefreshButton", "refreshCaptcha", "markInput", "isLow", "maxRefreshRetries",
        "suppressObserver", "observerSuppressed", "releaseObserver", "refreshed"]
for s in syms:
    a, b = (s in us), (s in ex)
    checks.append(a and b)
    print(f"  {s:20s}: 油猴 {'✓' if a else '✗'}  扩展 {'✓' if b else '✗'}")

# 确认旧的 0.60 阈值已彻底移除
print()
print("=" * 78)
print("4. 旧阈值 0.60 是否已清除")
print("=" * 78)
bad_us = len(re.findall(r"minConfidence\s*<\s*0\.60", us)) + len(re.findall(r"lowConfidence:\s*0\.60", us))
bad_ex = len(re.findall(r"minConfidence\s*<\s*0\.60", ex)) + len(re.findall(r"lowConfidence:\s*0\.60", ex))
checks.append(bad_us == 0 and bad_ex == 0)
print(f"  油猴残留 {bad_us} 处，扩展残留 {bad_ex} 处  "
      f"{'✓ 已清除' if bad_us == 0 and bad_ex == 0 else '✗ 仍有残留'}")

# 换图闸门必须真的接在观察者调度上，否则无限换图循环会复现
print()
print("=" * 78)
print("5. 换图闸门是否真的接在观察者调度上")
print("=" * 78)
for name, src in [("油猴", us), ("扩展", ex)]:
    # schedule() 里必须出现 observerSuppressed() 的调用
    i = src.index("const schedule = (why) =>")
    seg = src[i:i + 500]
    wired = "observerSuppressed()" in seg
    checks.append(wired)
    print(f"  {name} schedule() 内置闸门判断 : {'✓ 已接' if wired else '✗ 未接 —— 无限换图循环会复现'}")

# 「换图后回填历史最高置信」是个真实缺陷，必须保证它不会回来：
# 换图之后屏幕上已经不是那张图了，把 best 的答案填进去必然与眼前的验证码不符。
print()
print("=" * 78)
print("6. 不得存在「换图后回填历史最高置信」的逻辑")
print("=" * 78)
for name, src in [("油猴", us), ("扩展", ex)]:
    # 特征串必须精确：postprocess() 里的 argmax 也用 `let best = 0`，那是完全合法的
    # 局部变量，不能一刀切地禁掉 best 这个标识符，否则误报。
    # 要禁的是重试循环里那两句：`let best = res`（初始化）和 `best.minConfidence`（回退比较）。
    leftovers = (re.findall(r"let\s+best\s*=\s*res\b", src)
                 + re.findall(r"best\.minConfidence", src))
    ok = len(leftovers) == 0
    checks.append(ok)
    print(f"  {name}: {'✓ 已清除' if ok else '✗ 仍有 ' + str(len(leftovers)) + ' 处残留'}")

# 换图等待必须先把「src 真的变了」作为前置条件，不能无条件立刻 decode()：
# decode() 在 src 未变且旧图已解码时会立即兑现，异步站点上等于在旧图上识别。
print()
print("=" * 78)
print("7. 换图等待必须以「src 真的变了」为前提")
print("=" * 78)
for name, src in [("油猴", us), ("扩展", ex)]:
    i = src.index("async function refreshCaptcha(img)")
    seg = src[i:i + 6000]
    ok = ("oldSrc" in seg) and ("decodeNow" in seg)
    checks.append(ok)
    print(f"  {name}: {'✓ oldSrc + decodeNow 已就位' if ok else '✗ 仍是「无条件立刻 decode」的旧写法'}")

# v4.5.2 起换图判据由「最小 softmax 置信」改为「最小 logit 决策间隔」。
# 依据（220 张实测，两个独立子集分别验证）：在同样拦下全部错误的前提下，
#   调参集自动化率 80.8% -> 93.3%、留出集 89.0% -> 94.0%；
#   220 张整体换图率 15.5% -> 6.4%（300 张新样本 14.0% -> 7.0%）。
# 这一节守住四件事，防止判据被改回去或只改单边：
#   a) 两版 CFG 都有 lowMargin（值相同由第 2 节负责）
#   b) 两版 postprocess 都返回 minMargin
#   c) 两版 isLow 都用 minMargin 与 CFG.lowMargin 比较
#   d) 两版 isLow 都不得再残留 minConfidence 判据
print()
print("=" * 78)
print("8. 换图判据必须是「决策间隔」(minMargin) 而非置信度")
print("=" * 78)
for name, src in [("油猴", us), ("扩展", ex)]:
    has_cfg = bool(re.search(r"lowMargin\s*:\s*[0-9.]+", src))
    has_ret = bool(re.search(r"minMargin\s*:", src))
    islow_new = bool(re.search(r"const isLow = \(r\) => r && r\.minMargin < CFG\.lowMargin;", src))
    islow_old = bool(re.search(r"isLow[^\n]*minConfidence", src))
    ok = has_cfg and has_ret and islow_new and (not islow_old)
    checks.append(ok)
    print(f"  {name}: CFG.lowMargin {'✓' if has_cfg else '✗'}"
          f" | postprocess 返回 minMargin {'✓' if has_ret else '✗'}"
          f" | isLow 用间隔 {'✓' if islow_new else '✗'}"
          f" | isLow 残留 conf 判据 {'✗ 有' if islow_old else '✓ 无'}")

print()
print("=" * 78)
print(f"结果：{'全部通过 ✓' if all(checks) else '存在失败 ✗'}  ({sum(checks)}/{len(checks)})")
print("=" * 78)
sys.exit(0 if all(checks) else 1)
