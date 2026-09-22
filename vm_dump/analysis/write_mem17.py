import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""修复第十七轮记忆条目（先前在 bash 内联 python -c 里写反引号被命令替换吃掉）。"""
import io

P = _os.path.join(_REPO, ".workbuddy", "memory", "2026-09-21.md")

NOTE = """---

# 第十七轮 · 生成完整项目报告与修改记录说明

- 产出（工作区根目录）：
  - 项目报告.html（32.6 KB，10 章）—— 含目录锚点、KPI 卡片、版本时间线、事故复盘、教训清单
  - CHANGELOG.md —— v4.0.0 到 v4.5.1 逐版本记录（每版列出改了什么、修了哪些 bug、为什么）
- 报告素材来源：全部取自本项目各轮实跑记录与源码 CFG 注释，未新增任何推测性结论；
  未验证项已单独标注（如 jalogin 页面无法直接抓取验证 DOM）
- 关键数字复核后写入报告：215/220 = 97.7%；调参集 97.5% / 留出集 98.0%；
  McNemar 29:2 p<0.0001（对比用户原方案）、2:0 p=0.5（对比原版 ONNX 后处理）；
  冷启动 12.03 MB / 2.2 秒；低置信阈值表 95 / 99 / 99.5 / 99.9 四档
- 已同步到交付包 E:\\harness\\jAccount验证码识别-ResNet增强版\\ ，
  zip 由 12 条增至 14 条（83.2 KB 到 110.52 KB）
- 包内 README.txt 追加了文档索引（项目报告 / CHANGELOG / 实测报告 / 测试脚本）
- 校验：HTML 结构冒烟（html.parser 栈检查，未闭合为空、错误为空，11 个 section）；zip testzip 返回 None
- 备忘：报告里的「103 项 + 440 张次」= 15+16+9+7+8+22+26 条断言 + web_verify / verify_extension 各 220 张

## 本轮又踩一次的环境坑（已在此前记录过，仍复发）

在 bash 里用 python -c 内联写含反引号的中文内容，反引号被 shell 当作命令替换执行，
导致写入记忆的文件名（项目报告.html、CHANGELOG.md 等）全部丢失，且 shell 报
"command not found"。
正确做法：把要写入的内容放进独立 .py 文件（用文件写入工具创建，不经 shell），再运行它。
"""

s = io.open(P, encoding='utf-8').read()
i = s.find('# 第十七轮')
if i < 0:
    raise SystemExit('marker not found')
head = s[:i].rstrip('\n')
# 去掉第十七轮之前的那个分隔线（保持文档整洁）
if head.endswith('---'):
    head = head[:-3].rstrip('\n')
io.open(P, 'w', encoding='utf-8').write(head + '\n\n' + NOTE)
print('rewritten, chars =', len(io.open(P, encoding='utf-8').read()))
