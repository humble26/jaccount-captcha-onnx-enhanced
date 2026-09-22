import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""把油猴脚本升到 4.4.1 并追加 4.4.1 的变更日志。"""
p = _os.path.join(_REPO, "jaccount-captcha-onnx-enhanced.user.js")
s = open(p, encoding="utf-8").read()

s = s.replace("// @version      4.4.0", "// @version      4.4.1", 1)

anchor = " *     整条 ONNX 路径静默失效且无任何日志。现在回调内自兜异常，并加短轮询 + 换源兜底。\n */"

addition = """ *     整条 ONNX 路径静默失效且无任何日志。现在回调内自兜异常，并加短轮询 + 换源兜底。
 *
 *  [4.4.1 紧急修复 —— 4.4.0 我改坏了东西]
 * 36. 【致命】4.4.0 重写 loadScript 时，漏掉了 `s.src = url;` 这一行。
 *     后果：<script> 元素没有地址，浏览器根本不会去加载它，onload 永不触发，
 *     于是每个源都走到 15 秒超时 -> 全部源失败 -> ONNX 加载失败 ->
 *     回退 Tesseract（国内拉不到 tesseract.js 与语言包）-> 双双失败 ->
 *     输入框显示"识别失败，请手动输入"。也就是「所有识别全部失败」。
 *     已补回赋值；对照实验：修复前 6 个场景全部挂死，修复后正常加载 7ms 返回。
 * 37. 事件挂载顺序：现在是**先挂 onload/onerror 再赋 src**。若先赋 src 再挂事件，
 *     脚本命中强缓存时可能在同一个微任务里就加载完成，事件派发时 onload 还是 null，
 *     事件被永久错过 —— 同样是 Promise 永不 settle。该场景已单独验证。
 * 38. loadScript 增加超时（默认 15s）：真实网络里 onload/onerror 可能一个都不来
 *     （中间设备挂起连接、CSP 静默丢弃、其它扩展拦截），没有超时会把
 *     "报错"变成"没反应"，后者难排查得多。
 * 39. loadOrt 不再只凭"脚本 onload 了"就认定 window.ort 存在：onload 触发但
 *     全局没挂上的情况确实存在（CSP 拦截、被别的扩展搅乱）。现在回读全局，
 *     不在就换源再试一轮，并抛出说明真实原因的异常。
 */"""

assert anchor in s, "锚点未找到"
s = s.replace(anchor, addition, 1)
open(p, "w", encoding="utf-8").write(s)
print("已更新到 4.4.1")
