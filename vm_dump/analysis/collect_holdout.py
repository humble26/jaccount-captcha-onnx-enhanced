import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

"""采集 100 张留出集（holdout），只用于验证，不参与任何调参决策。"""
import os, time, json, socket, urllib.request

socket.setdefaulttimeout(10)
W = _VD
SD = os.path.join(W, "holdout")
os.makedirs(SD, exist_ok=True)

HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
    "Referer": "https://jaccount.sjtu.edu.cn/jaccount/jalogin",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

ok = fail = 0
for i in range(100):
    try:
        url = f"https://jaccount.sjtu.edu.cn/jaccount/captcha?_={int(time.time()*1000)}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADER), timeout=10) as r:
            b = r.read()
        if not (b[:2] == b"\xff\xd8" or b[:8] == b"\x89PNG\r\n\x1a\n"):
            fail += 1
            continue
        open(os.path.join(SD, f"h{i:03d}.png"), "wb").write(b)
        ok += 1
        if ok % 20 == 0:
            print(f"  {ok}/100 ...", flush=True)
    except Exception as e:
        fail += 1
        print(f"  [{i:03d}] {type(e).__name__}: {e}", flush=True)
    time.sleep(0.3)

print(f"\n留出集完成：{ok} 张成功，{fail} 张失败 -> {SD}")
