import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import urllib.request, os, socket

socket.setdefaulttimeout(8)
W = _VD
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0"}

cands = [
    "https://jaccount.sjtu.edu.cn/jaccount/captcha",
    "https://jaccount.sjtu.edu.cn/jaccount/captcha?t=1",
    "https://cdn.jsdelivr.net/gh/danyang685/jaccount-captcha-solver@master/captcha.jpg",
    "https://cdn.jsdelivr.net/gh/ElectronicElephant/jaccount-captcha-solver@master/captcha.jpg",
    "https://cdn.jsdelivr.net/gh/LightQuantumArchive/jaccount-captcha-solver@master/captcha.jpg",
]

saved = 0
for i, u in enumerate(cands):
    try:
        req = urllib.request.Request(u, headers=HDR)
        with urllib.request.urlopen(req, timeout=8) as r:
            b = r.read()
            ct = r.headers.get("Content-Type", "")
        magic_ok = b[:2] == b"\xff\xd8" or b[:8] == b"\x89PNG\r\n\x1a\n"
        print(f"[{i}] {u}  -> {len(b)}B  ct={ct}  image={magic_ok}")
        if magic_ok:
            ext = ".jpg" if b[:2] == b"\xff\xd8" else ".png"
            p = os.path.join(W, f"sample_{saved}{ext}")
            open(p, "wb").write(b)
            print("    saved ->", p)
            saved += 1
    except Exception as e:
        print(f"[{i}] {u}  -> FAIL {type(e).__name__}: {e}")

print("total images saved:", saved)
