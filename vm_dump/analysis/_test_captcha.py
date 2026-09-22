import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""测试：同一 uuid 连续请求验证码 URL，检查是否返回不同验证码。"""
import sys, io, urllib.request, hashlib
sys.path.insert(0, _HERE)
from PIL import Image

base = "https://jaccount.sjtu.edu.cn/jaccount/captcha?uuid=9b1e4ceb-ade7-4726-bb38-492ebbbe534f"

def fetch(u, ref="https://jaccount.sjtu.edu.cn/jaccount/"):
    req = urllib.request.Request(u, headers={"Referer": ref, "User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=15).read()

def img_id(data):
    import io as _io
    im = Image.open(_io.BytesIO(data)).convert("L")
    return hashlib.md5(im.tobytes()).hexdigest()[:12], im.size

for i in range(4):
    d = fetch(base + ("&t=%d" % (i + 100)))
    hid, sz = img_id(d)
    print(f"req{i} bytes={len(d)} size={sz} hash={hid}")