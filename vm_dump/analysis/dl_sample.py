import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import urllib.request, os

W = _VD
HDR = {"User-Agent": "Mozilla/5.0"}

cands = []
for repo in ["danyang685", "ElectronicElephant", "LightQuantumArchive", "PhotonQuantum"]:
    for br in ["master", "main"]:
        cands.append(f"https://cdn.jsdelivr.net/gh/{repo}/jaccount-captcha-solver@{br}/captcha.jpg")
        cands.append(f"https://raw.githubusercontent.com/{repo}/jaccount-captcha-solver/{br}/captcha.jpg")

ok = None
for u in cands:
    try:
        req = urllib.request.Request(u, headers=HDR)
        with urllib.request.urlopen(req, timeout=15) as r:
            b = r.read()
        if b[:2] == b"\xff\xd8" or b[:8] == b"\x89PNG\r\n\x1a\n":
            ok = (u, b)
            print("OK", u, len(b), "bytes")
            break
        else:
            print("skip (not image)", u, b[:20])
    except Exception as e:
        print("fail", u.split('//')[1][:60], type(e).__name__, e)

if ok:
    p = os.path.join(W, "captcha.jpg")
    open(p, "wb").write(ok[1])
    print("saved ->", p)
else:
    print("NO IMAGE FOUND")
