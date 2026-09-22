import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import urllib.request, os

OUT = _os.path.join(_VD, "ref")
os.makedirs(OUT, exist_ok=True)

targets = [
    ("PhotonQuantum", "master", "ocr.py"),
    ("PhotonQuantum", "master", "utils.py"),
    ("PhotonQuantum", "master", "ocr_legacy.py"),
    ("danyang685", "master", "ocr.py"),
]

HDR = {"User-Agent": "Mozilla/5.0"}

for repo, br, fn in targets:
    got = False
    for base in (f"https://cdn.jsdelivr.net/gh/{repo}/jaccount-captcha-solver@{br}/{fn}",
                 f"https://raw.githubusercontent.com/{repo}/jaccount-captcha-solver/{br}/{fn}"):
        try:
            req = urllib.request.Request(base, headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                b = r.read()
            p = os.path.join(OUT, f"{repo}_{fn}")
            open(p, "wb").write(b)
            print(f"OK {repo}/{fn}  {len(b)} bytes -> {p}")
            got = True
            break
        except Exception as e:
            print(f"FAIL {base} -> {type(e).__name__}: {e}")
    if not got:
        print(f"!! could not fetch {repo}/{fn}")
