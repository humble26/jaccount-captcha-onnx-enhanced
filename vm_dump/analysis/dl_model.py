import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import urllib.request, os, re, struct

OUT = _VD
os.makedirs(OUT, exist_ok=True)

urls = [
    "https://cdn.jsdelivr.net/gh/RyanStarFox/JAccountVerificationCode@main/model/nn_model.onnx",
    "https://raw.githubusercontent.com/RyanStarFox/JAccountVerificationCode/main/model/nn_model.onnx",
]

data = None
for u in urls:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        print("OK downloaded from", u, "bytes =", len(data))
        break
    except Exception as e:
        print("FAIL", u, type(e).__name__, e)

if data:
    p = os.path.join(OUT, "nn_model.onnx")
    open(p, "wb").write(data)
    print("saved:", p)

    # extract ASCII strings from protobuf
    strs = re.findall(rb"[\x20-\x7e]{3,}", data)
    dec = [s.decode() for s in strs]
    print("\n--- strings (first 120) ---")
    for s in dec[:120]:
        print("   ", s)

    # look for likely tensor names
    print("\n--- candidate names ---")
    for s in dec:
        if re.fullmatch(r"[A-Za-z0-9_.\/:\- ]{2,40}", s) and not s.startswith(("onnx", "ai.")):
            pass
    # print names that look like graph io
    key = [s for s in dec if s in ("input.1", "input", "output", "logits", "x", "y") or "input" in s.lower() or "output" in s.lower()]
    print(key[:60])
