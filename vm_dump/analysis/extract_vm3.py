import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import re, json, os

SRC = r"C:\Users\g1507\AppData\Local\Microsoft\Edge\User Data\Default\Local Extension Settings\eeagobfjdenkkddmbclomhiblgggliao\000003.log"
OUT_DIR = _VD
os.makedirs(OUT_DIR, exist_ok=True)

text = open(SRC, "rb").read().decode("utf-8", errors="ignore")

dec = json.JSONDecoder()
found = []
for m in re.finditer(r'"//\s*==UserScript==', text):
    start = m.start()
    try:
        code, end = dec.raw_decode(text[start:])
    except Exception as e:
        continue
    if not isinstance(code, str):
        continue
    name = ""
    nm = re.search(r"@name\s+(.+)", code)
    if nm:
        name = nm.group(1).strip()
    found.append((name, code, start, end))

print("found scripts:", len(found))
for i, (name, code, s, e) in enumerate(found):
    safe = re.sub(r"[^\w\u4e00-\u9fff-]", "_", name)[:50] or f"script{i}"
    fn = os.path.join(OUT_DIR, f"{i:02d}_{safe}.user.js")
    open(fn, "w", encoding="utf-8").write(code)
    print(f"  [{i:02d}] {name[:60]!r} len={len(code)} -> {os.path.basename(fn)}")
