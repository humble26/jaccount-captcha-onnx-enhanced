import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import urllib.request, os, re, gzip, io

W = _VD
HDR = {"User-Agent": "Mozilla/5.0"}

url = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js"
req = urllib.request.Request(url, headers=HDR)
with urllib.request.urlopen(req, timeout=60) as r:
    raw = r.read()
print("downloaded", len(raw), "bytes")

try:
    src = raw.decode("utf-8", "ignore")
except Exception:
    src = ""

p = os.path.join(W, "ort.min.js")
open(p, "wb").write(raw)

checks = [
    "outputNames", "inputNames", "wasmPaths", "numThreads",
    "InferenceSession", "Tensor", "graphOptimizationLevel",
    "executionProviders", "getOutputs", "getInputs",
]
print("\n=== ort.min.js API 存在性检查 ===")
for c in checks:
    print(f"  {c:26s} : {src.count(c)}")
