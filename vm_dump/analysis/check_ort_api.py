import urllib.request, os, re, gzip, io

W = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
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
