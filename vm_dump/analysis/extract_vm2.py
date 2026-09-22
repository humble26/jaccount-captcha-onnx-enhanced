import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import re, os

SRC = r"C:\Users\g1507\AppData\Local\Microsoft\Edge\User Data\Default\Local Extension Settings\eeagobfjdenkkddmbclomhiblgggliao\000003.log"
OUT_DIR = _VD
os.makedirs(OUT_DIR, exist_ok=True)

data = open(SRC, "rb").read()
text = data.decode("utf-8", errors="ignore")
open(os.path.join(OUT_DIR, "raw_strings2.txt"), "w", encoding="utf-8").write(text)
print("text len:", len(text))

print("\n=== all @name ===")
for m in re.finditer(r"//\s*@name\s+(.+)", text):
    print(" -", m.group(1).strip()[:80])

for kw in ["captcha", "Captcha", "CAPTCHA", "验证码", "识别", "ocr", "OCR", "tesseract", "ddddocr"]:
    print(f"{kw!r}: {text.count(kw)}")
