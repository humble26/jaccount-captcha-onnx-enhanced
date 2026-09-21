import re, os

SRC = r"C:\Users\g1507\AppData\Local\Microsoft\Edge\User Data\Default\Local Extension Settings\eeagobfjdenkkddmbclomhiblgggliao\000003.log"
OUT_DIR = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"
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
