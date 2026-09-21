import re, json, sys, os

SRC = r"C:\Users\g1507\AppData\Local\Microsoft\Edge\User Data\Default\Local Extension Settings\eeagobfjdenkkddmbclomhiblgggliao\000003.log"
OUT_DIR = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump"

os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC, "rb") as f:
    data = f.read()

print("total bytes:", len(data))

# Extract UTF-8 decodable runs
text_chunks = []
i = 0
n = len(data)
while i < n:
    # find run of printable / utf8-ish bytes
    j = i
    while j < n and (data[j] >= 0x20 or data[j] in (0x09, 0x0a, 0x0d)):
        j += 1
    if j - i > 40:
        chunk = data[i:j]
        try:
            s = chunk.decode("utf-8")
            text_chunks.append(s)
        except UnicodeDecodeError:
            pass
    i = max(j, i + 1)

blob = "\n".join(text_chunks)
print("extracted text bytes:", len(blob))

with open(os.path.join(OUT_DIR, "raw_strings.txt"), "w", encoding="utf-8") as f:
    f.write(blob)

# Look for keys of interest
for kw in ["captcha", "Captcha", "CAPTCHA", "验证码", "UserScript", "userscript", "vmScripts", "viola"]:
    idxs = [m.start() for m in re.finditer(re.escape(kw), blob)]
    print(f"kw={kw!r} hits={len(idxs)}")
    for k in idxs[:5]:
        print("   ...", blob[max(0,k-120):k+160].replace("\n", "\\n")[:300])
