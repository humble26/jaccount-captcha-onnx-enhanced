import urllib.request, os

OUT = r"C:\Users\g1507\WorkBuddy\2026-09-21-19-33-40\vm_dump\ref"
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
