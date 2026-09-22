import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""批量抓取 jAccount 验证码原图（验证码图无需会话，UUID 复用可换图）。
用法: python _grab_captcha.py [--n 120] [--out DIR] [--delay 0.4]
"""
import os, sys, io, time, argparse, hashlib, urllib.request
from PIL import Image

BASE = "https://jaccount.sjtu.edu.cn/jaccount/captcha?uuid=9b1e4ceb-ade7-4726-bb38-492ebbbe534f"
OUT = os.path.join(_VD, "new300")


def fetch(u):
    req = urllib.request.Request(u, headers={
        "Referer": "https://jaccount.sjtu.edu.cn/jaccount/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    })
    return urllib.request.urlopen(req, timeout=15).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--resume", action="store_true", help="跳过已抓取计数")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    existing = {f for f in os.listdir(args.out) if f.endswith(".png")}
    saved, failed, dup = 0, 0, 0
    t0 = time.time()
    i = 0
    while saved < args.n:
        i += 1
        try:
            data = fetch(BASE)
        except Exception as e:
            failed += 1
            print(f"[i{i}] ERR {e}", flush=True)
            time.sleep(1.2)
            continue
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if im.size != (110, 40):
            print(f"[i{i}] bad size {im.size}, skip", flush=True)
            failed += 1
            time.sleep(0.5)
            continue
        # 去重：内容 hash
        h = hashlib.md5(im.tobytes()).hexdigest()[:16]
        if h in existing:
            dup += 1
            continue
        fn = os.path.join(args.out, f"c{len(existing):03d}.png") if not args.resume \
             else os.path.join(args.out, f"c{len(existing):03d}.png")
        im.save(fn, "PNG")
        existing.add(h)
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved}/{args.n}, elapsed {time.time()-t0:.0f}s", flush=True)
        time.sleep(args.delay)
    print(f"完成: saved={saved} failed={failed} dup={dup} 用时{time.time()-t0:.0f}s")
    print(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()