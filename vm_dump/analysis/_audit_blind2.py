import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

# -*- coding: utf-8 -*-
"""盲测（修订版）：联络表上只印大号序号，不印文件名、不印预测。

为什么重做：上一版把文件名与真值印在图上，而我读小号文字标签时发生错位
（例如把别的样本的 truth 读成当前样本的），导致核对不可靠。
本版改为：
  - 图上只有大号序号 #01..#24（数字比长文件名好认得多）
  - 序号 → 文件名的映射只存在于盲测清单 json 里，看图时看不到
  - 我按序号写出读数，由 compare 子命令与真值比对

用法：
  python _audit_blind2.py gen      # 生成两张盲测图 + 清单
  python _audit_blind2.py compare reads.json   # 把我的读数与真值比对
"""
import os
import sys
import json
import random
from PIL import Image, ImageDraw

W = _VD
NEW = os.path.join(W, 'new300')
A = r'E:\harness\jAccount验证码识别-项目归档\02-重构研究\analysis\_sampling_out'
OUT = os.path.join(W, 'audit_sheets')
os.makedirs(OUT, exist_ok=True)
PICK_FILE = os.path.join(OUT, 'blind2_pick.json')

final = {r['file']: r for r in json.load(open(os.path.join(A, 'final_labels_300.json'), encoding='utf-8'))}


def gen():
    hot = sorted([r for r in final.values() if r['pos4'] in ('w', 'g', 'z')],
                 key=lambda r: r['confidence'])
    hot_files = [r['file'] for r in hot][:24]
    rest = [f for f in sorted(final) if f not in set(hot_files)]
    rnd = random.Random(20260922)
    rand_files = sorted(rnd.sample(rest, 24))
    pick = hot_files + rand_files          # #01..#24 = 高危位4；#25..#48 = 随机
    json.dump({'hot': hot_files, 'random': rand_files},
              open(PICK_FILE, 'w', encoding='utf-8'), indent=1)

    SCALE, PAD, BAR = 4, 8, 46

    def tile(fn, idx):
        im = Image.open(os.path.join(NEW, fn)).convert('L').convert('RGB')
        im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
        canvas = Image.new('RGB', (im.width + PAD * 2, im.height + BAR + PAD), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 4), '#%02d' % idx, fill=(200, 0, 0), font_size=34)
        canvas.paste(im, (PAD, BAR))
        return canvas

    def sheet(items, path, title, cols=3, start=1):
        tiles = [tile(f, start + i) for i, f in enumerate(items)]
        tw = max(t.width for t in tiles); th = max(t.height for t in tiles)
        rows = (len(tiles) + cols - 1) // cols
        canvas = Image.new('RGB', (tw * cols, th * rows + 26), (245, 246, 248))
        ImageDraw.Draw(canvas).text((6, 6), title, fill=(20, 20, 20), font_size=18)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas.paste(t, (c * tw, 26 + r * th))
        canvas.save(path)
        print('已生成 %s  %dx%d  %d 张（#%02d 起）' % (path, canvas.width, canvas.height, len(tiles), start))

    sheet(pick[:24], os.path.join(OUT, 'blind2_A.png'), 'SHEET A  #01-#24', 3, 1)
    sheet(pick[24:], os.path.join(OUT, 'blind2_B.png'), 'SHEET B  #25-#48', 3, 25)
    print()
    print('清单已存:', PICK_FILE)


def compare():
    reads = json.load(open(sys.argv[2], encoding='utf-8'))
    pick = json.load(open(PICK_FILE, encoding='utf-8'))
    order = pick['hot'] + pick['random']
    n = len(order)
    bad = []
    miss = []
    for i, f in enumerate(order, 1):
        key = '%02d' % i
        mine = reads.get(key, reads.get(str(i), '')).strip().lower()
        if not mine:
            miss.append(key); continue
        truth = final[f]['truth']
        if mine != truth:
            # 允许长度差异时标注
            bad.append((key, f, mine, truth, final[f]['confidence'], final[f]['source']))
    print('盲测 %d 张：我未读的 %d 张，读数与真值不一致 %d 张' % (n, len(miss), len(bad)))
    if miss:
        print('  未读:', miss)
    for key, f, mine, truth, conf, src in bad:
        print('  #%s %-10s 我读=%s  真值=%s  (conf=%.5f source=%s)' % (key, f, mine, truth, conf, src))
    if not bad:
        print('  ==> 全部一致：这批样本的自标注真值在这 48 张上零错误')
    print()
    print('分组统计：')
    for grp, name in (('hot', '位4∈{w,g,z} 组'), ('random', '随机组')):
        fs = pick[grp]
        idx0 = 1 if grp == 'hot' else 25
        ok = 0; tot = 0
        for j, f in enumerate(fs):
            key = '%02d' % (idx0 + j)
            mine = reads.get(key, '').strip().lower()
            if not mine:
                continue
            tot += 1
            if mine == final[f]['truth']:
                ok += 1
        print('  %-18s %d/%d 一致' % (name, ok, tot))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'gen'
    {'gen': gen, 'compare': compare}[cmd]()
