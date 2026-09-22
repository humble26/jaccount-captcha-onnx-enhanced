// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * Tesseract 参数消融实验：6 种图像变体 × 4 种参数集 = 24 个组合，同一 worker 复用，可比。
 */
const fs = require('fs');
const path = require('path');
const Tesseract = require('tesseract.js');

const W = _VD;
const ROOT = path.join(W, 'ablation');
const OUT = path.join(W, 'ablation_result.json');

const TRUTH = {
  c00: 'yvcew', c01: 'fcvtf', c02: 'ymoad', c03: 'qgjnf', c04: 'txju',
  c05: 'fomr', c06: 'bbmbh', c07: 'wxmh', c08: 'kwwkc', c09: 'afex',
  c10: 'fsyeu', c11: 'jgvc', c12: 'riixo', c13: 'oanu', c14: 'ugal',
  c15: 'odeb', c16: 'mutww', c17: 'ohlf', c18: 'rbrmw', c19: 'whto',
};

const PARAMS = {
  'p0-无参数': {},
  'p1-仅白名单': { tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz' },
  'p2-白名单+PSM8': {
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz',
    tessedit_pageseg_mode: '8',
  },
  'p3-白名单+PSM7': {
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz',
    tessedit_pageseg_mode: '7',
  },
};

const clean = s => (s || '').toLowerCase().replace(/[^a-z]/g, '');
const ok = (p, t) => p === t;

(async () => {
  const variants = fs.readdirSync(ROOT).filter(v => fs.statSync(path.join(ROOT, v)).isDirectory());
  const results = [];

  for (const pname of Object.keys(PARAMS)) {
    const worker = await Tesseract.createWorker('eng', 1);
    await worker.setParameters(PARAMS[pname]);

    for (const v of variants) {
      const dir = path.join(ROOT, v);
      const files = fs.readdirSync(dir).filter(f => f.endsWith('.png')).sort();
      let hit = 0;
      let ms = 0;
      const misses = [];
      for (const f of files) {
        const id = f.replace('.png', '');
        const t0 = Date.now();
        const { data } = await worker.recognize(path.join(dir, f));
        ms += Date.now() - t0;
        const p = clean(data.text);
        if (ok(p, TRUTH[id])) hit++;
        else misses.push(`${id}:${p || '∅'}≠${TRUTH[id]}`);
      }
      results.push({ params: pname, variant: v, hit, n: files.length,
                     acc: hit / files.length, avgMs: ms / files.length, misses });
      console.log(`  ${pname.padEnd(14)} ${v.padEnd(12)} ${hit}/${files.length}  ${(hit / files.length * 100).toFixed(0)}%  ${(ms / files.length).toFixed(0)}ms`);
    }
    await worker.terminate();
  }

  results.sort((a, b) => b.acc - a.acc || a.avgMs - b.avgMs);
  console.log('\n================ 排名 ================');
  for (const r of results.slice(0, 10)) {
    console.log(`  ${(r.acc * 100).toFixed(0).padStart(3)}%  ${r.avgMs.toFixed(0).padStart(4)}ms  ${r.params}  ×  ${r.variant}`);
  }
  console.log('\n------ 最优组合的错例 ------');
  for (const m of results[0].misses) console.log('   ', m);

  fs.writeFileSync(OUT, JSON.stringify(results, null, 1), 'utf8');
  console.log('\n已写入 ' + OUT);
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
