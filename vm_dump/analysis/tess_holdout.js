// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
const fs = require('fs');
const path = require('path');
const Tesseract = require('tesseract.js');

const W = _VD;
const A = path.join(W, 'hold_A_original');
const B = path.join(W, 'hold_B_bin156');
const GT = JSON.parse(fs.readFileSync(path.join(W, 'ground_truth_holdout.json'), 'utf8'));
const OUT = path.join(W, 'tess_holdout.json');

const cleanOld = s => (s || '').replace(/[^a-zA-Z0-9]/g, '');
const cleanNew = s => (s || '').toLowerCase().replace(/[^a-z]/g, '');

(async () => {
  const out = { A: {}, B: {}, meta: {} };
  const aF = fs.readdirSync(A).filter(f => f.endsWith('.jpg')).sort();
  const bF = fs.readdirSync(B).filter(f => f.endsWith('.png')).sort();

  const tA = Date.now();
  for (const f of aF) {
    const id = f.replace('.jpg', '');
    const { data } = await Tesseract.recognize(path.join(A, f), 'eng');
    out.A[id] = { pred: cleanOld(data.text) };
  }
  out.meta.A_ms = Date.now() - tA;

  const worker = await Tesseract.createWorker('eng', 1);
  await worker.setParameters({
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz',
    tessedit_pageseg_mode: '7',
    preserve_interword_spaces: '0',
    classify_bln_numeric_mode: '0',
  });
  const tB = Date.now();
  for (const f of bF) {
    const id = f.replace('.png', '');
    const { data } = await worker.recognize(path.join(B, f));
    out.B[id] = { pred: cleanNew(data.text) };
  }
  out.meta.B_ms = Date.now() - tB;
  await worker.terminate();

  const k = o => Object.keys(o).filter(id => o[id].pred === GT[id]).length;
  out.meta.A_hit = k(out.A); out.meta.B_hit = k(out.B); out.meta.n = aF.length;
  console.log(`留出集 Tesseract  A(原图直喂) = ${out.meta.A_hit}/${out.meta.n}  B(调优) = ${out.meta.B_hit}/${out.meta.n}`);
  fs.writeFileSync(OUT, JSON.stringify(out, null, 1), 'utf8');
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
