// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 全部 120 张真实样本上跑 Tesseract 两种配置：
 *   A = 用户当前脚本的方式（Tesseract.recognize 原尺寸 JPEG、无参数、每次新建 worker）
 *   B = 新脚本兜底方式（156 二值化 + 字母白名单 + PSM7，单 worker 复用）
 */
const fs = require('fs');
const path = require('path');
const Tesseract = require('tesseract.js');

const W = _VD;
const A = path.join(W, 'all_A_original');
const B = path.join(W, 'all_B_bin156');
const GT = JSON.parse(fs.readFileSync(path.join(W, 'ground_truth_all.json'), 'utf8'));
const OUT = path.join(W, 'tess_120.json');

const cleanOld = s => (s || '').replace(/[^a-zA-Z0-9]/g, '');
const cleanNew = s => (s || '').toLowerCase().replace(/[^a-z]/g, '');

(async () => {
  const out = { A: {}, B: {}, meta: {} };
  const aFiles = fs.readdirSync(A).filter(f => f.endsWith('.jpg')).sort();
  const bFiles = fs.readdirSync(B).filter(f => f.endsWith('.png')).sort();
  console.log(`A 组 ${aFiles.length} 张 / B 组 ${bFiles.length} 张`);

  console.log('\n--- A：用户当前脚本的方式 ---');
  const tA = Date.now();
  let n = 0;
  for (const f of aFiles) {
    const id = f.replace('.jpg', '');
    const t0 = Date.now();
    const { data } = await Tesseract.recognize(path.join(A, f), 'eng');
    out.A[id] = { pred: cleanOld(data.text), raw: data.text, ms: Date.now() - t0 };
    if (++n % 20 === 0) console.log(`   ${n}/${aFiles.length} ...`);
  }
  out.meta.A_total_ms = Date.now() - tA;
  console.log(`   A 组完成，总耗时 ${out.meta.A_total_ms}ms`);

  console.log('\n--- B：新脚本兜底方式 ---');
  const tB = Date.now();
  const worker = await Tesseract.createWorker('eng', 1);
  await worker.setParameters({
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyz',
    tessedit_pageseg_mode: '7',
    preserve_interword_spaces: '0',
    classify_bln_numeric_mode: '0',
  });
  out.meta.B_warmup_ms = Date.now() - tB;
  n = 0;
  for (const f of bFiles) {
    const id = f.replace('.png', '');
    const t0 = Date.now();
    const { data } = await worker.recognize(path.join(B, f));
    out.B[id] = { pred: cleanNew(data.text), ms: Date.now() - t0, conf: data.confidence };
    if (++n % 20 === 0) console.log(`   ${n}/${bFiles.length} ...`);
  }
  await worker.terminate();
  out.meta.B_total_ms = Date.now() - tB;

  const score = k => Object.keys(out[k]).filter(id => out[k][id].pred === GT[id]).length;
  out.meta.A_hit = score('A'); out.meta.B_hit = score('B');
  out.meta.n = Object.keys(GT).length;
  out.meta.A_avg_ms = out.meta.A_total_ms / out.meta.n;
  out.meta.B_avg_ms = (out.meta.B_total_ms - out.meta.B_warmup_ms) / out.meta.n;

  console.log('\n================ 汇总（120 张）================');
  console.log(`A 用户当前 : ${out.meta.A_hit}/${out.meta.n} = ${(out.meta.A_hit / out.meta.n * 100).toFixed(1)}%   平均 ${out.meta.A_avg_ms.toFixed(0)}ms/张`);
  console.log(`B 新脚本兜底: ${out.meta.B_hit}/${out.meta.n} = ${(out.meta.B_hit / out.meta.n * 100).toFixed(1)}%   平均 ${out.meta.B_avg_ms.toFixed(1)}ms/张 (预热 ${out.meta.B_warmup_ms}ms)`);

  fs.writeFileSync(OUT, JSON.stringify(out, null, 1), 'utf8');
  console.log('\n已写入 ' + OUT);
})().catch(e => { console.error('FAILED:', e); process.exit(1); });
