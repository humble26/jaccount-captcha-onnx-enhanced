// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * bug 猎杀：把两个版本的 postprocess / softmax 原文抽出执行，做边界与实现一致性测试。
 * 重点验证我怀疑的几处：
 *   A. probBuf 复用 —— softmax(data, n, probBuf) 把 n 之外的残留旧值带入 sum 吗？
 *   B. blank 跳过时 confidences 为空 -> minConfidence = 0 的语义
 *   C. 输出名排序在 name 非数字时的稳定性
 *   D. dims 长度非 2 时的 n 推断
 *   E. 两版 postprocess 是否行为完全一致
 */
const fs = require('fs');
const path = require('path');

const WS = _REPO;
const MONKEY = path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js');
const EXT = path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js');

const CFG = { numClasses: 26, blankIndex: 26, charset: 'abcdefghijklmnopqrstuvwxyz' };

function extractPost(file, isExt) {
  const s = fs.readFileSync(file, 'utf8');
  const start = isExt ? s.indexOf('function softmax(') : s.indexOf('function softmax(');
  const endMarker = '/* ------------------------------------------------------------ ONNX 引擎';
  const end = s.indexOf(endMarker, start);
  if (start < 0 || end < 0) throw new Error('定位失败 ' + file);
  return new Function('CFG', 'log', 'warn',
    s.slice(start, end) + '; return { softmax, postprocess };')(CFG, () => { }, (...a) => console.log('   [warn]', ...a));
}

const M = extractPost(MONKEY, false);
const E = extractPost(EXT, true);

// ---- 构造一个假 session / 假 tensor ----
function mkSession(names) { return { outputNames: names }; }
function tensor32(vals) { return { data: Float32Array.from(vals), dims: [1, vals.length] }; }

let fail = 0;
const ok = (c, m) => { console.log(`  ${c ? '✓' : '✗'} ${m}`); if (!c) fail++; };

console.log('=== A. probBuf 复用是否污染 softmax ===');
// 手工构造一个"上一次残留了巨大值"的 buffer 场景。
// softmax(src, n, out) 只写 out[0..n-1]，而它自己的 sum 也只累加 0..n-1，
// 所以缓冲区外的残留不会影响本次结果。这里验证 n 变小（27 -> 26）时结论。
{
  // 模拟：5 个输出头，前 4 个 26 类，第 5 个 27 类。
  // 我们的 probBuf 初始 32，够用；但假如它先被 27 用过再被 26 用，
  // out[26] 仍是上一个头的旧值 —— 只要 sum 不读它就没问题。
  const src27 = new Array(27).fill(-5); src27[26] = 10; src27[0] = 0;
  const buf = new Float32Array(32).fill(999);         // 故意塞满垃圾
  const p = M.softmax(src27, 27, buf);
  const s = Array.from(p.slice(0, 27)).reduce((a, b) => a + b, 0);
  ok(Math.abs(s - 1) < 1e-5, `27 类 softmax 和为 1（实际 ${s.toFixed(8)}），垃圾残留未污染`);
  ok(buf[29] === 999, '缓冲区外(索引29)保持原值 999，未被 softmax 触碰 -> 无越界写');
}
{
  // n 从 27 降到 26：out[26] 是上一次写的旧值，本次不应参与
  const src26 = new Array(26).fill(0); src26[0] = 1;
  const buf = new Float32Array(32);
  M.softmax(new Array(27).fill(0).map((_, i) => i), 27, buf);   // 先用 27 写一遍
  const before26 = buf[26];
  const p = M.softmax(src26, 26, buf);
  const s = Array.from(p.slice(0, 26)).reduce((a, b) => a + b, 0);
  ok(Math.abs(s - 1) < 1e-5, `26 类 softmax 和为 1（实际 ${s.toFixed(8)}）`);
  ok(buf[26] === before26, `out[26] 未被改写（${before26} -> ${buf[26]}），不参与 sum`);
}

console.log('\n=== B. 全 blank 时 minConfidence 语义 ===');
{
  // 5 个头全部 argmax 到 blank(26)
  const out = {
    '218': tensor32([...new Array(26).fill(0), 9]),
    '219': tensor32([...new Array(26).fill(0), 9]),
    '220': tensor32([...new Array(26).fill(0), 9]),
    '221': tensor32([...new Array(26).fill(0), 9]),
    '222': tensor32([...new Array(26).fill(0), 9])
  };
  const r = M.postprocess(mkSession(['218', '219', '220', '221', '222']), out);
  ok(r.text === '', `全 blank -> text 为空字符串（"${r.text}"）`);
  ok(r.minConfidence === 0, `minConfidence = 0（实际 ${r.minConfidence}）`);
  console.log('   -> 上层 recognize() 用 text.length!==4&&!==5 拒绝填入，行为正确');
}

console.log('\n=== C. 正常 4 位 / 5 位识别 ===');
{
  const mk = (idx) => tensor32(new Array(27).fill(-10).map((v, i) => i === idx ? 5 : v));
  // 4 位：第 5 位 blank
  const out4 = { '218': mk(0), '219': mk(1), '220': mk(2), '221': mk(3), '222': mk(26) };
  const r4 = M.postprocess(mkSession(['218', '219', '220', '221', '222']), out4);
  ok(r4.text === 'abcd', `4 位 -> "${r4.text}"`);
  ok(r4.minConfidence > 0.99, `minConfidence 合理 (${r4.minConfidence.toFixed(4)})`);

  const out5 = { '218': mk(0), '219': mk(1), '220': mk(2), '221': mk(3), '222': mk(4) };
  const r5 = M.postprocess(mkSession(['218', '219', '220', '221', '222']), out5);
  ok(r5.text === 'abcde', `5 位 -> "${r5.text}"`);
}

console.log('\n=== D. blank 出现在中间（第2位 blank）===');
{
  const mk = (idx) => tensor32(new Array(27).fill(-10).map((v, i) => i === idx ? 5 : v));
  const out = { '218': mk(0), '219': mk(26), '220': mk(2), '221': mk(3), '222': mk(4) };
  const r = M.postprocess(mkSession(['218', '219', '220', '221', '222']), out);
  ok(r.text === 'acde', `中间 blank 被跳过 -> "${r.text}"（长度 4，可接受）`);
  // 注意：这会把 5 位结构压缩成 4 位，字位错位。模型正常不会出现，但记录下语义。
}

console.log('\n=== E. dims 异常时的 n 推断 ===');
{
  const t = { data: Float32Array.from(new Array(27).fill(0).map((_, i) => i === 25 ? 5 : -1)), dims: [1] };
  const out = { '218': t, '219': tensor32(new Array(26).fill(-1)), '220': tensor32(new Array(26).fill(-1)), '221': tensor32(new Array(26).fill(-1)), '222': tensor32([...new Array(26).fill(-1), 9]) };
  const r = M.postprocess(mkSession(['218', '219', '220', '221', '222']), out);
  ok(r.text.length >= 0, `dims=[1] 不抛异常，text="${r.text}"`);
}

console.log('\n=== F. outputNames 缺失退化路径 ===');
{
  const mk = (idx) => tensor32(new Array(26).fill(-10).map((v, i) => i === idx ? 5 : v));
  const out = { '218': mk(0), '219': mk(1), '220': mk(2), '221': mk(3), '222': mk(4) };
  const r = M.postprocess(mkSession(undefined), out);
  ok(r.text === 'abcde' || r.text.length === 5, `无 outputNames -> 按 key 枚举，text="${r.text}"`);
  const r2 = M.postprocess({}, out);
  ok(r2.text.length === 5, `session={} -> text="${r2.text}"`);
}

console.log('\n=== G. 两版 postprocess 行为完全一致 ===');
{
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  let diff = 0;
  for (let k = 0; k < 3000; k++) {
    const mk = () => {
      const n = Math.random() < 0.2 ? 27 : 26;
      const d = new Float32Array(n);
      for (let i = 0; i < n; i++) d[i] = (rnd() - 0.5) * 20;
      return { data: d, dims: [1, n] };
    };
    const out = { '218': mk(), '219': mk(), '220': mk(), '221': mk(), '222': mk() };
    const a = M.postprocess(mkSession(['218', '219', '220', '221', '222']), out);
    const b = E.postprocess(mkSession(['218', '219', '220', '221', '222']), out);
    if (a.text !== b.text || Math.abs(a.minConfidence - b.minConfidence) > 1e-12) diff++;
  }
  ok(diff === 0, `3000 组随机输入，两版输出完全一致（差异 ${diff} 组）`);
}

console.log(`\n结论: ${fail ? fail + ' 项异常' : '全部通过'}`);
