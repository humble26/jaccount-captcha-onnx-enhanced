// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 精确复现覆盖逻辑缺陷。直接抽出两版 recognize 的判定片段做真值表。
 */
const fs = require('fs');
const path = require('path');
const WS = _REPO;

// 从源码抽出判定表达式（原文照抄，保证测的是真代码）
function loadDecide(file, isExt) {
  const s = fs.readFileSync(file, 'utf8');
  const start = s.indexOf('const activeNow = document.activeElement === input;');
  const end = s.indexOf('setValue(input, text);', start);
  if (start < 0 || end < 0) throw new Error('定位失败');
  const body = s.slice(start, end);
  return new Function('input', 'lastFilled', 'log',
    'const document = arguments[3];' + body + '; return "FILL";')(null, null, null, null) ? null : body;
}

const monkeySrc = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');
const extSrc = fs.readFileSync(path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js'), 'utf8');

function buildDecider(src) {
  const start = src.indexOf('const activeNow = document.activeElement === input;');
  // 只取到那个 if 块结束为止，别把后面 setValue/clearOwnPlaceholder 也带进来
  const end = src.indexOf('}', src.indexOf('log(\'用户正在验证码框内输入，跳过自动填充\')', start));
  const body = src.slice(start, end + 1);
  return new Function('input', 'lastFilled', 'document', 'log',
    body + '\nreturn "FILL";');
}

const decideM = buildDecider(monkeySrc);
const decideE = buildDecider(extSrc);

console.log('场景真值表（FILL = 覆盖用户内容并填入；SKIP = 尊重用户不填）\n');
console.log('  用户行为                      焦点在内  框内值        当前结果   应该');
console.log('  ' + '-'.repeat(72));

const input = { value: '' };
const scenarios = [
  { name: '用户清空准备重新输入', focus: true, value: '', last: 'abcd', want: 'SKIP' },
  { name: '用户清空准备重新输入(无lastFilled)', focus: true, value: '', last: null, want: 'SKIP' },
  { name: '用户已手输 2 个字符', focus: true, value: 'xy', last: 'abcd', want: 'SKIP' },
  { name: '用户已手输完整答案', focus: true, value: 'wxyz', last: 'abcd', want: 'SKIP' },
  { name: '框内是我们上次填的、焦点在内', focus: true, value: 'abcd', last: 'abcd', want: 'FILL' },
  { name: '框内是空、焦点在别处(自动流程)', focus: false, value: '', last: 'abcd', want: 'FILL' },
  { name: '框内是我们填的、焦点在别处', focus: false, value: 'abcd', last: 'abcd', want: 'FILL' },
  { name: '用户改过值、焦点已离开(点换图)', focus: false, value: 'xy', last: 'abcd', want: 'FILL' },
];

let bad = 0;
for (const sc of scenarios) {
  input.value = sc.value;
  const doc = { activeElement: sc.focus ? input : { tagName: 'BUTTON' } };
  let got;
  try { got = decideM(input, sc.last, doc, () => { }); } catch (e) { got = 'ERR:' + e.message; }
  const mark = got === sc.want ? ' ' : '✗';
  if (got !== sc.want) bad++;
  console.log(`  ${mark} ${sc.name.padEnd(30)} ${String(sc.focus).padEnd(9)} "${sc.value}"`.padEnd(66) + `${String(got).padEnd(10)} ${sc.want}`);
}

console.log('\n扩展版是否同样行为:');
let same = true;
for (const sc of scenarios) {
  input.value = sc.value;
  const doc = { activeElement: sc.focus ? input : { tagName: 'BUTTON' } };
  const a = decideM(input, sc.last, doc, () => { });
  const b = decideE(input, sc.last, doc, () => { });
  if (a !== b) { same = false; console.log(`  ✗ 不一致于场景: ${sc.name} -> ${a} vs ${b}`); }
}
console.log(same ? '  ✓ 两版判定逻辑完全一致（同一个 bug 被复制了两份）' : '  ✗ 两版有差异');

console.log(`\n问题场景数: ${bad}`);
console.log('\n根因: valueIsOursOrEmpty 把「空值」当成「可以覆盖」，');
console.log('      但用户清空输入框正是"我要自己输"的强信号，尤其焦点还在框内时。');
