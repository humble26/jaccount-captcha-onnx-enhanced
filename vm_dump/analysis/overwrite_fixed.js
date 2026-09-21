/**
 * 验证修复：覆盖逻辑真值表（改后版本）。
 * 直接抽两版源码里的判定片段执行，确认用户输入不再被冲掉。
 */
const fs = require('fs');
const path = require('path');
const WS = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40';

function buildDecider(file, isExt) {
  const s = fs.readFileSync(file, 'utf8');
  const start = s.indexOf('const activeNow = document.activeElement === input;');
  if (start < 0) throw new Error('未找到判定起点 ' + file);
  // 判定块结束：到 "setValue(input, text);" 之前
  const end = s.indexOf('setValue(input, text);', start);
  const body = s.slice(start, end);
  return new Function('input', 'lastFilled', 'document', 'log', body + '\nreturn "FILL";');
}

const decideM = buildDecider(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), false);
const decideE = buildDecider(path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js'), true);

const input = { value: '' };
const other = { tagName: 'BUTTON' };

const scenarios = [
  { name: '用户清空准备重新输入',              focus: true,  value: '',     last: 'abcd', want: 'SKIP' },
  { name: '用户清空(无 lastFilled)',           focus: true,  value: '',     last: null,   want: 'SKIP' },
  { name: '用户已手输 2 个字符',               focus: true,  value: 'xy',   last: 'abcd', want: 'SKIP' },
  { name: '用户已手输完整答案',                focus: true,  value: 'wxyz', last: 'abcd', want: 'SKIP' },
  { name: '框内是我们填的、焦点仍在框内',       focus: true,  value: 'abcd', last: 'abcd', want: 'SKIP' },
  { name: '空值、焦点在别处(自动流程)',         focus: false, value: '',     last: 'abcd', want: 'FILL' },
  { name: '是我们填的、焦点在别处',             focus: false, value: 'abcd', last: 'abcd', want: 'FILL' },
  { name: '用户改过值、焦点已离开(点换图后)',   focus: false, value: 'xy',   last: 'abcd', want: 'FILL' },
];

console.log('修复后真值表：\n');
console.log('    用户行为                              焦点在内  框内值     结果    期望');
console.log('    ' + '-'.repeat(70));
let bad = 0;
for (const sc of scenarios) {
  input.value = sc.value;
  const doc = { activeElement: sc.focus ? input : other };
  let got;
  try { got = decideM(input, sc.last, doc, () => { }); } catch (e) { got = 'ERR'; }
  const okk = got === sc.want;
  if (!okk) bad++;
  console.log(`    ${okk ? '✓' : '✗'} ${sc.name.padEnd(34)} ${String(sc.focus).padEnd(9)} ${('"' + sc.value + '"').padEnd(10)} ${String(got).padEnd(7)} ${sc.want}`);
}

console.log('\n扩展版一致性：');
let diff = 0;
for (const sc of scenarios) {
  input.value = sc.value;
  const doc = { activeElement: sc.focus ? input : other };
  const a = decideM(input, sc.last, doc, () => { });
  const b = decideE(input, sc.last, doc, () => { });
  if (a !== b) { diff++; console.log(`    ✗ ${sc.name}: ${a} vs ${b}`); }
}
console.log(diff ? `    ✗ ${diff} 处不一致` : '    ✓ 两版完全一致');

console.log(`\n结论: ${bad ? '仍有 ' + bad + ' 个场景不符' : '全部符合预期 —— 用户输入不再被覆盖'}`);
process.exit(bad ? 1 : 0);
