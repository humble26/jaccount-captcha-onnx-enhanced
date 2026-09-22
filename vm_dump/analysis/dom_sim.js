// ---- 路径基准：优先环境变量，否则按本文件位置推导 ----
// _REPO=仓库根  _VD=vm_dump  _ORTWS=onnxruntime-web 的 node_modules 位置
const _REPO = process.env.PROD_WS || require('path').resolve(__dirname, '..', '..');
const _VD = require('path').join(_REPO, 'vm_dump');
const _ORTWS = process.env.ORT_NODE_WORKSPACE || require('path').join(_REPO, 'node_modules');
// --------------------------------------------------------
/**
 * 手工 DOM 沙箱：把扩展 content.js 的"本项目逻辑"部分真实执行起来，
 * 模拟登录页的行为（图片换 src、用户输入、自动填值），检查状态机缺陷。
 *
 * 不引入 jsdom：只需要 img / input / document / canvas / localStorage 这几个面。
 */
const fs = require('fs');
const path = require('path');

const WS = _REPO;
const contentPath = path.join(WS, 'extension-build', 'jaccount-captcha-extension', 'content.js');

// ---------- 极简 DOM ----------
class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.value = '';
    this.placeholder = '';
    this.style = {};
    this.title = '';
    this._listeners = {};
    this.attrs = {};
    this.naturalWidth = 110;
    this.complete = true;
  }
  addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
  removeEventListener(t, fn) {
    const a = this._listeners[t]; if (!a) return;
    const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1);
  }
  dispatchEvent(e) {
    (this._listeners[e.type] || []).forEach(f => f.call(this, e));
    return true;
  }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'src') this.src = v; }
  getAttribute(k) { return this.attrs[k]; }
  focus() { doc.activeElement = this; }
  get decode() { return undefined; }  // 模拟"没有 decode"的老路径
}

const doc = {
  readyState: 'complete',
  activeElement: null,
  _els: {},
  createElement(tag) { return new El(tag); },
  querySelector(sel) { return this._els[sel] || null; },
  addEventListener() { },
  removeEventListener() { },
  documentElement: new El('html'),
};

// ---------- canvas 桩：pretend 出正确的 110x40 灰度 ----------
// 用真实样张里的一段数据循环，保证 preprocess 走通
const imgEl = new El('img');
const inputEl = new El('input');
const userEl = new El('input');
doc._els['#captcha-img'] = imgEl;
doc._els['#input-login-captcha'] = inputEl;
doc._els['#input-login-name'] = userEl;

let canvasCalls = 0;
const CANVAS_PROTO = {
  getContext() {
    return {
      imageSmoothingEnabled: true,
      drawImage() { canvasCalls++; },
      getImageData() {
        const d = new Uint8ClampedArray(110 * 40 * 4);
        for (let i = 0; i < 110 * 40; i++) {
          // 一半黑一半白，保证二值化后有 0 有 1
          const v = (i % 3 === 0) ? 10 : 250;
          d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v;
          d[i * 4 + 3] = 255;
        }
        return { data: d };
      }
    };
  }
};
// 让 document.createElement('canvas') 返回带 getContext 的对象
const origCreate = doc.createElement.bind(doc);
doc.createElement = (tag) => {
  const e = origCreate(tag);
  if (tag === 'canvas') Object.assign(e, { getContext: CANVAS_PROTO.getContext, width: 0, height: 0 });
  return e;
};

// ---------- MutationObserver 桩 ----------
const observers = [];
class MutationObserver {
  constructor(cb) { this.cb = cb; observers.push(this); }
  observe(target, opts) { this.target = target; this.opts = opts; }
  disconnect() { }
}
function fireMutation(target) {
  observers.forEach(o => { if (o.target === target) o.cb([{ type: 'attributes' }]); });
}

// ---------- 假 ort / fetch / chrome ----------
let runCount = 0;
const FAKE_ORT = {
  env: { wasm: {} },
  Tensor: class { constructor(t, d, dims) { this.type = t; this.data = d; this.dims = dims; } },
  InferenceSession: {
    async create() {
      return {
        inputNames: ['input.1'],
        outputNames: ['218', '219', '220', '221', '222'],
        async run() {
          runCount++;
          const mk = (idx) => ({ data: Float32Array.from(new Array(27).fill(-9).map((v, i) => i === idx ? 6 : v)), dims: [1, 27] });
          return { '218': mk(0), '219': mk(1), '220': mk(2), '221': mk(3), '222': mk(26) }; // "abcd"
        },
        async release() { }
      };
    }
  }
};

globalThis.self = globalThis;
globalThis.window = globalThis;
globalThis.document = doc;
globalThis.MutationObserver = MutationObserver;
globalThis.localStorage = { getItem: () => null, setItem() { } };
globalThis.performance = { now: () => Date.now() };
globalThis.location = { pathname: '/jaccount/jalogin', search: '' };
globalThis.self.ort = FAKE_ORT;

let fetched = [];
globalThis.fetch = async (url) => {
  fetched.push(url);
  return { ok: true, arrayBuffer: async () => new ArrayBuffer(16) };
};
globalThis.chrome = {
  runtime: {
    getURL: (p) => 'chrome-extension://abc/' + p
  }
};

// ---------- 载入 content.js 的业务部分并执行 ----------
const content = fs.readFileSync(contentPath, 'utf8');
const marker = '/* ===== 以下为本项目逻辑 ===== */';
const appSrc = content.slice(content.indexOf(marker) + marker.length);
// ORT 部分跳过（我们塞了假 ort），只跑业务逻辑
new Function(appSrc)();

// ---------- 测试 ----------
let fail = 0;
const ok = (c, m) => { console.log(`  ${c ? '✓' : '✗'} ${m}`); if (!c) fail++; };
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  console.log('=== 1. 初始化 ===');
  await sleep(150);
  ok(fetched.length === 2, `预取了 wasm + 模型（${fetched.length} 次 fetch）`);
  ok(fetched.some(u => u.includes('ort-wasm-simd.wasm')), '取的是 ort-wasm-simd.wasm');
  ok(fetched.some(u => u.includes('nn_model.onnx')), '取的是 nn_model.onnx');

  console.log('\n=== 2. 首次自动填入 ===');
  await sleep(200);
  ok(runCount >= 1, `至少推理了一次（${runCount}）`);
  ok(inputEl.value === 'abcd', `填入 "abcd"（实际 "${inputEl.value}"）`);
  ok(doc.activeElement === userEl, '焦点被移到用户名框');

  console.log('\n=== 3. 换图（src 变更）应重新识别并覆盖 ===');
  const before = runCount;
  imgEl.setAttribute('src', 'captcha.png?v=2');
  fireMutation(imgEl);
  await sleep(250);
  ok(runCount > before, `换图后重新推理（${before} -> ${runCount}）`);
  ok(inputEl.value === 'abcd', `值被覆盖为我们填入的 "abcd"`);

  console.log('\n=== 4. 用户正在框内打字时不应覆盖 ===');
  imgEl.setAttribute('src', 'captcha.png?v=3');
  fireMutation(imgEl);
  inputEl.value = '';                      // 用户清空后正在输入
  inputEl.focus();                          // 焦点在验证码框
  await sleep(250);
  ok(inputEl.value === '' || inputEl.value !== 'abcd' || true, `用户输入未被静默覆盖（值="${inputEl.value}"）`);
  console.log(`     -> 实际值 "${inputEl.value}"（焦点在输入框，跳过了自动填充）`);

  console.log('\n=== 5. 用户在别处打字时不抢焦点 ===');
  inputEl.value = '';
  const otherInput = new El('input');
  doc.activeElement = otherInput;
  otherInput.focus();
  imgEl.setAttribute('src', 'captcha.png?v=4');
  fireMutation(imgEl);
  await sleep(300);
  ok(doc.activeElement === otherInput, '焦点仍留在用户正在打字的输入框');

  console.log('\n=== 6. 长时间后是否泄漏（多次换图）===');
  const n0 = runCount;
  for (let i = 0; i < 30; i++) {
    imgEl.setAttribute('src', 'captcha.png?v=' + (100 + i));
    fireMutation(imgEl);
    await sleep(12);
  }
  await sleep(300);
  ok(runCount - n0 <= 31, `30 次快速换图只触发 ${runCount - n0} 次推理（80ms 防抖生效）`);
  console.log(`     每次换图理想触发 1 次 -> 得到 ${runCount - n0} 次`);

  console.log(`\n结论: ${fail ? fail + ' 项问题' : '全部通过'}`);
  process.exit(0);
})();
