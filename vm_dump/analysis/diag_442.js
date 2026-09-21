/**
 * 验证 4.4.2 的诊断能力：在沙箱里**故意制造**各类失败，
 * 检查用户看到的 placeholder 是否包含可定位的具体原因。
 *
 * 覆盖场景：
 *   1. ORT 脚本全部加载失败（网络/墙/CSP）
 *   2. ORT 加载了但 window.ort 没出现（CSP 拦截）
 *   3. 模型全部源不可用
 *   4. session.run 抛异常（wasm backend 起不来）
 *   5. 推理成功但位数不对（模型给出意外结果）
 *   6. ONNX 挂 + Tesseract 也挂
 *   7. recognize 里的未捕获异常（事件派发抛错）
 */
const fs = require('fs');
const path = require('path');
const WS = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40';
const src = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');
const marker = '(function () {';
let body = src.slice(src.indexOf(marker));
body = body.replace('debug: false,', 'debug: true,');
if (!body.includes('debug: true,')) throw new Error('未能打开 debug 开关');

// ---------- DOM 桩 ----------
class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.value = ''; this.placeholder = ''; this.style = {}; this.title = '';
    this._listeners = {}; this.attrs = {};
    this.naturalWidth = 110; this.complete = true;
  }
  addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
  removeEventListener(t, fn) {
    const a = this._listeners[t]; if (!a) return;
    const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1);
  }
  dispatchEvent(e) {
    if (dispatchShouldThrow) throw new Error('页面监听器炸了');
    (this._listeners[e.type] || []).forEach(f => f.call(this, e)); return true;
  }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'src') this.src = v; }
  getAttribute(k) { return this.attrs[k]; }
  focus() { doc.activeElement = this; }
  appendChild(c) { this._children = this._children || []; this._children.push(c); return c; }
}
let dispatchShouldThrow = false;

const doc = {
  readyState: 'loading', activeElement: null, _els: {},
  head: new El('head'), documentElement: new El('html'),
  _domListeners: {},
  createElement(tag) { return new El(tag); },
  querySelector(sel) { return this._els[sel] || null; },
  addEventListener(t, fn) { (this._domListeners[t] = this._domListeners[t] || []).push(fn); },
  removeEventListener(t, fn) {
    const a = this._domListeners[t]; if (!a) return;
    const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1);
  },
};
const imgEl = new El('img'), inputEl = new El('input'), userEl = new El('input');
doc._els['#captcha-img'] = imgEl;
doc._els['#input-login-captcha'] = inputEl;
doc._els['#input-login-name'] = userEl;

doc.createElement = (tag) => {
  const e = new El(tag);
  if (tag === 'canvas') {
    e.getContext = () => ({
      imageSmoothingEnabled: true, drawImage() { },
      getImageData() {
        const d = new Uint8ClampedArray(110 * 40 * 4);
        for (let i = 0; i < 110 * 40; i++) {
          const v = (i % 3 === 0) ? 10 : 250;
          d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
        }
        return { data: d };
      },
      putImageData() { }
    });
  }
  return e;
};

globalThis.MutationObserver = class {
  constructor(cb) { this.cb = cb; }
  observe() { } disconnect() { }
};
globalThis.document = doc;
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.performance = { now: () => Date.now() };
globalThis.location = { pathname: '/jaccount/jalogin', search: '' };
globalThis.localStorage = { getItem: () => null };

// ---------- 场景参数 ----------
const SCEN = process.argv[2] || 'ort_script_fail';
let scriptMode = 'ok';      // ok | reject | silent | no_global
let modelMode = 'ok';       // ok | fail
let runMode = 'ok';         // ok | throw
let outMode = 'ok';         // ok | short

const gmCalls = [];
globalThis.GM_xmlhttpRequest = (opts) => {
  gmCalls.push(opts.url);
  if (opts.url.includes('.onnx')) {
    if (modelMode === 'fail') return setTimeout(() => opts.onerror && opts.onerror({ status: 0 }), 5);
    return setTimeout(() => opts.onload({ status: 200, response: new ArrayBuffer(1024 * 1024) }), 5);
  }
  setTimeout(() => opts.onload({ status: 200, response: new ArrayBuffer(8) }), 5);
};

// script 注入行为
const injected = [];
const origCreate = doc.createElement;
doc.createElement = (tag) => {
  const e = origCreate(tag);
  if (tag === 'script') {
    let _src = '';
    Object.defineProperty(e, 'src', {
      get() { return _src; },
      set(v) {
        _src = v; injected.push(v);
        if (scriptMode === 'silent') return;              // 既不成功也不失败
        if (scriptMode === 'reject') { setTimeout(() => e.onerror && e.onerror(), 5); return; }
        setTimeout(() => {
          if (scriptMode === 'no_global') { e.onload && e.onload(); return; }  // 加载了但不挂全局
          globalThis.ort = {
            env: { wasm: {} },
            Tensor: class { constructor(t, d, dims) { this.data = d; this.dims = dims; } },
            InferenceSession: {
              async create() {
                return {
                  inputNames: ['input.1'],
                  outputNames: ['218', '219', '220', '221', '222'],
                  async run() {
                    if (runMode === 'throw') throw new Error('no available backend found. ERR: [wasm] RuntimeError: wasm 编译失败');
                    const n = outMode === 'short' ? 3 : 5;
                    const mk = (i) => ({ data: Float32Array.from(new Array(27).fill(-9).map((v, k) => k === i ? 6 : v)), dims: [1, 27] });
                    if (outMode === 'short') return { '218': mk(0) };
                    return { '218': mk(0), '219': mk(1), '220': mk(2), '221': mk(3), '222': mk(26) };
                  },
                  async release() { }
                };
              }
            }
          };
          e.onload && e.onload();
        }, 5);
      }
    });
  }
  return e;
};

let tessMode = 'ok';
globalThis.Tesseract = {
  createWorker: async () => {
    if (tessMode === 'fail') throw new Error('Failed to fetch tesseract core wasm');
    return { setParameters: async () => { }, recognize: async () => ({ data: { text: 'zzzz', confidence: 90 } }) };
  }
};

const logs = [];
console.log = (...a) => logs.push('LOG ' + a.join(' '));
console.warn = (...a) => logs.push('WARN ' + a.join(' '));
console.error = (...a) => logs.push('ERR ' + a.join(' '));

// ---------- 按场景配置 ----------
switch (SCEN) {
  case 'ort_script_fail': scriptMode = 'reject'; modelMode = 'ok'; break;
  case 'ort_no_global': scriptMode = 'no_global'; break;
  case 'model_fail': modelMode = 'fail'; break;
  case 'run_throw': runMode = 'throw'; break;
  case 'short_output': outMode = 'short'; break;
  case 'both_fail': scriptMode = 'reject'; tessMode = 'fail'; break;
  case 'uncaught': dispatchShouldThrow = true; break;
  default: break;
}

let threw = null;
try { new Function(body)(); } catch (e) { threw = e; }

const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  await sleep(80);
  doc.readyState = 'interactive';
  (doc._domListeners['DOMContentLoaded'] || []).forEach(f => f());
  await sleep(2500);

  console.log = (...a) => process.stdout.write(a.join(' ') + '\n');
  console.log.warn = null;

  const P = inputEl.placeholder;
  console.log('场景: ' + SCEN);
  console.log('  顶层异常  : ' + (threw ? threw.message : '无'));
  console.log('  注入 script: ' + injected.length + (injected.length ? ' -> ' + injected[0].split('/').pop() : ''));
  console.log('  输入框值  : ' + JSON.stringify(inputEl.value));
  console.log('  placeholder: ' + JSON.stringify(P));
  console.log('  title      : ' + JSON.stringify(inputEl.title));
  const ok = /识别失败|识别异常/.test(P) ? (P.includes('[') ? '✓ 含具体原因' : '✗ 缺少原因') : '· 无失败提示（可能成功）';
  console.log('  判定       : ' + ok);
  console.log('  相关日志:');
  logs.filter(l => /WARN|ERR/.test(l)).slice(0, 5).forEach(l => console.log('    ' + l.slice(0, 200)));
  process.exit(0);
})();
