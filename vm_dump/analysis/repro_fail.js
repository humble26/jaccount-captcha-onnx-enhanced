/**
 * 复现"全部识别失败"：在 DOM 沙箱里跑**真实的油猴版代码**（不是扩展版），
 * 包含它的 loadScript / loadOrt / cachedBytes / recognizeWithONNX / Tesseract 兜底。
 *
 * 关键：模拟 @run-at document-start 时 document.head 与 documentElement 的实际状态，
 * 并伪造 GM_xmlhttpRequest / window.ort 的加载过程。
 */
const fs = require('fs');
const path = require('path');
const WS = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40';
const src = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');

const marker = '(function () {';
let body = src.slice(src.indexOf(marker));
// 打开调试日志（真实脚本里是 CFG.debug = false）
body = body.replace('debug: false,', 'debug: true,');
if (!body.includes('debug: true,')) throw new Error('未能打开 debug 开关');

// ---------- 极简 DOM ----------
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
  dispatchEvent(e) { (this._listeners[e.type] || []).forEach(f => f.call(this, e)); return true; }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'src') this.src = v; }
  getAttribute(k) { return this.attrs[k]; }
  focus() { doc.activeElement = this; }
  appendChild(c) { this._children = this._children || []; this._children.push(c); return c; }
}

const doc = {
  readyState: 'loading',
  activeElement: null,
  _els: {},
  head: null,               // 模拟 document-start：head 还没建
  documentElement: null,    // 甚至连 documentElement 都没有
  _domListeners: {},
  createElement(tag) { return new El(tag); },
  querySelector(sel) { return this._els[sel] || null; },
  addEventListener(t, fn) { (this._domListeners[t] = this._domListeners[t] || []).push(fn); },
  removeEventListener(t, fn) {
    const a = this._domListeners[t]; if (!a) return;
    const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1);
  },
};

const imgEl = new El('img');
const inputEl = new El('input');
const userEl = new El('input');
doc._els['#captcha-img'] = imgEl;
doc._els['#input-login-captcha'] = inputEl;
doc._els['#input-login-name'] = userEl;

// canvas 桩
doc.createElement = (tag) => {
  const e = new El(tag);
  if (tag === 'canvas') {
    e.getContext = () => ({
      imageSmoothingEnabled: true,
      drawImage() { },
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

// ---------- 环境 ----------
const observers = [];
globalThis.MutationObserver = class {
  constructor(cb) { this.cb = cb; observers.push(this); }
  observe(t, o) { this.target = t; }
  disconnect() { }
};
globalThis.document = doc;
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.performance = { now: () => Date.now() };
globalThis.location = { pathname: '/jaccount/jalogin', search: '' };
globalThis.localStorage = { getItem: () => null };

// 关键：模拟 GM_xmlhttpRequest 正常工作（国内可直连 jsdelivr 的场景）
let gmCalls = [];
// 按 URL 区分返回内容 —— 这是本桩的关键。
// 早先所有 URL 一律返回 new ArrayBuffer(1024*1024)（全零字节），
// 于是 UMD 源码 = 1MB 的 \0，被 new Function 执行必然抛
// `Invalid or unexpected token`，让每次 ONNX 路径都"失败"。
// 那是桩的假数据造成的假失败，会掩盖真实的通过状态。
const FAKE_UMD_SRC = [
  '!function(e,t){',
  '  "object"==typeof exports&&"object"==typeof module?module.exports=t():',
  '  "function"==typeof define&&define.amd?define([],t):',
  '  "object"==typeof exports?exports.ort=t():e.ort=t()',
  '}(self,function(){',
  '  return {',
  '    env:{ wasm:{} },',
  '    Tensor:function Tensor(type,data,dims){ this.type=type; this.data=data; this.dims=dims; },',
  '    InferenceSession:{ create:function(){',
  '      return Promise.resolve({',
  '        inputNames:["input.1"], outputNames:["218","219","220","221","222"],',
  '        run:function(){',
  '          var mk=function(i){',
  '            var a=new Array(27); for(var k=0;k<27;k++) a[k]=(k===i?6:-9);',
  '            return { data:Float32Array.from(a), dims:[1,27] };',
  '          };',
  '          return Promise.resolve({"218":mk(0),"219":mk(1),"220":mk(2),"221":mk(3),"222":mk(26)});',
  '        },',
  '        release:function(){ return Promise.resolve(); }',
  '      });',
  '    } }',
  '  };',
  '});'
].join('\n');
const FAKE_UMD_BYTES = (() => {
  const b = new Uint8Array(FAKE_UMD_SRC.length);
  for (let i = 0; i < FAKE_UMD_SRC.length; i++) b[i] = FAKE_UMD_SRC.charCodeAt(i) & 0xff;
  return b.buffer;
})();
// 模型：真实的 1MB 二进制（内容不重要，只要长度与类型正确）
const FAKE_MODEL_BYTES = new ArrayBuffer(1024 * 1024);

globalThis.GM_xmlhttpRequest = (opts) => {
  gmCalls.push(opts.url);
  const isModel = /\.onnx(\?|$)/i.test(opts.url);
  const payload = isModel ? FAKE_MODEL_BYTES : FAKE_UMD_BYTES;
  setTimeout(() => opts.onload({ status: 200, response: payload }), 5);
};

// script 注入：模拟 <script> 真的被插入并 onload
let injected = [];
const origCreate = doc.createElement;
doc.createElement = (tag) => {
  const e = origCreate(tag);
  if (tag === 'script') {
    let _src = '';
    Object.defineProperty(e, 'src', {
      get() { return _src; },
      set(v) {
        _src = v; injected.push(v);
        // 模拟加载完成：给 window.ort 赋值
        setTimeout(() => {
          globalThis.ort = {
            env: { wasm: {} },
            Tensor: class { constructor(t, d, dims) { this.data = d; this.dims = dims; } },
            InferenceSession: {
              async create() {
                return {
                  inputNames: ['input.1'],
                  outputNames: ['218', '219', '220', '221', '222'],
                  async run() {
                    const mk = (i) => ({ data: Float32Array.from(new Array(27).fill(-9).map((v, k) => k === i ? 6 : v)), dims: [1, 27] });
                    return { '218': mk(0), '219': mk(1), '220': mk(2), '221': mk(3), '222': mk(26) };
                  },
                  async release() { }
                };
              }
            }
          };
          if (typeof e.onload === 'function') e.onload();
        }, 5);
      }
    });
  }
  return e;
};

// Tesseract 也桩掉（不该被走到；走到就说明 ONNX 失败了）
let tessCalls = 0;
globalThis.Tesseract = {
  createWorker: async () => { tessCalls++; return { setParameters: async () => { }, recognize: async () => ({ data: { text: 'zzzz', confidence: 90 } }) }; }
};

// 捕获 console
const logs = [];
const realLog = console.log, realWarn = console.warn, realErr = console.error;
console.log = (...a) => { logs.push('LOG ' + a.join(' ')); };
console.warn = (...a) => { logs.push('WARN ' + a.join(' ')); };
console.error = (...a) => { logs.push('ERR ' + a.join(' ')); };

// 打开 debug
process.env.JACC_DEBUG = '1';

// ---------- 执行脚本 ----------
const MODE = process.argv[2] || 'nostart';
if (MODE === 'normal') {
  // 正常情况：document-start 时 head 通常已由解析器建好
  doc.head = new El('head');
  doc.documentElement = new El('html');
}

let threw = null;
try {
  new Function(body)();
} catch (e) {
  threw = e;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  await sleep(100);
  // 触发 DOMContentLoaded，让 init() 真正跑起来
  doc.readyState = 'interactive';
  (doc._domListeners['DOMContentLoaded'] || []).forEach(f => f());
  await sleep(2000);   // 给足时间走完 loadScript -> 模型 -> 推理

  console.log = realLog; console.warn = realWarn; console.error = realErr;

  console.log('=== 执行结果 ===');
  console.log('  顶层抛异常:', threw ? threw.message : '无');
  console.log('  GM_xmlhttpRequest 调用:', gmCalls.length, gmCalls.map(u => u.split('/').slice(-1)[0]).join(', '));
  console.log('  注入的 <script>:', injected.length);
  injected.forEach(u => console.log('    -', u));
  console.log('  Tesseract 被调用次数:', tessCalls, tessCalls > 0 ? '  ← ⚠ 说明 ONNX 路径失败了' : '');
  console.log();
  console.log('  document.head 是否被创建:', doc.head ? '是' : '否');
  console.log('  document.documentElement 是否被创建:', doc.documentElement ? '是' : '否');
  console.log();
  console.log('  输入框最终值:', JSON.stringify(inputEl.value));
  console.log('  输入框 placeholder:', JSON.stringify(inputEl.placeholder));
  console.log();
  console.log('=== 内部日志 ===');
  logs.slice(0, 40).forEach(l => console.log('  ' + l));
  if (logs.length > 40) console.log(`  ... 共 ${logs.length} 条`);

  process.exit(0);
})();
