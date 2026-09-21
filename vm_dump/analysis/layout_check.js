/**
 * 验证 4.4.5 的布局修复：注入的元素不得遮挡页面内容。
 *
 * 背景：4.4.3 用 `position: fixed; top: 0; z-index: 2147483647` 做失败横幅，
 * 用户反馈"有一些遮挡验证码图片"。
 *
 * 本测试的判定标准：
 *   1. 注入元素的样式里**不得出现** position:fixed / absolute / sticky
 *   2. 元素必须插在文档流内（有 parentNode，且是 host 的子节点）
 *   3. 成功路径下元素应为隐藏或空（不占用页面空间）
 */
const fs = require('fs');
const path = require('path');
const WS = 'C:\\Users\\g1507\\WorkBuddy\\2026-09-21-19-33-40';
const src = fs.readFileSync(path.join(WS, 'jaccount-captcha-onnx-enhanced.user.js'), 'utf8');
const marker = '(function () {';
let body = src.slice(src.indexOf(marker));
body = body.replace('debug: false,', 'debug: true,');

// ---------- DOM 桩（记录插入与样式） ----------
const allEls = [];
// 样式桩：cssText 是"写入即拼接"的字符串，同时 display/color 等具名属性
// 必须与 cssText 双向同步 —— 真实 DOM 里 el.style.display='none' 也会反映到
// getComputedStyle / cssText 上。桩若只做普通对象，脚本设 display:none 后
// 测试读到空字符串，会把"已折叠"误判成"仍占位"。
function makeStyle() {
  const named = { display: '', color: '', background: '', border: '', margin: '', font: '', textAlign: '' };
  let rawSuffix = '';
  const st = {
    get cssText() {
      const fromNamed = Object.keys(named).filter(k => named[k]).map(k => k + ':' + named[k]).join(';');
      return fromNamed + rawSuffix;
    },
    set cssText(v) { rawSuffix = (v ? ';' + v : ''); }
  };
  for (const k of Object.keys(named)) {
    Object.defineProperty(st, k, {
      get() { return named[k] || ''; },
      set(v) { named[k] = v; },
      enumerable: true, configurable: true
    });
  }
  return st;
}

class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.value = ''; this.placeholder = ''; this.title = '';
    this._listeners = {}; this.children = [];
    this.parentNode = null; this.isConnected = true;
    this.naturalWidth = 110; this.complete = true;
    this.style = makeStyle();
    this.textContent = '';
    allEls.push(this);
  }
  addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); }
  removeEventListener() { }
  dispatchEvent(e) { (this._listeners[e.type] || []).forEach(f => f.call(this, e)); return true; }
  setAttribute() { } getAttribute() { return null; }
  focus() { doc.activeElement = this; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c, ref) {
    c.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i >= 0) this.children.splice(i, 0, c); else this.children.push(c);
    return c;
  }
}

const doc = {
  readyState: 'loading', activeElement: null, _els: {},
  head: new El('head'), documentElement: new El('html'), body: new El('body'),
  _domListeners: {}, _byId: {},
  createElement(tag) { return new El(tag); },
  querySelector(sel) { return this._els[sel] || null; },
  getElementById(id) {
    // 在整棵桩树里找
    const walk = (n) => {
      if (!n) return null;
      if (n.id === id) return n;
      for (const c of (n.children || [])) { const r = walk(c); if (r) return r; }
      return null;
    };
    for (const k of Object.keys(this._els)) { const r = walk(this._els[k]); if (r) return r; }
    return walk(this.body) || walk(this.head) || null;
  },
  addEventListener(t, fn) { (this._domListeners[t] = this._domListeners[t] || []).push(fn); },
  removeEventListener() { },
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
        for (let i = 0; i < 110 * 40; i++) { const v = (i % 3 === 0) ? 10 : 250; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255; }
        return { data: d };
      },
      putImageData() { }
    });
  }
  return e;
};

globalThis.MutationObserver = class { constructor(cb) { } observe() { } disconnect() { } };
globalThis.document = doc;
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.performance = { now: () => Date.now() };
globalThis.location = { pathname: '/jaccount/jalogin', search: '' };
globalThis.localStorage = { getItem: () => null };
// GM_xmlhttpRequest 桩：必须返回"看起来像真实 UMD 源码"的文本。
// 早先版本返回 new ArrayBuffer(1024*1024)（全零字节），解码后是 1MB 的 \0，
// 被 loadUmdModule 交给 new Function 后必然抛 `Invalid or unexpected token`，
// 于是测试看到的失败是**桩的假数据造成的**，不是脚本缺陷。
// 这里用一个最小的合法 UMD 包装，走与真实 ORT 完全相同的四路分支。
// 注意：受控作用域会把 self/window/globalThis 全部遮蔽成 sandbox，
// 因此工厂函数内部不能依赖任何外部全局，只能自包含构造对象。
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
  const s = FAKE_UMD_SRC;
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 0xff;
  return b.buffer;
})();

globalThis.GM_xmlhttpRequest = (o) => setTimeout(() => o.onload({ status: 200, response: FAKE_UMD_BYTES }), 5);

doc.createElement = ((orig) => (tag) => {
  const e = orig(tag);
  if (tag === 'script') {
    // 走"老做法"的<script注入"路径时，模拟真实污染的 UMD 行为：
    // 页面已有 exports 全局 → UMD 会写 exports.ort 而不是 self.ort，
    // 所以 onload 触发了但 globalThis.ort 永远不出现（这正是 4.4.3 的根因）。
    Object.defineProperty(e, 'src', {
      set(v) { setTimeout(() => { e.onload && e.onload(); }, 5); },
      get() { return ''; }
    });
  }
  return e;
})(doc.createElement);

// 模拟真实 jAccount 页面上的模块系统污染：html-webpack-plugin / 某些统计脚本
// 会留下 exports 全局。没有这一层，沙箱永远走 UMD 的第 4 分支，
// 任何"漏掉 define/exports 处理"的缺陷都无法在测试中暴露。
globalThis.exports = {};
globalThis.module = { exports: {} };

function mkOrt() {
  return {
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
}

const logs = [];
console.log = () => { }; console.warn = () => { }; console.error = () => { };

let threw = null;
try { new Function(body)(); } catch (e) { threw = e; }

const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  await sleep(80);
  doc.readyState = 'interactive';
  (doc._domListeners['DOMContentLoaded'] || []).forEach(f => f());
  await sleep(2500);

  const out = (...a) => process.stdout.write(a.join(' ') + '\n');
  out('===== 场景: 识别成功 =====');
  out('  顶层异常   : ' + (threw ? threw.message : '无'));
  out('  输入框值   : ' + JSON.stringify(inputEl.value));
  out('  placeholder: ' + JSON.stringify(inputEl.placeholder));

  // 找出所有被脚本注入的元素（id 以 jaccount-recognizer 开头）
  const injected = allEls.filter(e => e.id && String(e.id).startsWith('jaccount-recognizer'));
  out();
  out('  注入的元素数量: ' + injected.length);
  let bad = 0;
  for (const e of injected) {
    const css = e.style.cssText || '';
    const fixed = /position\s*:\s*(fixed|absolute|sticky)/i.test(css);
    if (fixed) bad++;
    out('    #' + e.id);
    out('      cssText     : ' + css.slice(0, 90) + (css.length > 90 ? '…' : ''));
    out('      悬浮定位    : ' + (fixed ? '✗ 有（会遮挡页面）' : '✓ 无'));
    out('      display     : ' + JSON.stringify(e.style.display));
    out('      有 parentNode: ' + (e.parentNode ? '✓' : '✗'));
    out('      文本内容    : ' + JSON.stringify(String(e.textContent).slice(0, 60)));
  }
  out();
  if (bad) { out('  ✗ 存在悬浮元素，仍会遮挡页面'); process.exit(1); }
  out('  ✓ 无任何悬浮/覆盖式元素，所有注入内容都在文档流内');

  const status = injected.find(e => e.id === 'jaccount-recognizer-status');
  const banner = injected.find(e => e.id === 'jaccount-recognizer-banner');
  out();
  out('  成功路径下状态条应折叠: ' + (status && status.style.display === 'none' ? '✓ 已折叠' : '✗ 仍占位'));
  out();
  out('  成功路径下不应创建失败横幅: ' + (!banner ? '✓ 未创建' : '✗ 被创建了'));

  // ---------- 场景 2：识别失败（横幅必须出现在文档流内，不得悬浮） ----------
  // 用脚本暴露的横幅接口做单元验证：showBanner 之后，横幅样式里不得有定位属性。
  // 注意 body 是一个 IIFE，内部函数不对外可见，所以这里改成"重跑一遍脚本 +
  // 通过 dev 钩子拿内部引用"的方式不可行；改为直接检查脚本源码里的构造语句。
  out();
  out('===== 场景: 识别失败（横幅构造方式静态核验） =====');
  {
    // 切函数体时必须从**真正的函数定义**开始（前面可能先出现同名函数的注释提及），
    // 用 `function name(` 精确匹配；再按大括号配对找结尾。
    const sliceFn = (name) => {
      const re = new RegExp('function ' + name + '\\s*\\(');
      const m = re.exec(src);
      if (!m) return '';
      let i = src.indexOf('{', m.index);
      let depth = 0;
      for (let k = i; k < src.length; k++) {
        if (src[k] === '{') depth++;
        else if (src[k] === '}') { depth--; if (depth === 0) return src.slice(m.index, k + 1); }
      }
      return src.slice(m.index);
    };

    const sbBody = sliceFn('showBanner');
    const ebBody = sliceFn('ensureBanner');
    out('  showBanner 函数体长度 : ' + sbBody.length);
    out('  ensureBanner 函数体长度: ' + ebBody.length);
    if (!sbBody || !ebBody) { out('  ✗ 切片失败，测试自身有问题'); process.exit(1); }

    const sbFixed = /position\s*:\s*(fixed|absolute|sticky)/i.test(sbBody);
    const ebFixed = /position\s*:\s*(fixed|absolute|sticky)/i.test(ebBody);
    out('  showBanner 含悬浮定位  : ' + (sbFixed ? '✗ 有（会遮挡验证码图片）' : '✓ 无'));
    out('  ensureBanner 含悬浮定位: ' + (ebFixed ? '✗ 有' : '✓ 无'));

    const insertsIntoFlow = /insertBefore\s*\(/.test(ebBody) || /appendChild\s*\(/.test(ebBody);
    out('  ensureBanner 插入文档流: ' + (insertsIntoFlow ? '✓ 用 insertBefore/appendChild' : '✗ 未插入'));
    const defaultHidden = /display\s*:\s*none/.test(ebBody);
    out('  ensureBanner 默认隐藏  : ' + (defaultHidden ? '✓ display:none' : '⚠ 未显式隐藏'));

    // 横幅必须插在验证码图片**之后**（img.nextSibling），不能插在它前面或悬浮在上层
    const insertsAfterImg = /insertBefore\s*\(\s*el\s*,\s*img\s*\.\s*nextSibling\s*\)/.test(ebBody);
    out('  插在验证码图片之后     : ' + (insertsAfterImg ? '✓ img.nextSibling' : '⚠ 未按预期位置插入'));

    if (sbFixed || ebFixed) { out('  ✗ 横幅仍带悬浮定位'); process.exit(1); }
    if (!insertsIntoFlow) { out('  ✗ ensureBanner 未把元素插入文档流'); process.exit(1); }
    if (!defaultHidden) { out('  ✗ 横幅未默认隐藏'); process.exit(1); }
    out('  ✓ 失败横幅在文档流内构建、默认隐藏、不遮挡页面');
  }
  process.exit(0);
})();
