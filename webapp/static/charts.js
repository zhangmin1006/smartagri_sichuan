/* charts.js -- 极简 SVG 折线图
 *
 * 刻意不引入任何图表库：本平台需要能在无外网的内网或田间办公环境中运行，
 * 而通过 CDN 加载的图表库在断网时会静默失效，只留下空白画布。
 * 这里只需要「基准 vs 政策」双线对比，自行绘制即可，且完全可控。
 */
(function (global) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var COL = { policy: '#1f6f5c', baseline: '#8a99a8', shock: '#c8553d' };

  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) {
      if (attrs[k] !== null && attrs[k] !== undefined) {
        n.setAttribute(k, attrs[k]);
      }
    }
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function niceTicks(lo, hi, count) {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (hi - lo < 1e-12) { hi = lo + 1; }
    var raw = (hi - lo) / Math.max(count, 1);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
    var start = Math.floor(lo / step) * step;
    var out = [];
    for (var v = start; v <= hi + step * 0.5; v += step) out.push(v);
    return out;
  }

  function fmt(v, style) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    if (style === 'pct') return (v * 100).toFixed(1) + '%';
    if (style === 'money') {
      if (Math.abs(v) >= 1e8) return (v / 1e8).toFixed(2) + '亿';
      if (Math.abs(v) >= 1e4) return (v / 1e4).toFixed(1) + '万';
      return v.toFixed(0);
    }
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    if (Math.abs(v) >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }

  /* 绘制一张折线图。
     opts: {title, subtitle, x[], policy[], baseline[], shocks[], style} */
  function lineChart(container, opts) {
    var W = 420, H = 210,
        P = { t: 10, r: 12, b: 26, l: 52 };
    var x = opts.x || [];
    var series = [];
    if (opts.baseline) series.push({ key: 'baseline', v: opts.baseline });
    if (opts.policy) series.push({ key: 'policy', v: opts.policy });

    var vals = [];
    series.forEach(function (s) {
      s.v.forEach(function (v) { if (v !== null && isFinite(v)) vals.push(v); });
    });

    var svg = el('svg', {
      viewBox: '0 0 ' + W + ' ' + H,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img'
    });

    if (!vals.length) {
      svg.appendChild(el('text', {
        x: W / 2, y: H / 2, 'text-anchor': 'middle',
        fill: '#8a99a8', 'font-size': 12
      }, '无数据'));
      container.appendChild(svg);
      return;
    }

    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    // 让常数序列也能显示在图中央，而不是压在坐标轴上
    if (hi - lo < 1e-9) { lo -= Math.abs(lo) * 0.1 + 0.01; hi += Math.abs(hi) * 0.1 + 0.01; }
    else { var pad = (hi - lo) * 0.12; lo -= pad; hi += pad; }
    if (lo > 0 && lo / (hi - lo) < 0.35) lo = 0;   // 接近零时锚定到零

    var ticks = niceTicks(lo, hi, 4);
    lo = Math.min(lo, ticks[0]);
    hi = Math.max(hi, ticks[ticks.length - 1]);

    var iw = W - P.l - P.r, ih = H - P.t - P.b;
    var px = function (i) {
      return P.l + (x.length < 2 ? iw / 2 : iw * i / (x.length - 1));
    };
    var py = function (v) { return P.t + ih * (1 - (v - lo) / (hi - lo)); };

    // 网格与 y 轴刻度
    ticks.forEach(function (t) {
      var y = py(t);
      if (y < P.t - 1 || y > P.t + ih + 1) return;
      svg.appendChild(el('line', {
        x1: P.l, x2: P.l + iw, y1: y, y2: y,
        stroke: '#eaeef2', 'stroke-width': 1
      }));
      svg.appendChild(el('text', {
        x: P.l - 7, y: y + 3.5, 'text-anchor': 'end',
        fill: '#8a99a8', 'font-size': 9.5, 'font-family': 'Consolas, monospace'
      }, fmt(t, opts.style)));
    });

    // 发生冲击的季度加竖向标记
    (opts.shocks || []).forEach(function (s, i) {
      if (!s) return;
      svg.appendChild(el('line', {
        x1: px(i), x2: px(i), y1: P.t, y2: P.t + ih,
        stroke: COL.shock, 'stroke-width': 1, 'stroke-dasharray': '2 3',
        opacity: 0.5
      }));
    });

    // x 轴
    svg.appendChild(el('line', {
      x1: P.l, x2: P.l + iw, y1: P.t + ih, y2: P.t + ih,
      stroke: '#dde3e9', 'stroke-width': 1
    }));
    var every = Math.max(1, Math.ceil(x.length / 8));
    x.forEach(function (s, i) {
      if (i % every && i !== x.length - 1) return;
      svg.appendChild(el('text', {
        x: px(i), y: P.t + ih + 15, 'text-anchor': 'middle',
        fill: '#8a99a8', 'font-size': 9.5, 'font-family': 'Consolas, monospace'
      }, s));
    });

    // 折线
    series.forEach(function (s) {
      var d = '', pen = false;
      s.v.forEach(function (v, i) {
        if (v === null || !isFinite(v)) { pen = false; return; }
        d += (pen ? ' L' : ' M') + px(i).toFixed(1) + ',' + py(v).toFixed(1);
        pen = true;
      });
      if (!d) return;
      svg.appendChild(el('path', {
        d: d.trim(), fill: 'none', stroke: COL[s.key],
        'stroke-width': s.key === 'policy' ? 2 : 1.5,
        'stroke-dasharray': s.key === 'baseline' ? '4 3' : null,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round'
      }));
      // 末端点，便于读出终值
      for (var i = s.v.length - 1; i >= 0; i--) {
        if (s.v[i] !== null && isFinite(s.v[i])) {
          svg.appendChild(el('circle', {
            cx: px(i), cy: py(s.v[i]), r: 2.6, fill: COL[s.key]
          }));
          break;
        }
      }
    });

    container.appendChild(svg);
  }

  global.Charts = { lineChart: lineChart, fmt: fmt, COL: COL };
})(window);
