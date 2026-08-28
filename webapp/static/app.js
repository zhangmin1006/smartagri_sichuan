/* app.js -- 情景设定界面与结果呈现 */
(function () {
  'use strict';

  var S = null;            // 由 /api/schema 返回的参数结构
  var STATE = {            // 用户当前的全部设定
    overrides: { population: {}, counties: {}, risk: {}, behaviour: {},
                 technologies: {}, shocks: {} },
    instruments: {},       // {P1: {subsidy_rate: 0.4, ...}}
    enabled: {},           // {P1: true}
    forced: [],            // [{season: 5, shock: 'D3'}]
    seed: 20260825,
    replicates: 1,
    compare_baseline: true
  };
  var CURRENT_JOB = null;
  var LAST_RESULT = null;
  var CHART_KEYS = ['mitigation_rate', 'effective_use_rate', 'mean_wait_days',
                    'mean_income'];

  var $ = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ================= 通用控件 ================= */

  // 数值字段：滑块 + 数字框联动。onChange 收到 null 表示「恢复默认」。
  function numberField(f, current, onChange) {
    var wrap = document.createElement('div');
    wrap.className = 'field';
    var val = (current === undefined || current === null) ? f.default : current;
    var step = f.step || 0.01;
    var dec = String(step).indexOf('.') >= 0 ? String(step).split('.')[1].length : 0;

    wrap.innerHTML =
      '<div class="flabel"><span>' + esc(f.label) + '</span>' +
      '<span class="fval"></span></div>' +
      '<div class="frow">' +
      '<input type="range" min="' + f.min + '" max="' + f.max + '" step="' + step + '">' +
      '<input type="number" min="' + f.min + '" max="' + f.max + '" step="' + step + '">' +
      '</div>' +
      (f.help ? '<div class="fhelp">' + esc(f.help) + '</div>' : '');

    var range = wrap.querySelector('input[type=range]');
    var num = wrap.querySelector('input[type=number]');
    var badge = wrap.querySelector('.fval');

    function paint(v) {
      range.value = v;
      num.value = Number(v).toFixed(dec);
      badge.textContent = Number(v).toFixed(dec);
      var changed = Math.abs(Number(v) - Number(f.default)) > step / 1000;
      badge.classList.toggle('changed', changed);
      badge.title = changed ? '默认值 ' + f.default + '（双击恢复）' : '默认值';
    }

    function commit(v) {
      v = Math.max(f.min, Math.min(f.max, Number(v)));
      if (!isFinite(v)) v = f.default;
      paint(v);
      onChange(v);
    }

    range.addEventListener('input', function () { commit(range.value); });
    num.addEventListener('change', function () { commit(num.value); });
    badge.addEventListener('dblclick', function () {
      paint(f.default); onChange(null);
    });
    paint(val);
    return wrap;
  }

  function boolField(f, current, onChange) {
    var wrap = document.createElement('div');
    wrap.className = 'field';
    var on = current === undefined || current === null ? !!f.default : !!current;
    wrap.innerHTML =
      '<label class="switch"><input type="checkbox"' + (on ? ' checked' : '') +
      '><span>' + esc(f.label) + '</span></label>' +
      (f.help ? '<div class="fhelp">' + esc(f.help) + '</div>' : '');
    wrap.querySelector('input').addEventListener('change', function (e) {
      onChange(e.target.checked);
    });
    return wrap;
  }

  function selectField(f, current, onChange) {
    var wrap = document.createElement('div');
    wrap.className = 'field';
    var val = current === undefined || current === null ? f.default : current;
    var opts = (f.options || []).map(function (o) {
      return '<option value="' + esc(o.value) + '"' +
        (String(o.value) === String(val) ? ' selected' : '') + '>' +
        esc(o.label) + '</option>';
    }).join('');
    wrap.innerHTML =
      '<div class="flabel"><span>' + esc(f.label) + '</span></div>' +
      '<select>' + opts + '</select>' +
      (f.help ? '<div class="fhelp">' + esc(f.help) + '</div>' : '');
    wrap.querySelector('select').addEventListener('change', function (e) {
      onChange(e.target.value);
    });
    return wrap;
  }

  function field(f, current, onChange) {
    if (f.type === 'bool') return boolField(f, current, onChange);
    if (f.type === 'select') return selectField(f, current, onChange);
    return numberField(f, current, onChange);
  }

  // 可折叠分组
  function group(opts) {
    var g = document.createElement('div');
    g.className = 'group' + (opts.open ? ' open' : '');
    var head = document.createElement('div');
    head.className = 'grouphead';
    head.innerHTML =
      '<span class="chev">&#9654;</span>' +
      '<div class="gtitle"><div class="gname">' +
      (opts.id ? '<span class="gid">' + esc(opts.id) + '</span>' : '') +
      esc(opts.name) + '</div>' +
      (opts.meta ? '<div class="gmeta">' + esc(opts.meta) + '</div>' : '') +
      '</div>';
    if (opts.toggle) {
      var lab = document.createElement('label');
      lab.className = 'switch';
      lab.style.flexShrink = '0';
      lab.innerHTML = '<input type="checkbox"' + (opts.on ? ' checked' : '') + '>';
      lab.addEventListener('click', function (e) { e.stopPropagation(); });
      lab.querySelector('input').addEventListener('change', function (e) {
        opts.onToggle(e.target.checked);
        if (e.target.checked) g.classList.add('open');
      });
      head.appendChild(lab);
    }
    head.addEventListener('click', function () { g.classList.toggle('open'); });
    var body = document.createElement('div');
    body.className = 'groupbody';
    g.appendChild(head);
    g.appendChild(body);
    return { root: g, body: body };
  }

  /* ================= 各面板渲染 ================= */

  function renderInstruments() {
    var host = $('instrumentList');
    host.innerHTML = '';
    S.instruments.forEach(function (inst) {
      var g = group({
        id: inst.id, name: inst.name_zh, meta: inst.equity_flag,
        toggle: true, on: !!STATE.enabled[inst.id], open: !!STATE.enabled[inst.id],
        onToggle: function (on) {
          STATE.enabled[inst.id] = on;
          if (on) {
            if (!STATE.instruments[inst.id]) STATE.instruments[inst.id] = {};
          } else {
            delete STATE.instruments[inst.id];
          }
          $('presetSelect').value = '';
        }
      });

      if (inst.mechanism) {
        var m = document.createElement('div');
        m.className = 'mech';
        m.innerHTML = '<b>作用机制：</b>' + esc(inst.mechanism);
        if (inst.side_effects && inst.side_effects.length) {
          m.innerHTML += '<ul class="sidefx">' + inst.side_effects.map(function (s) {
            return '<li>' + esc(s) + '</li>';
          }).join('') + '</ul>';
        }
        g.body.appendChild(m);
      }

      inst.variables.forEach(function (v) {
        var cur = (STATE.instruments[inst.id] || {})[v.key];
        g.body.appendChild(field(v, cur, function (nv) {
          if (!STATE.instruments[inst.id]) STATE.instruments[inst.id] = {};
          if (nv === null) delete STATE.instruments[inst.id][v.key];
          else STATE.instruments[inst.id][v.key] = nv;
          // 改动决策变量即视为启用该工具，否则设定会被静默忽略
          if (!STATE.enabled[inst.id]) {
            STATE.enabled[inst.id] = true;
            renderInstruments();
          }
          $('presetSelect').value = '';
        }));
      });
      host.appendChild(g.root);
    });
  }

  function renderShocks() {
    var host = $('shockList');
    host.innerHTML = '';
    S.shocks.forEach(function (d) {
      var g = group({ id: d.id, name: d.name_zh, meta: d.name_en, open: d.id === 'D3' });
      if (d.why) {
        var m = document.createElement('div');
        m.className = 'mech';
        m.innerHTML = '<b>为何纳入：</b>' + esc(d.why);
        g.body.appendChild(m);
      }
      d.fields.forEach(function (f) {
        var parts = f.key.split('.');
        var cur = (STATE.overrides.shocks[parts[0]] || {})[parts[1]];
        g.body.appendChild(field(f, cur, function (v) {
          setNested(STATE.overrides.shocks, parts[0], parts[1], v);
        }));
      });
      host.appendChild(g.root);
    });
    renderForced();
  }

  function renderForced() {
    var host = $('forcedRows');
    host.innerHTML = '';
    STATE.forced.forEach(function (row, i) {
      var div = document.createElement('div');
      div.className = 'forcedrow';
      var opts = S.shocks.map(function (d) {
        return '<option value="' + d.id + '"' +
          (d.id === row.shock ? ' selected' : '') + '>' +
          d.id + ' ' + esc(d.name_zh) + '</option>';
      }).join('');
      div.innerHTML =
        '<span style="font-size:12px;color:#5b6b7c">第</span>' +
        '<input type="number" min="1" max="30" step="1" value="' + row.season + '">' +
        '<span style="font-size:12px;color:#5b6b7c">季</span>' +
        '<select>' + opts + '</select>' +
        '<button type="button" class="rmbtn">×</button>';
      div.querySelector('input').addEventListener('change', function (e) {
        row.season = Math.max(1, parseInt(e.target.value, 10) || 1);
      });
      div.querySelector('select').addEventListener('change', function (e) {
        row.shock = e.target.value;
      });
      div.querySelector('.rmbtn').addEventListener('click', function () {
        STATE.forced.splice(i, 1); renderForced();
      });
      host.appendChild(div);
    });
  }

  function renderTech() {
    var host = $('techList');
    host.innerHTML = '';
    S.technologies.forEach(function (t) {
      var g = group({ id: t.id, name: t.name_zh, meta: t.channel });
      if (t.mechanism) {
        var m = document.createElement('div');
        m.className = 'mech';
        m.innerHTML = '<b>韧性机制：</b>' + esc(t.mechanism);
        g.body.appendChild(m);
      }
      t.fields.forEach(function (f) {
        var parts = f.key.split('.');
        var cur = (STATE.overrides.technologies[parts[0]] || {})[parts[1]];
        g.body.appendChild(field(f, cur, function (v) {
          setNested(STATE.overrides.technologies, parts[0], parts[1], v);
        }));
      });
      host.appendChild(g.root);
    });
  }

  function renderFarmer() {
    var pop = $('popList'); pop.innerHTML = '';
    var h = document.createElement('div');
    h.className = 'subhead'; h.textContent = '人口与规模';
    pop.appendChild(h);
    S.population.fields.forEach(function (f) {
      pop.appendChild(field(f, STATE.overrides.population[f.key], function (v) {
        if (v === null) delete STATE.overrides.population[f.key];
        else STATE.overrides.population[f.key] = v;
      }));
    });

    var risk = $('riskList'); risk.innerHTML = '';
    var h2 = document.createElement('div');
    h2.className = 'subhead'; h2.textContent = '风险态度分布';
    risk.appendChild(h2);
    S.population.risk.forEach(function (f) {
      risk.appendChild(field(f, STATE.overrides.risk[f.key], function (v) {
        if (v === null) delete STATE.overrides.risk[f.key];
        else STATE.overrides.risk[f.key] = v;
      }));
    });

    var beh = $('behavList'); beh.innerHTML = '';
    var h3 = document.createElement('div');
    h3.className = 'subhead'; h3.textContent = '契约与行为参数';
    beh.appendChild(h3);
    S.population.behaviour.forEach(function (f) {
      beh.appendChild(field(f, STATE.overrides.behaviour[f.key], function (v) {
        if (v === null) delete STATE.overrides.behaviour[f.key];
        else STATE.overrides.behaviour[f.key] = v;
      }));
    });

    var cty = $('countyList'); cty.innerHTML = '';
    var h4 = document.createElement('div');
    h4.className = 'subhead'; h4.textContent = '县域结构（六个代表性县）';
    cty.appendChild(h4);
    var note = document.createElement('div');
    note.className = 'fhelp';
    note.style.marginBottom = '10px';
    note.textContent = '调整任一县的农户数后，总农户数将自动等于各县之和。';
    cty.appendChild(note);
    S.population.counties.forEach(function (c) {
      var g = group({ id: c.id, name: c.label, meta: c.terrain });
      c.fields.forEach(function (f) {
        var parts = f.key.split('.');
        var cur = (STATE.overrides.counties[parts[0]] || {})[parts[1]];
        g.body.appendChild(field(f, cur, function (v) {
          setNested(STATE.overrides.counties, parts[0], parts[1], v);
        }));
      });
      cty.appendChild(g.root);
    });
  }

  function renderRun() {
    var host = $('runList'); host.innerHTML = '';
    host.appendChild(field(
      { label: '随机种子', type: 'number', min: 1, max: 99999999, step: 1,
        default: S.default_seed,
        help: '相同种子给出完全可重现的结果。政策情景与基准情景共用种子，因此比较是配对的' },
      STATE.seed, function (v) { STATE.seed = v === null ? S.default_seed : v; }));

    host.appendChild(field(
      { label: '重复次数', type: 'number', min: 1, max: 10, step: 1, default: 1,
        help: '用连续种子重复整个配对实验并取平均。单次运行的结果很大程度上取决于'
            + '灾害是否恰好落在模拟期内；重复 3 次以上才能判断差异是否真实' },
      STATE.replicates, function (v) { STATE.replicates = v === null ? 1 : v; }));

    host.appendChild(field(
      { label: '同时运行基准情景以作比较', type: 'bool', default: true,
        help: '关闭后只运行政策情景，速度加倍，但无法判断效应大小' },
      STATE.compare_baseline, function (v) { STATE.compare_baseline = v; }));
  }

  function setNested(obj, a, b, v) {
    if (v === null) {
      if (obj[a]) { delete obj[a][b]; if (!Object.keys(obj[a]).length) delete obj[a]; }
      return;
    }
    if (!obj[a]) obj[a] = {};
    obj[a][b] = v;
  }

  /* ================= 运行与轮询 ================= */

  function runSimulation() {
    var forced = {};
    STATE.forced.forEach(function (r) {
      if (!forced[r.season]) forced[r.season] = [];
      if (forced[r.season].indexOf(r.shock) < 0) forced[r.season].push(r.shock);
    });

    var payload = {
      overrides: STATE.overrides,
      instruments: STATE.instruments,
      forced_shocks: forced,
      seed: STATE.seed,
      replicates: STATE.replicates,
      compare_baseline: STATE.compare_baseline
    };

    $('placeholder').classList.add('hidden');
    $('resultBox').classList.add('hidden');
    $('errorBox').classList.add('hidden');
    $('progressBox').classList.remove('hidden');
    $('btnRun').disabled = true;
    $('btnRun').textContent = '运行中…';
    setProgress(0.02, '提交任务');

    fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
        return j;
      });
    }).then(function (j) {
      CURRENT_JOB = j.job_id;
      poll(j.job_id);
    }).catch(function (e) { fail(e.message); });
  }

  function poll(id) {
    fetch('/api/job/' + id).then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.error && j.status === undefined) throw new Error(j.error);
        setProgress(j.progress || 0, j.stage || '运行中');
        if (j.status === 'done') { finish(j.result); return; }
        if (j.status === 'error') { fail(j.error); return; }
        setTimeout(function () { poll(id); }, 700);
      }).catch(function (e) { fail(e.message); });
  }

  function setProgress(p, stage) {
    $('progBar').style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + '%';
    $('progStage').textContent = stage;
  }

  function fail(msg) {
    $('progressBox').classList.add('hidden');
    $('errorBox').classList.remove('hidden');
    $('errorBox').innerHTML = '<strong>运行失败</strong><br>' + esc(msg);
    $('btnRun').disabled = false;
    $('btnRun').textContent = '运行仿真';
  }

  /* ================= 结果呈现 ================= */

  function finish(res) {
    LAST_RESULT = res;
    $('progressBox').classList.add('hidden');
    $('resultBox').classList.remove('hidden');
    $('btnRun').disabled = false;
    $('btnRun').textContent = '运行仿真';

    var names = Object.keys(res.instruments || {});
    $('resultMeta').textContent =
      '种子 ' + res.seed + ' · 重复 ' + res.replicates + ' 次 · ' +
      (names.length ? '启用工具 ' + names.join(', ') : '未启用任何政策工具') +
      ' · 缓存 ' + (res.cache ? res.cache.entries : '?') + ' 条';

    renderKpis(res);
    renderChartControls();
    renderCharts(res);
    renderTable(res);
  }

  function fmtBy(key, v) {
    var o = objFor(key);
    return Charts.fmt(v, o ? o.fmt : null);
  }

  function objFor(key) {
    for (var i = 0; i < S.objectives.length; i++) {
      if (S.objectives[i].key === key) return S.objectives[i];
    }
    return null;
  }

  function renderKpis(res) {
    var host = $('kpiGrid');
    host.innerHTML = '';
    var cmp = res.comparison;

    if (!cmp) {
      host.innerHTML = '<div class="kpi"><div class="klabel">未运行基准情景</div>' +
        '<div class="kvals">开启「同时运行基准情景」后，此处显示政策效应</div></div>';
      return;
    }

    S.objectives.forEach(function (o) {
      var c = cmp[o.key];
      if (!c) return;
      var better = o.direction === 'max' ? c.diff > 0 : c.diff < 0;
      var flat = c.distinguishable === false;
      var cls = flat ? 'flat' : (better ? 'up' : 'down');
      var sign = c.diff > 0 ? '+' : '';
      var relTxt = (c.rel === null || c.rel === undefined || !isFinite(c.rel))
        ? '' : '（' + (c.rel > 0 ? '+' : '') + (c.rel * 100).toFixed(1) + '%）';

      var d = document.createElement('div');
      d.className = 'kpi';
      d.innerHTML =
        '<div class="klabel">' + esc(o.label) +
        '<span class="arrowhint">' + (o.direction === 'max' ? '越高越好' : '越低越好') +
        '</span></div>' +
        '<div class="kdelta ' + cls + '">' + sign + fmtBy(o.key, c.diff) +
        '<span style="font-size:11px;font-weight:400;color:#8a99a8"> ' +
        relTxt + '</span></div>' +
        '<div class="kvals">政策 ' + fmtBy(o.key, c.policy) +
        ' ← 基准 ' + fmtBy(o.key, c.baseline) + '</div>' +
        (flat ? '<div class="knoise">与随机波动无法区分，建议提高重复次数</div>' : '');
      d.title = o.help || '';
      host.appendChild(d);
    });
  }

  function renderChartControls() {
    var host = $('chartControls');
    host.innerHTML = '';
    var choices = [
      'mitigation_rate', 'effective_use_rate', 'mean_wait_days', 'mean_income',
      'mean_loss_fraction', 'adopt_T2', 'adopt_T3', 'fiscal_cumulative',
      'equity_gap', 'mountain_gap', 'backlog_mu', 'exit_rate',
      'capacity_units', 'trust', 'gini_income', 'income_p10'
    ];
    choices.forEach(function (k) {
      var b = document.createElement('button');
      b.className = 'chip' + (CHART_KEYS.indexOf(k) >= 0 ? ' on' : '');
      b.type = 'button';
      b.textContent = S.metrics[k] || k;
      b.addEventListener('click', function () {
        var i = CHART_KEYS.indexOf(k);
        if (i >= 0) CHART_KEYS.splice(i, 1); else CHART_KEYS.push(k);
        b.classList.toggle('on');
        renderCharts(LAST_RESULT);
      });
      host.appendChild(b);
    });
  }

  function renderCharts(res) {
    var host = $('chartGrid');
    host.innerHTML = '';
    if (!res) return;
    var pol = res.policy.series;
    var base = res.baseline ? res.baseline.series : null;

    CHART_KEYS.forEach(function (k) {
      if (!pol[k]) return;
      var card = document.createElement('div');
      card.className = 'chartcard';
      var o = objFor(k);
      card.innerHTML = '<h4>' + esc(S.metrics[k] || k) + '</h4>' +
        '<div class="csub">' + esc(o && o.help ? o.help : '逐季演化') + '</div>';
      var holder = document.createElement('div');
      card.appendChild(holder);

      Charts.lineChart(holder, {
        x: pol.season, policy: pol[k], baseline: base ? base[k] : null,
        shocks: pol.shocks, style: o ? o.fmt : null
      });

      var lg = document.createElement('div');
      lg.className = 'legend';
      lg.innerHTML =
        '<span><i style="background:' + Charts.COL.policy + '"></i>政策情景</span>' +
        (base ? '<span><i style="background:' + Charts.COL.baseline +
                '"></i>基准情景</span>' : '') +
        (pol.shocks ? '<span><i style="background:' + Charts.COL.shock +
                      '"></i>发生冲击的季度</span>' : '');
      card.appendChild(lg);
      host.appendChild(card);
    });

    if (!host.children.length) {
      host.innerHTML = '<div class="chartcard"><div class="csub">' +
        '请在上方选择至少一个指标</div></div>';
    }
  }

  function renderTable(res) {
    var t = $('cmpTable');
    var cmp = res.comparison;
    var keys = Object.keys(res.policy.summary);

    var head = '<thead><tr><th>指标</th><th>政策情景</th>' +
      (cmp ? '<th>基准情景</th><th>差异</th><th>相对变化</th>' : '') +
      '<th>重复间标准差</th></tr></thead>';
    var rows = keys.map(function (k) {
      var p = res.policy.summary[k];
      if (!p || p.mean === null) return '';
      var c = cmp ? cmp[k] : null;
      var cells = '<td>' + esc(S.metrics[k] || k) + '</td>' +
        '<td>' + fmtBy(k, p.mean) + '</td>';
      if (cmp) {
        if (c) {
          var o = objFor(k);
          var cls = '';
          if (o) cls = (o.direction === 'max' ? c.diff > 0 : c.diff < 0) ? 'pos' : 'neg';
          if (c.distinguishable === false) cls = 'dim';
          cells += '<td>' + fmtBy(k, c.baseline) + '</td>' +
            '<td class="' + cls + '">' + (c.diff > 0 ? '+' : '') +
            fmtBy(k, c.diff) + '</td>' +
            '<td class="' + cls + '">' +
            ((c.rel === null || c.rel === undefined || !isFinite(c.rel)) ? '—' :
              (c.rel > 0 ? '+' : '') + (c.rel * 100).toFixed(1) + '%') + '</td>';
        } else {
          cells += '<td>—</td><td>—</td><td>—</td>';
        }
      }
      cells += '<td class="dim">' + (p.sd === null ? '—' : fmtBy(k, p.sd)) + '</td>';
      return '<tr>' + cells + '</tr>';
    }).join('');
    t.innerHTML = head + '<tbody>' + rows + '</tbody>';
  }

  /* ================= 预设与重置 ================= */

  function applyPreset(key) {
    STATE.instruments = {};
    STATE.enabled = {};
    var p = null;
    S.presets.forEach(function (x) { if (x.key === key) p = x; });
    if (p) {
      Object.keys(p.instruments).forEach(function (id) {
        STATE.instruments[id] = Object.assign({}, p.instruments[id]);
        STATE.enabled[id] = true;
      });
    }
    renderInstruments();
  }

  function resetAll() {
    STATE.overrides = { population: {}, counties: {}, risk: {}, behaviour: {},
                        technologies: {}, shocks: {} };
    STATE.instruments = {};
    STATE.enabled = {};
    STATE.forced = [];
    STATE.seed = S.default_seed;
    STATE.replicates = 1;
    STATE.compare_baseline = true;
    $('presetSelect').value = '';
    renderAll();
  }

  function renderAll() {
    renderInstruments(); renderShocks(); renderTech();
    renderFarmer(); renderRun();
  }

  /* ================= 启动 ================= */

  function init() {
    fetch('/api/schema').then(function (r) { return r.json(); })
      .then(function (schema) {
        S = schema;
        STATE.seed = S.default_seed;
        $('verBadge').textContent = '模型 v' + S.model_version;

        var sel = $('presetSelect');
        S.presets.forEach(function (p) {
          var o = document.createElement('option');
          o.value = p.key; o.textContent = p.label;
          sel.appendChild(o);
        });
        sel.addEventListener('change', function (e) { applyPreset(e.target.value); });

        renderAll();
      }).catch(function (e) {
        $('placeholder').innerHTML =
          '<h2 style="color:#b23c2e">无法载入参数结构</h2><p>' + esc(e.message) + '</p>';
      });

    document.querySelectorAll('.tab').forEach(function (b) {
      b.addEventListener('click', function () {
        document.querySelectorAll('.tab').forEach(function (x) {
          x.classList.remove('active');
        });
        document.querySelectorAll('.tabpane').forEach(function (x) {
          x.classList.remove('active');
        });
        b.classList.add('active');
        $('pane-' + b.dataset.tab).classList.add('active');
      });
    });

    $('btnRun').addEventListener('click', runSimulation);
    $('btnReset').addEventListener('click', resetAll);
    $('addForced').addEventListener('click', function () {
      STATE.forced.push({ season: STATE.forced.length + 3, shock: 'D3' });
      renderForced();
    });
    $('btnCsv').addEventListener('click', function () {
      if (CURRENT_JOB) window.location = '/api/job/' + CURRENT_JOB + '/csv';
    });
    $('btnHelp').addEventListener('click', function () {
      $('helpModal').classList.remove('hidden');
    });
    $('closeHelp').addEventListener('click', function () {
      $('helpModal').classList.add('hidden');
    });
    $('helpModal').addEventListener('click', function (e) {
      if (e.target === $('helpModal')) $('helpModal').classList.add('hidden');
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
