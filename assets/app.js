/* ==========================================================================
   CF²-GET demo · page logic
   Builds the example cards from config.js, loads MIDI files lazily, draws
   aligned piano rolls, and runs one shared player for the whole page.
   ========================================================================== */
(function () {
  'use strict';

  var CFG = window.CF2GET_CONFIG || {};
  var M = window.CF2MIDI;
  var GENRES = [
    { key: 'classical', label: 'Classical' }, { key: 'jazz', label: 'Jazz' }, { key: 'rock', label: 'Rock' },
    { key: 'pop', label: 'Pop' }, { key: 'country', label: 'Country' }, { key: 'latin', label: 'Latin' }
  ];
  var GENRE = {};
  GENRES.forEach(function (g) { GENRE[g.key] = g; });
  function gLabel(k) { return GENRE[k] ? GENRE[k].label : (k || '').charAt(0).toUpperCase() + (k || '').slice(1); }

  var METHOD_NOTES = {
    'source': 'The original 16-bar passage in its source genre.',
    'cf2get': 'Inverts the flow under the source genre, keeps the noise û, and regenerates under the target genre.',
    'midi-vae': 'Direct symbolic baseline. Swaps genre coordinates inside a joint VAE bottleneck.',
    'latent-midi-vae': 'The same coordinate swap, done with recurrent networks on our shared latent space.',
    'ddim': 'Diffusion baseline on the same DiT backbone: DDIM inversion, then sampling under the target genre.',
    'wo-hsic': 'Ablation: CF²-GET trained without the HSIC noise–genre independence term.'
  };

  var host = location.hostname;
  var DEV = location.protocol === 'file:' || /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|)$/.test(host) ||
    /^(10|192\.168|172\.(1[6-9]|2\d|3[01]))\./.test(host) || /[?&]dev\b/.test(location.search);

  function $(sel, root) { return (root || document).querySelector(sel); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function fmt(s) { s = Math.max(0, s || 0); var m = Math.floor(s / 60), r = Math.floor(s % 60); return m + ':' + (r < 10 ? '0' : '') + r; }
  function joinPath(folder, file) { return (folder || '').replace(/\/+$/, '') + '/' + file; }

  /* ---------------------------------------------------------------- toast */
  var toastTimer = 0;
  function toast(msg, ms) {
    var t = $('#toast');
    t.textContent = msg; t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.hidden = true; }, ms || 4200);
  }

  /* ---------------------------------------------------------------- links */
  function setupLinks() {
    var links = CFG.links || {};
    var code = (links.code || '').trim(), paper = (links.paper || '').trim();
    document.querySelectorAll('[data-link="code"]').forEach(function (a) {
      if (code) { a.href = code; return; }
      a.removeAttribute('href');
      a.setAttribute('aria-disabled', 'true');
      a.setAttribute('title', 'The code release is coming soon');
      var lab = a.querySelector('[data-link-label]');
      if (lab) lab.textContent = a.closest('.topbar') ? 'Code soon' : 'Code coming soon';
    });
    document.querySelectorAll('[data-link="paper"]').forEach(function (a) {
      if (paper) { a.href = paper; a.hidden = false; }
    });
  }

  /* ---------------------------------------------------------------- theme */
  function colors() {
    var cs = getComputedStyle(document.documentElement);
    function v(n) { return cs.getPropertyValue(n).trim(); }
    var c = { ink: v('--ink'), muted: v('--muted'), rule: v('--rule'), ruleStrong: v('--rule-strong'), surface: v('--surface'), g: {} };
    GENRES.forEach(function (g) { c.g[g.key] = v('--g-' + g.key); });
    return c;
  }
  var COL = null;
  function themeChanged() {
    COL = colors();
    allRows.forEach(function (r) { if (r.card.view) { r.render(); r.draw(r.showsHead() ? r.pos : null); } });
    window.dispatchEvent(new Event('cf2get:theme'));
  }
  function setupTheme() {
    var btn = $('#theme-toggle');
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    btn.addEventListener('click', function () {
      var cur = document.documentElement.getAttribute('data-theme') || (mq.matches ? 'dark' : 'light');
      var next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('cf2get-theme', next); } catch (e) {}
      themeChanged();
    });
    if (mq.addEventListener) mq.addEventListener('change', themeChanged); else if (mq.addListener) mq.addListener(themeChanged);
  }

  /* ---------------------------------------------------------------- MIDI loading */
  var cache = {};
  function fetchMidi(url) {
    if (cache[url]) return cache[url];
    cache[url] = (async function () {
      var resp;
      try { resp = await fetch(encodeURI(url), { cache: 'no-cache' }); }
      catch (e) { return { status: location.protocol === 'file:' ? 'blocked' : 'missing' }; }
      if (!resp.ok) return { status: 'missing', code: resp.status };
      var buf = await resp.arrayBuffer();
      var head = String.fromCharCode.apply(null, new Uint8Array(buf, 0, Math.min(4, buf.byteLength)));
      if (head !== 'MThd') {
        var first = String.fromCharCode.apply(null, new Uint8Array(buf, 0, Math.min(64, buf.byteLength))).trim();
        if (!buf.byteLength || first.charAt(0) === '<') return { status: 'missing' };
        return { status: 'error', message: 'Not a MIDI file' };
      }
      try { return { status: 'ok', parsed: M.parseMidi(buf) }; }
      catch (e) { return { status: 'error', message: e.message }; }
    })();
    return cache[url];
  }

  /* ---------------------------------------------------------------- player */
  var engine = new M.AudioEngine({
    soundfont: CFG.soundfont || 'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus',
    onModeChange: function (mode) {
      if (mode === 'synth' || mode === 'builtin') toast('The instrument soundfont could not load, so playback uses a simple built-in synthesizer.', 6000);
    }
  });
  var active = null, playToken = 0, raf = 0;

  function loop() {
    raf = 0;
    if (!active || (active.state !== 'playing' && active.state !== 'preparing')) return;
    if (active.state === 'playing') {
      var pos = Math.min(engine.position(), active.parsed.duration);
      active.pos = pos;
      active.draw(pos);
      active.setTime(pos);
    }
    raf = requestAnimationFrame(loop);
  }
  function startLoop() { if (!raf) raf = requestAnimationFrame(loop); }

  function pause(row) {
    if (!row) return;
    playToken++;
    if (row.state === 'playing') row.pos = Math.min(engine.position(), row.parsed.duration);
    engine.stop();
    row.setState(row.pos > 0.05 ? 'paused' : 'idle');
    row.draw(row.state === 'paused' ? row.pos : null);
    row.setTime(row.pos);
  }

  async function play(row, from) {
    engine.unlock();                           // must run inside the click handler
    if (!row.parsed) return;
    if (active && active !== row) { pause(active); var prev = active; prev.setState('idle'); prev.pos = 0; prev.draw(null); prev.setTime(0); }
    if (active === row && (row.state === 'playing' || row.state === 'preparing') && from == null) { pause(row); return; }
    active = row;
    var offset = from != null ? from : (row.pos >= row.parsed.duration - 0.05 ? 0 : row.pos);
    row.pos = offset;
    var token = ++playToken;
    row.setState('preparing');
    row.draw(offset > 0 ? offset : null);
    try { await engine.prepare(row.sequence(null)); }
    catch (e) { /* engine already fell back */ }
    if (token !== playToken) return;
    engine.start(row.sequence(engine.core), offset, function () {
      if (active !== row) return;
      row.pos = 0; row.setState('idle'); row.draw(null); row.setTime(0);
      active = null;
    });
    row.setState('playing');
    startLoop();
  }

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && active && active.state === 'playing') pause(active);
  });

  /* ---------------------------------------------------------------- rows */
  var allRows = [];
  var tpl = $('#row-tpl');

  function Row(card, spec) {
    this.card = card;
    this.spec = spec;
    this.url = spec.url;
    this.genre = spec.genre;
    this.state = 'loading';
    this.pos = 0;
    this.parsed = null;
    var node = tpl.content.firstElementChild.cloneNode(true);
    this.el = node;
    node.setAttribute('data-genre', spec.genre);
    if (spec.ours) node.classList.add('is-ours');
    if (spec.ablation) node.setAttribute('data-ablation', '');
    var name = $('.row__name', node);
    var dot = el('span', 'dot'); dot.setAttribute('aria-hidden', 'true');
    name.appendChild(dot);
    name.appendChild(document.createTextNode(spec.label));
    if (spec.ours) name.appendChild(el('span', 'tag', 'Ours'));
    $('.row__detail', node).textContent = spec.detail || '';
    this.btn = $('.play', node);
    this.btn.setAttribute('aria-label', 'Play ' + spec.label + ' (' + spec.ariaContext + ')');
    this.canvas = $('canvas', node);
    this.msg = $('.roll__msg', node);
    this.time = $('.row__time', node);
    this.dl = $('.row__dl', node);
    this.dl.href = encodeURI(spec.url);
    this.dl.setAttribute('download', spec.url.split('/').slice(-2).join('_'));
    var self = this;
    this.btn.addEventListener('click', function () { if (self.parsed) play(self); });
    this.canvas.addEventListener('click', function (e) {
      if (!self.parsed || !self.card.view) return;
      var r = self.canvas.getBoundingClientRect();
      var t = self.toTime((e.clientX - r.left) / r.width * self.card.view.nBars);
      t = Math.max(0, Math.min(self.parsed.duration - 0.05, t));
      play(self, t);
    });
    allRows.push(this);
  }
  /* Rows share a bar axis rather than a seconds axis, so files written at
     different tempi still line up bar for bar. */
  Row.prototype.barsOf = function () { return (this.parsed && this.parsed.bars) || this.card.view.bars; };
  Row.prototype.toBar = function (t) {
    var b = this.barsOf(), n = b.length;
    if (n < 2 || t <= b[0]) return 0;
    for (var i = 0; i < n - 1; i++) if (t < b[i + 1]) return i + (t - b[i]) / (b[i + 1] - b[i]);
    return (n - 1) + (t - b[n - 1]) / (b[n - 1] - b[n - 2]);
  };
  Row.prototype.toTime = function (bar) {
    var b = this.barsOf(), n = b.length;
    if (n < 2) return 0;
    var i = Math.max(0, Math.min(n - 2, Math.floor(bar)));
    return b[i] + (bar - i) * (b[i + 1] - b[i]);
  };
  Row.prototype.setState = function (s) { this.state = s; this.el.setAttribute('data-state', s); };
  Row.prototype.showsHead = function () { return this.state === 'playing' || this.state === 'paused' || (this.state === 'preparing' && this.pos > 0); };
  Row.prototype.setTime = function (pos) {
    if (!this.parsed) { this.time.textContent = '–:––'; return; }
    var d = this.parsed.duration;
    this.time.textContent = (this.state === 'playing' || this.state === 'paused' || this.state === 'preparing') ? fmt(pos) + ' / ' + fmt(d) : fmt(d);
  };
  Row.prototype.sequence = function (core) {
    if (!this._seq || (core && !this._seqCore)) { this._seq = M.toNoteSequence(this.parsed, core); this._seqCore = !!core; }
    return this._seq;
  };
  Row.prototype.load = async function () {
    var res = await fetchMidi(this.url);
    this.result = res;
    if (res.status === 'ok') {
      this.parsed = res.parsed;
      if (!this.parsed.notes.length) { this.parsed = null; this.setState('error'); this.msg.textContent = ''; this.msg.appendChild(el('span', null, 'This MIDI file has no notes')); return; }
      this.setState('idle');
      this.setTime(0);
      return;
    }
    this.setState(res.status === 'error' ? 'error' : 'missing');
    this.msg.textContent = '';
    var box = el('span');
    this.msg.appendChild(box);
    if (res.status === 'error') { box.textContent = 'Could not read this MIDI file'; return; }
    if (DEV) {
      box.appendChild(document.createTextNode(res.status === 'blocked' ? 'Needs a local server · ' : 'Add '));
      box.appendChild(el('code', null, this.url));
    } else {
      box.textContent = 'Sample coming soon';
    }
  };

  /* static layer: bar grid, notes, drum hits */
  Row.prototype.render = function () {
    var view = this.card.view, c = this.canvas;
    var w = Math.round(c.clientWidth), h = Math.round(c.clientHeight);
    if (!w || !h || !view) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = w; this.h = h; this.dpr = dpr;
    if (!this.layer) this.layer = document.createElement('canvas');
    this.layer.width = Math.round(w * dpr); this.layer.height = Math.round(h * dpr);
    var g = this.layer.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    var col = COL, gc = col.g[this.genre] || col.ink;
    var self = this, nBars = view.nBars;
    var drumH = view.drums ? Math.max(8, Math.round(h * 0.15)) : 0;
    var pitchH = h - drumH - (drumH ? 3 : 1);
    var span = view.maxP - view.minP + 1, rh = pitchH / span;
    function X(t) { return self.toBar(t) / nBars * w; }
    function Y(p) { return (view.maxP - p) * rh + 0.5; }
    var bars = this.barsOf();

    // alternate bar shading and bar lines
    g.fillStyle = col.ink;
    for (var i = 0; i < bars.length - 1; i++) {
      if (i % 2 === 1) { g.globalAlpha = 0.028; g.fillRect(X(bars[i]), 0, X(bars[i + 1]) - X(bars[i]), h); }
    }
    for (i = 0; i < bars.length; i++) {
      var x = Math.round(X(bars[i])) + 0.5;
      g.globalAlpha = i % 4 === 0 ? 0.22 : 0.09;
      g.fillRect(x - 0.5, 0, 1, h);
    }
    // octave guides
    g.globalAlpha = 0.06;
    for (var p = view.minP; p <= view.maxP; p++) if (p % 12 === 0) g.fillRect(0, Math.round(Y(p) + rh) , w, 1);
    if (drumH) { g.globalAlpha = 0.12; g.fillRect(0, h - drumH - 1.5, w, 1); }
    g.globalAlpha = 1;

    if (!this.parsed) { this.drawnLayer = true; return; }

    // notes: top voice strongest, lower voices lighter
    var nh = Math.max(1.5, rh - (rh > 3.5 ? 1 : 0.25));
    var rad = Math.min(1.5, nh / 2);
    g.fillStyle = gc;
    var notes = this.parsed.notes;
    for (i = 0; i < notes.length; i++) {
      var n = notes[i];
      if (n.isDrum) continue;
      var a = n.voiceRank === 0 ? 1 : n.voiceRank === 1 ? 0.62 : 0.42;
      g.globalAlpha = a * (0.55 + 0.45 * n.velocity / 127);
      var x0 = X(n.startTime), x1 = Math.max(x0 + 1.5, X(n.endTime) - 0.6);
      var y0 = Y(n.pitch);
      if (g.roundRect) { g.beginPath(); g.roundRect(x0, y0, x1 - x0, nh, rad); g.fill(); }
      else g.fillRect(x0, y0, x1 - x0, nh);
    }
    // drum hits along the bottom lane
    if (drumH) {
      g.fillStyle = col.ink;
      for (i = 0; i < notes.length; i++) {
        var d = notes[i];
        if (!d.isDrum) continue;
        var k = d.pitch === 35 || d.pitch === 36 ? 1 : (d.pitch === 38 || d.pitch === 40 || d.pitch === 37 || d.pitch === 39) ? 0.72 : 0.42;
        g.globalAlpha = 0.18 + 0.3 * d.velocity / 127;
        var hh = Math.max(2, (drumH - 2) * k);
        g.fillRect(Math.round(X(d.startTime)), h - hh, 1.5, hh);
      }
    }
    g.globalAlpha = 1;
  };

  /* visible frame: static layer + playhead */
  Row.prototype.draw = function (pos) {
    var c = this.canvas;
    if (!this.layer || !this.w) return;
    if (c.width !== this.layer.width || c.height !== this.layer.height) { c.width = this.layer.width; c.height = this.layer.height; }
    var ctx = c.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(this.layer, 0, 0);
    if (pos == null || !this.parsed) return;
    var dpr = this.dpr, w = this.w, h = this.h, view = this.card.view, self = this;
    function X(t) { return self.toBar(t) / view.nBars * w; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var gc = COL.g[this.genre] || COL.ink;
    var x = X(pos);
    ctx.fillStyle = gc;
    ctx.globalAlpha = 0.08;
    ctx.fillRect(0, 0, x, h);
    // light up sounding notes
    var drumH = view.drums ? Math.max(8, Math.round(h * 0.15)) : 0;
    var pitchH = h - drumH - (drumH ? 3 : 1);
    var span = view.maxP - view.minP + 1, rh = pitchH / span;
    var nh = Math.max(1.5, rh - (rh > 3.5 ? 1 : 0.25));
    ctx.globalAlpha = 1;
    ctx.strokeStyle = COL.ink;
    ctx.lineWidth = 1;
    var notes = this.parsed.notes;
    for (var i = 0; i < notes.length; i++) {
      var n = notes[i];
      if (n.startTime > pos) break;
      if (n.isDrum || n.endTime <= pos) continue;
      var x0 = X(n.startTime), x1 = Math.max(x0 + 1.5, X(n.endTime) - 0.6), y0 = (view.maxP - n.pitch) * rh + 0.5;
      ctx.fillStyle = gc;
      ctx.fillRect(x0, y0 - 0.5, x1 - x0, nh + 1);
      ctx.strokeRect(x0 + 0.5, y0, x1 - x0 - 1, nh);
    }
    // playhead
    ctx.fillStyle = COL.ink;
    ctx.fillRect(Math.round(x) - 1, 0, 2, h);
    ctx.beginPath();
    ctx.moveTo(Math.round(x) - 4, 0); ctx.lineTo(Math.round(x) + 4, 0); ctx.lineTo(Math.round(x), 5); ctx.closePath(); ctx.fill();
  };

  /* ---------------------------------------------------------------- cards */
  var cards = [];

  function Card(opts) {
    this.opts = opts;
    this.el = el('article', 'card');
    this.el.setAttribute('aria-label', opts.aria);
    this.rows = [];
    this.view = null;
    this.loaded = false;
    var head = el('div', 'card__head');
    var left = el('div');
    left.appendChild(el('div', 'card__id', opts.id));
    var title = el('h3', 'card__title');
    opts.title.forEach(function (part) {
      var s = el('span', part.genre ? 'gname' : 'arrow', part.text);
      if (part.genre) s.setAttribute('data-genre', part.genre);
      title.appendChild(s);
    });
    left.appendChild(title);
    head.appendChild(left);
    var doEl = el('span', 'card__do');
    doEl.innerHTML = opts.doHtml;
    head.appendChild(doEl);
    if (opts.note) head.appendChild(el('p', 'card__note', opts.note));
    this.el.appendChild(head);

    var ruler = el('div', 'ruler');
    ruler.setAttribute('aria-hidden', 'true');
    ruler.appendChild(el('span', 'ruler__unit', 'Bar'));
    this.scale = el('div', 'ruler__scale');
    ruler.appendChild(this.scale);
    this.el.appendChild(ruler);

    var self = this;
    opts.rows.forEach(function (spec) {
      var r = new Row(self, spec);
      self.rows.push(r);
      self.el.appendChild(r.el);
    });
    cards.push(this);
    this.layout();
  }

  Card.prototype.load = function () {
    if (this.loading) return this.loading;
    var self = this;
    this.loading = Promise.all(this.rows.map(function (r) { return r.load(); })).then(function () {
      self.loaded = true;
      self.layout();
      updateDevPanel();
    });
    return this.loading;
  };

  Card.prototype.layout = function () {
    var ok = this.rows.filter(function (r) { return r.parsed; });
    var minP = Infinity, maxP = -Infinity, drums = false;
    ok.forEach(function (r) {
      minP = Math.min(minP, r.parsed.minPitch); maxP = Math.max(maxP, r.parsed.maxPitch);
      drums = drums || r.parsed.hasDrums;
    });
    if (!ok.length) { minP = 48; maxP = 84; }
    minP -= 1; maxP += 1;
    if (maxP - minP < 30) { var mid = (maxP + minP) / 2; minP = Math.floor(mid - 15); maxP = Math.ceil(mid + 15); }
    var nBars = 0;
    ok.forEach(function (r) { nBars = Math.max(nBars, r.parsed.bars.length - 1); });
    if (!nBars) nBars = 16;
    var bars = [];
    for (var i = 0; i <= nBars; i++) bars.push(i);   // unit bars for rows without a file yet
    this.view = { minP: minP, maxP: maxP, nBars: nBars, drums: drums, bars: bars };
    this.drawRuler();
    this.rows.forEach(function (r) { r.render(); r.draw(r.showsHead() ? r.pos : null); });
  };

  Card.prototype.drawRuler = function () {
    var v = this.view, s = this.scale;
    s.textContent = '';
    for (var i = 0; i < v.nBars; i++) {
      var left = (i / v.nBars * 100) + '%';
      var tick = el('i', i % 4 === 0 ? 'major' : '');
      tick.style.left = left;
      s.appendChild(tick);
      if (i % 4 === 0) { var lab = el('span', null, String(i + 1)); lab.style.left = left; s.appendChild(lab); }
    }
  };

  function transferCard(ex, i) {
    var methods = CFG.methods || [];
    var id = 'Example ' + String(i + 1).padStart(2, '0');
    var card = new Card({
      id: id,
      aria: id + ': ' + gLabel(ex.source) + ' to ' + gLabel(ex.target),
      title: [{ text: gLabel(ex.source), genre: ex.source }, { text: '→' }, { text: gLabel(ex.target), genre: ex.target }],
      doHtml: 'x : ' + ex.source + ' &nbsp;·&nbsp; do(G := <b>' + ex.target + '</b>)',
      note: ex.note,
      rows: methods.map(function (m) {
        var isSrc = m.key === 'source';
        return {
          url: joinPath(ex.folder, m.file),
          label: m.label,
          detail: isSrc ? (m.detail || 'Original') + ' · ' + gLabel(ex.source) : m.detail,
          genre: isSrc ? ex.source : ex.target,
          ours: !!m.ours,
          ablation: !!m.ablation,
          ariaContext: id
        };
      })
    });
    card.el.setAttribute('data-source', ex.source);
    card.el.setAttribute('data-target', ex.target);
    return card;
  }

  /* ---------------------------------------------------------------- sections */
  function setupLegend() {
    var dl = $('#method-legend');
    (CFG.methods || []).forEach(function (m) {
      var d = el('div');
      if (m.ours) d.className = 'is-ours';
      var dt = el('dt');
      dt.appendChild(document.createTextNode(m.label));
      if (m.ours) dt.appendChild(el('span', 'tag', 'Ours'));
      if (m.ablation) dt.appendChild(el('span', 'tag tag--line', 'Ablation'));
      d.appendChild(dt);
      d.appendChild(el('dd', null, m.description || METHOD_NOTES[m.key] || m.detail || ''));
      dl.appendChild(d);
    });
  }

  var filter = 'all';
  function setupFilters(examples) {
    var box = $('#genre-filter');
    var present = GENRES.filter(function (g) { return examples.some(function (ex) { return ex.source === g.key || ex.target === g.key; }); });
    var opts = [{ key: 'all', label: 'All genres' }].concat(present);
    opts.forEach(function (g) {
      var b = el('button', 'chip');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(g.key === 'all'));
      if (g.key !== 'all') { b.setAttribute('data-genre', g.key); var dot = el('span', 'dot'); dot.setAttribute('aria-hidden', 'true'); b.appendChild(dot); }
      b.appendChild(document.createTextNode(g.label));
      b.addEventListener('click', function () {
        filter = g.key;
        box.querySelectorAll('.chip').forEach(function (c) { c.setAttribute('aria-pressed', String(c === b)); });
        applyFilter();
      });
      box.appendChild(b);
    });
    var tog = $('#ablation-toggle');
    tog.addEventListener('change', function () {
      $('#transfer-examples').classList.toggle('show-ablation', tog.checked);
      if (!tog.checked && active && active.spec.ablation) { pause(active); active = null; }
      cards.forEach(function (c) { if (c.view) c.rows.forEach(function (r) { if (r.spec.ablation) { r.render(); r.draw(null); } }); });
    });
    var hasAblation = (CFG.methods || []).some(function (m) { return m.ablation; });
    if (!hasAblation) tog.closest('.switch').hidden = true;
  }

  function applyFilter() {
    var shown = 0;
    document.querySelectorAll('#transfer-examples .card').forEach(function (c) {
      var ok = filter === 'all' || c.getAttribute('data-source') === filter || c.getAttribute('data-target') === filter;
      c.hidden = !ok;
      if (ok) shown++;
    });
    if (active && active.card.el.hidden) { pause(active); active = null; }
    var total = (CFG.transferExamples || []).length;
    $('#example-count').textContent = filter === 'all'
      ? total + (total === 1 ? ' example' : ' examples') + ' · all 16 bars long'
      : shown + ' of ' + total + ' examples involve ' + gLabel(filter);
    cards.forEach(function (c) { if (!c.el.hidden && c.view) c.layout(); });
  }

  function setupBib() {
    var pre = $('#bibtex');
    pre.textContent = CFG.bibtex || '';
    $('#bib-copy').addEventListener('click', function () {
      var text = pre.textContent;
      function selectIt() { var r = document.createRange(); r.selectNodeContents(pre); var s = getSelection(); s.removeAllRanges(); s.addRange(r); toast('Selected. Press Ctrl+C or ⌘C to copy.'); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(function () { toast('BibTeX copied'); }, selectIt);
      else selectIt();
    });
  }

  /* ---------------------------------------------------------------- local checklist */
  var devPanel = null;
  function updateDevPanel() {
    if (!DEV) return;
    if (!devPanel) {
      devPanel = el('aside', 'devpanel');
      devPanel.setAttribute('aria-label', 'Local preview checklist');
      devPanel.innerHTML = '<header><strong>Sample checklist</strong><span class="meter"><i></i></span><span class="n"></span><button type="button" aria-label="Collapse checklist">–</button></header><div class="body"></div>';
      document.body.appendChild(devPanel);
      $('button', devPanel).addEventListener('click', function () {
        devPanel.classList.toggle('is-collapsed');
        this.textContent = devPanel.classList.contains('is-collapsed') ? '+' : '–';
      });
    }
    var rows = allRows, done = rows.filter(function (r) { return r.result; });
    var ok = rows.filter(function (r) { return r.result && r.result.status === 'ok'; });
    var blocked = rows.some(function (r) { return r.result && r.result.status === 'blocked'; });
    var missing = done.filter(function (r) { return r.result.status !== 'ok'; });
    $('.n', devPanel).textContent = ok.length + ' / ' + rows.length;
    $('.meter i', devPanel).style.width = (rows.length ? ok.length / rows.length * 100 : 0) + '%';
    var body = $('.body', devPanel);
    var html = '';
    if (blocked) {
      html += '<p>Browsers block MIDI files on pages opened straight from disk. In this folder run <code>python3 -m http.server 8000</code> and open <code>http://localhost:8000</code>.</p>';
    } else {
      html += '<p>Only visible while previewing locally. Drop each file at the path shown and reload.</p>';
    }
    var code = ((CFG.links || {}).code || '').trim();
    html += '<p>' + (code ? '✓ GitHub link set in <code>config.js</code>' : '✗ GitHub link empty. Set <code>links.code</code> in <code>config.js</code>.') + '</p>';
    if (missing.length && !blocked) {
      html += '<ul>' + missing.map(function (r) {
        return '<li>' + (r.result.status === 'error' ? '⚠ unreadable · ' : '· ') + r.url.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</li>';
      }).join('') + '</ul>';
    } else if (!blocked && done.length === rows.length) {
      html += '<ul><li class="ok">✓ Every MIDI slot is filled.</li></ul>';
    }
    body.innerHTML = html;
  }

  /* ---------------------------------------------------------------- boot */
  function init() {
    COL = colors();
    setupLinks();
    setupTheme();
    setupLegend();
    setupBib();

    var examples = CFG.transferExamples || [];
    var tBox = $('#transfer-examples');
    examples.forEach(function (ex, i) { tBox.appendChild(transferCard(ex, i).el); });
    if (!examples.length) tBox.appendChild(el('p', 'empty', 'Examples will appear here soon.'));
    setupFilters(examples);
    applyFilter();

    // load cards as they approach the viewport (all at once while previewing locally)
    if (DEV || !('IntersectionObserver' in window)) {
      cards.forEach(function (c) { c.load(); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          io.unobserve(e.target);
          var c = cards.filter(function (x) { return x.el === e.target; })[0];
          if (c) c.load();
        });
      }, { rootMargin: '900px 0px' });
      cards.forEach(function (c) { io.observe(c.el); });
    }
    if (DEV) updateDevPanel();

    // redraw rolls when their width changes
    var pending = false;
    function relayout() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () {
        pending = false;
        cards.forEach(function (c) { if (!c.el.hidden && c.view) { c.rows.forEach(function (r) { r.render(); r.draw(r.showsHead() ? r.pos : null); }); } });
      });
    }
    if ('ResizeObserver' in window) {
      var ro = new ResizeObserver(relayout);
      cards.forEach(function (c) { ro.observe(c.el); });
    } else window.addEventListener('resize', relayout);

    // warm up the audio library once the page is idle, so the first play starts quickly
    var warm = function () { engine.loadCore().catch(function () {}); };
    if ('requestIdleCallback' in window) requestIdleCallback(warm, { timeout: 4000 }); else setTimeout(warm, 2500);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
