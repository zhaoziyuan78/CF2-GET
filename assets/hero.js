/* ==========================================================================
   CF²-GET demo · hero schematic
   Dots start as a piano roll (x), run backward along straight OT paths to a
   fixed noise cloud (û), then run forward under a new genre (x^cf).
   The noise positions never change between cycles, which is the point:
   only the genre condition is intervened on. Illustration only.
   ========================================================================== */
(function () {
  'use strict';

  var canvas = document.getElementById('flow-canvas');
  if (!canvas || !canvas.getContext) return;
  var fig = canvas.closest('.flowfig');
  var pill = document.getElementById('flow-state');
  var pillText = pill.querySelector('span');
  var tRead = document.getElementById('flow-t');
  var toggle = document.getElementById('flow-toggle');
  var stageEls = Array.prototype.slice.call(document.querySelectorAll('#flow-stages .stage'));
  var ctx = canvas.getContext('2d');

  var ORDER = ['classical', 'jazz', 'rock', 'latin', 'pop', 'country'];
  var LABEL = { classical: 'Classical', jazz: 'Jazz', rock: 'Rock', latin: 'Latin', pop: 'Pop', country: 'Country' };
  var STEPS = 64;                         // four bars of sixteenths
  var PH = { hold: 2.4, abd: 2.3, int: 1.4, pred: 2.3 };
  var CYCLE = PH.hold + PH.abd + PH.int + PH.pred;

  /* ---- seeded random so the figure looks the same on every visit ---- */
  var seed = 7;
  function rnd() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
  function gauss() { var u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /* ---- one melody, six arrangements ---- */
  var MELODY = [
    [0, 4, 67], [4, 2, 69], [6, 2, 67], [8, 4, 64], [12, 4, 60],
    [16, 4, 62], [20, 2, 64], [22, 2, 65], [24, 8, 64],
    [32, 4, 67], [36, 2, 69], [38, 2, 71], [40, 4, 72], [44, 4, 67],
    [48, 4, 65], [52, 4, 64], [56, 8, 62]
  ];
  var TRIADS = [[48, 52, 55], [47, 50, 55], [45, 48, 52], [47, 50, 53]];
  var ROOTS = [36, 43, 45, 43];
  var FIFTHS = [43, 38, 40, 38];

  function melody(shift) {
    var m = MELODY.map(function (n) { return n.slice(); });
    if (shift) shift(m);
    for (var i = 0; i < m.length - 1; i++) m[i][1] = Math.min(m[i][1], m[i + 1][0] - m[i][0]);
    return m.map(function (n) { return [n[0], n[1], n[2], 'mel']; });
  }
  function each(fn) { var out = []; for (var b = 0; b < 4; b++) fn(b, b * 16, out); return out; }

  var ARR = {
    classical: function () {
      var acc = each(function (b, o, out) {
        var c = TRIADS[b], pat = [c[0], c[2], c[1], c[2]];
        for (var k = 0; k < 8; k++) out.push([o + k * 2, 1.8, pat[k % 4], 'acc']);
        out.push([o, 15.6, ROOTS[b] - 12, 'bass']);
      });
      return { notes: melody().concat(acc), drums: [] };
    },
    jazz: function () {
      var mel = melody(function (m) {
        m.forEach(function (n) { if (n[0] % 4 === 2) n[0] += 0.67; });
        m.forEach(function (n) { if (n[0] === 16 || n[0] === 32) n[0] -= 1.33; });
        for (var i = 0; i < m.length - 1; i++) m[i][1] = m[i + 1][0] - m[i][0];
      });
      var shells = [[52, 55, 59], [50, 53, 59], [48, 52, 55], [50, 53, 59]];
      var walk = [[36, 40, 43, 45], [43, 41, 40, 38], [45, 43, 40, 41], [43, 38, 40, 42]];
      var acc = each(function (b, o, out) {
        shells[b].forEach(function (p) { out.push([o, 3, p, 'acc']); out.push([o + 6.67, 2, p, 'acc']); });
        for (var k = 0; k < 4; k++) out.push([o + k * 4, 3.5, walk[b][k], 'bass']);
      });
      var drums = each(function (b, o, out) {
        [0, 4, 6.67, 8, 12, 14.67].forEach(function (s) { out.push([o + s, 'ride']); });
        [4, 12].forEach(function (s) { out.push([o + s, 'hat']); });
      });
      return { notes: mel.concat(acc), drums: drums };
    },
    rock: function () {
      var roots = [48, 43, 45, 43];
      var acc = each(function (b, o, out) {
        var r = roots[b];
        for (var k = 0; k < 8; k++) { out.push([o + k * 2, 1.7, r, 'acc']); out.push([o + k * 2, 1.7, r + 7, 'acc']); }
        for (var j = 0; j < 8; j++) out.push([o + j * 2, 1.7, r - 12, 'bass']);
      });
      var drums = each(function (b, o, out) {
        [0, 6, 8, 10].forEach(function (s) { out.push([o + s, 'kick']); });
        [4, 12].forEach(function (s) { out.push([o + s, 'snare']); });
        for (var k = 0; k < 8; k++) out.push([o + k * 2, 'hat']);
      });
      return { notes: melody().concat(acc), drums: drums };
    },
    latin: function () {
      var mel = melody(function (m) { m.forEach(function (n) { if (n[0] % 16 === 8) n[0] -= 2; }); });
      var acc = each(function (b, o, out) {
        var c = TRIADS[b];
        [0, 3, 6, 8, 11, 14].forEach(function (s, k) { out.push([o + s, 1.5, c[k % 3] + (k % 2 ? 12 : 0), 'acc']); });
        out.push([o + 6, 5.6, FIFTHS[b], 'bass']);
        out.push([o + 12, 3.6, ROOTS[b], 'bass']);
      });
      var drums = each(function (b, o, out) {
        (b % 2 === 0 ? [0, 6, 12] : [4, 8]).forEach(function (s) { out.push([o + s, 'clave']); });
        [12, 14].forEach(function (s) { out.push([o + s, 'conga']); });
        for (var k = 0; k < 8; k++) out.push([o + k * 2 + 1, 'hat']);
      });
      return { notes: mel.concat(acc), drums: drums };
    },
    pop: function () {
      var acc = each(function (b, o, out) {
        TRIADS[b].forEach(function (p) { out.push([o, 15.5, p, 'acc']); });
        [[0, 5], [6, 2], [8, 5.5], [14, 1.8]].forEach(function (s) { out.push([o + s[0], s[1], ROOTS[b], 'bass']); });
      });
      var drums = each(function (b, o, out) {
        [0, 4, 8, 12].forEach(function (s) { out.push([o + s, 'kick']); });
        [4, 12].forEach(function (s) { out.push([o + s, 'snare']); });
        [2, 6, 10, 14].forEach(function (s) { out.push([o + s, 'hat']); });
      });
      return { notes: melody().concat(acc), drums: drums };
    },
    country: function () {
      var acc = each(function (b, o, out) {
        out.push([o, 3.6, ROOTS[b], 'bass']);
        out.push([o + 8, 3.6, FIFTHS[b], 'bass']);
        [4, 12].forEach(function (s) { TRIADS[b].forEach(function (p) { out.push([o + s, 1.8, p, 'acc']); }); });
      });
      var drums = each(function (b, o, out) {
        [0, 8].forEach(function (s) { out.push([o + s, 'kick']); });
        [4, 12].forEach(function (s) { out.push([o + s, 'snare']); });
        for (var k = 0; k < 8; k++) out.push([o + k * 2, 'hat']);
      });
      return { notes: melody().concat(acc), drums: drums };
    }
  };

  var DRUM_ROWS = { kick: 3, snare: 2, clave: 2, conga: 1, ride: 0, hat: 0 };
  var DRUM_FRAC = 0.13;

  var arr = {}, minP = 127, maxP = 0;
  ORDER.forEach(function (g) {
    arr[g] = ARR[g]();
    arr[g].notes.forEach(function (n) { minP = Math.min(minP, n[2]); maxP = Math.max(maxP, n[2]); });
  });
  minP -= 1; maxP += 1;
  var SPAN = maxP - minP + 1;

  function noteBox(n) {         // normalized rectangle of a note in plot space
    return {
      x: n[0] / STEPS, w: n[1] / STEPS,
      y: (maxP - n[2]) / SPAN * (1 - DRUM_FRAC), h: (1 / SPAN) * (1 - DRUM_FRAC),
      role: n[3]
    };
  }
  function drumBox(d) {
    var row = DRUM_ROWS[d[1]];
    return { x: d[0] / STEPS, w: 0.35 / STEPS, y: 1 - DRUM_FRAC + (row / 4) * DRUM_FRAC + 0.12 * DRUM_FRAC / 4, h: DRUM_FRAC / 4 * 0.76, role: 'drum' };
  }

  /* ---- particles ---- */
  var N = 0, noise = [], targets = {}, boxes = {}, jitter = [];
  function build() {
    seed = 7;
    N = canvas.clientWidth < 560 ? 520 : 900;
    noise = []; jitter = [];
    for (var i = 0; i < N; i++) {
      noise.push([Math.min(0.97, Math.max(0.03, 0.5 + gauss() * 0.17)), Math.min(0.97, Math.max(0.04, 0.47 + gauss() * 0.2)), rnd() * 6.283]);
      jitter.push(rnd());
    }
    noise.sort(function (a, b) { return a[0] - b[0]; });
    ORDER.forEach(function (g) {
      var bx = arr[g].notes.map(noteBox).concat(arr[g].drums.map(drumBox));
      boxes[g] = bx;
      var weights = bx.map(function (b) { return b.role === 'drum' ? 0.012 : b.w * (b.role === 'mel' ? 2.4 : b.role === 'bass' ? 1.1 : 0.9); });
      var total = weights.reduce(function (a, b) { return a + b; }, 0);
      var pts = [], alloc = weights.map(function (w) { return Math.max(1, Math.floor(w / total * N)); });
      var used = alloc.reduce(function (a, b) { return a + b; }, 0), k = 0;
      while (used < N) { alloc[k % alloc.length]++; used++; k += 7; }
      while (used > N) { var j = alloc.indexOf(Math.max.apply(null, alloc)); alloc[j]--; used--; }
      bx.forEach(function (b, i) {
        for (var c = 0; c < alloc[i]; c++) pts.push([b.x + rnd() * b.w, b.y + (0.2 + rnd() * 0.6) * b.h]);
      });
      pts.sort(function (a, b) { return a[0] - b[0]; });
      targets[g] = pts;
    });
  }

  /* ---- colours from the page tokens ---- */
  var col = {};
  function readColors() {
    var cs = getComputedStyle(document.documentElement);
    function v(n) { return cs.getPropertyValue(n).trim(); }
    col = { rule: v('--rule'), ruleStrong: v('--rule-strong'), noise: v('--noise'), ink: v('--ink'), muted: v('--muted'), g: {} };
    ORDER.forEach(function (g) { col.g[g] = v('--g-' + g); });
  }
  function hexRgb(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mix(a, b, t) {
    var A = hexRgb(a), B = hexRgb(b);
    return 'rgb(' + [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }).join(',') + ')';
  }

  /* ---- layout ---- */
  var W = 0, H = 0, dpr = 1, plot = { x: 0, y: 0, w: 0, h: 0 };
  function resize() {
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    var rebuild = (w < 560) !== (W < 560) || !N;
    W = w; H = h;
    var padX = w < 560 ? 14 : 24;
    plot = { x: padX, y: w < 560 ? 54 : 58, w: w - padX * 2, h: h - (w < 560 ? 54 : 58) - 16 };
    if (rebuild) build();
    draw();
  }

  /* ---- timeline ---- */
  var clock = 0, cycle = 0, running = false, userPaused = false, visible = true, last = 0, raf = 0;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function clamp01(t) { return t < 0 ? 0 : t > 1 ? 1 : t; }

  function phase() {
    var tau = clock % CYCLE, src = ORDER[cycle % ORDER.length], tgt = ORDER[(cycle + 1) % ORDER.length];
    if (tau < PH.hold) return { name: 'hold', s: tau / PH.hold, src: src, tgt: tgt };
    tau -= PH.hold;
    if (tau < PH.abd) return { name: 'abd', s: tau / PH.abd, src: src, tgt: tgt };
    tau -= PH.abd;
    if (tau < PH.int) return { name: 'int', s: tau / PH.int, src: src, tgt: tgt };
    tau -= PH.int;
    return { name: 'pred', s: tau / PH.pred, src: src, tgt: tgt };
  }

  var lastHud = '';
  function hud(ph) {
    var text, color, t, stage = 0;
    if (ph.name === 'hold') { text = (cycle === 0 ? 'x · ' : 'xᶜᶠ · ') + LABEL[ph.src]; color = col.g[ph.src]; t = 1; }
    else if (ph.name === 'abd') { text = 'abduction · ' + LABEL[ph.src] + ' → û'; color = col.g[ph.src]; t = 1 - ease(ph.s); stage = 1; }
    else if (ph.name === 'int') { text = 'û held fixed · do(G := ' + LABEL[ph.tgt] + ')'; color = col.g[ph.tgt]; t = 0; stage = 2; }
    else { text = 'prediction · û → ' + LABEL[ph.tgt]; color = col.g[ph.tgt]; t = ease(ph.s); stage = 3; }
    var key = text + stage;
    if (key !== lastHud) {
      pillText.textContent = text;
      fig.style.setProperty('--hud-color', color);
      stageEls.forEach(function (el, i) { el.classList.toggle('is-active', i + 1 === stage); if (i + 1 !== stage) el.querySelector('.stage__bar i').style.width = '0'; });
      lastHud = key;
    }
    if (stage) stageEls[stage - 1].querySelector('.stage__bar i').style.width = (ph.s * 100).toFixed(1) + '%';
    tRead.textContent = 't = ' + t.toFixed(2);
  }

  /* ---- drawing ---- */
  function px(nx) { return plot.x + nx * plot.w; }
  function py(ny) { return plot.y + ny * plot.h; }

  function grid() {
    ctx.lineWidth = 1;
    ctx.strokeStyle = col.rule;
    ctx.beginPath();
    for (var b = 0; b <= 4; b++) { var x = Math.round(px(b / 4)) + 0.5; ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); }
    for (var p = minP; p <= maxP; p++) {
      if (p % 12 !== 0) continue;
      var y = Math.round(py((maxP - p + 1) / SPAN * (1 - DRUM_FRAC))) + 0.5;
      ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    }
    var yl = Math.round(py(1 - DRUM_FRAC)) + 0.5;
    ctx.moveTo(plot.x, yl); ctx.lineTo(plot.x + plot.w, yl);
    ctx.stroke();
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    for (var s = 1; s < 16; s++) { if (s % 4 === 0) continue; var xb = Math.round(px(s / 16)) + 0.5; ctx.moveTo(xb, plot.y + plot.h - 4); ctx.lineTo(xb, plot.y + plot.h); }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = col.muted;
    ctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textBaseline = 'bottom';
    for (var bb = 0; bb < 4; bb++) ctx.fillText(String(bb + 1), px(bb / 4) + 4, plot.y - 3);
  }

  function notes(g, alpha) {
    if (alpha <= 0.01) return;
    var c = col.g[g];
    boxes[g].forEach(function (b) {
      var a = b.role === 'mel' ? 0.95 : b.role === 'drum' ? 0.5 : 0.5;
      ctx.globalAlpha = a * alpha;
      ctx.fillStyle = b.role === 'drum' ? col.muted : c;
      var x = px(b.x), y = py(b.y), w = Math.max(2, b.w * plot.w - 1.5), h = Math.max(2, b.h * plot.h - 1);
      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(2, h / 2)); ctx.fill(); }
      else ctx.fillRect(x, y, w, h);
    });
    ctx.globalAlpha = 1;
  }

  function particles(from, to, s, color, size, streak, shimmer, alpha) {
    var i, n = noise.length;
    var J = 0.3;
    if (alpha == null) alpha = 1;
    if (alpha <= 0.01) return;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9 * alpha;
    var prevs = streak ? [] : null;
    for (i = 0; i < n; i++) {
      var si = ease(clamp01(s * (1 + J) - jitter[i] * J));
      var a = from[i], b = to[i];
      var x = a[0] + (b[0] - a[0]) * si, y = a[1] + (b[1] - a[1]) * si;
      if (shimmer) { x += Math.cos(clock * 2.1 + noise[i][2]) * 0.0022; y += Math.sin(clock * 2.6 + noise[i][2]) * 0.004; }
      var X = px(x), Y = py(y);
      ctx.fillRect(X - size / 2, Y - size / 2, size, size);
      if (prevs) {
        var sp = ease(clamp01((s - 0.035) * (1 + J) - jitter[i] * J));
        prevs.push(X, Y, px(a[0] + (b[0] - a[0]) * sp), py(a[1] + (b[1] - a[1]) * sp));
      }
    }
    if (prevs) {
      ctx.strokeStyle = color; ctx.globalAlpha = 0.35 * alpha; ctx.lineWidth = 1;
      ctx.beginPath();
      for (i = 0; i < prevs.length; i += 4) { ctx.moveTo(prevs[i], prevs[i + 1]); ctx.lineTo(prevs[i + 2], prevs[i + 3]); }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function draw() {
    if (!W) return;
    var ph = phase();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    grid();
    var src = ph.src, tgt = ph.tgt, T = targets;
    if (ph.name === 'hold') {
      notes(src, 1);
    } else if (ph.name === 'abd') {
      notes(src, 1 - clamp01(ph.s * 5));
      particles(T[src], noise, ph.s, mix(col.g[src], col.noise, ease(ph.s)), 2, ph.s > 0.02 && ph.s < 0.98, false, clamp01(ph.s * 8));
    } else if (ph.name === 'int') {
      particles(noise, noise, 1, mix(col.noise, col.g[tgt], ease(ph.s) * 0.55), 2, false, true, 1);
    } else {
      var fade = clamp01((ph.s - 0.84) / 0.16);
      notes(tgt, fade);
      particles(noise, T[tgt], ph.s, mix(col.noise, col.g[tgt], 0.55 + 0.45 * ease(ph.s)), 2, ph.s > 0.02 && ph.s < 0.98, false, 1 - fade);
    }
    hud(ph);
  }

  function frame(now) {
    raf = 0;
    if (!running) return;
    var dt = Math.max(0, Math.min(0.05, (now - last) / 1000)); last = now;
    clock += dt;
    cycle = Math.floor(clock / CYCLE);
    draw();
    raf = requestAnimationFrame(frame);
  }
  function play() { if (running) return; running = true; last = performance.now(); raf = requestAnimationFrame(frame); }
  function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
  function sync() { if (!userPaused && visible && !document.hidden) play(); else stop(); }

  toggle.addEventListener('click', function () {
    userPaused = !userPaused;
    toggle.setAttribute('aria-pressed', String(userPaused));
    toggle.setAttribute('aria-label', userPaused ? 'Play animation' : 'Pause animation');
    sync();
  });
  document.addEventListener('visibilitychange', sync);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; sync(); }, { threshold: 0.05 }).observe(canvas);
  }
  if ('ResizeObserver' in window) new ResizeObserver(function () { resize(); }).observe(canvas);
  else window.addEventListener('resize', resize);
  window.addEventListener('cf2get:theme', function () { readColors(); lastHud = ''; draw(); });

  readColors();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { draw(); });
  if (reduce) { userPaused = true; toggle.setAttribute('aria-pressed', 'true'); toggle.setAttribute('aria-label', 'Play animation'); }
  resize();
  sync();
})();
