/* ==========================================================================
   CF²-GET demo · MIDI parsing and playback
   - parseMidi(): a small Standard MIDI File reader (format 0/1, PPQ timing)
   - AudioEngine: plays note sequences with the SGM+ General MIDI soundfont
     through Magenta.js, falling back to Magenta's synth and finally to a
     built-in Web Audio synth if anything along the way cannot load.
   ========================================================================== */
(function (global) {
  'use strict';

  /* ------------------------------------------------------------------------
     Standard MIDI File parser
     ------------------------------------------------------------------------ */
  function parseMidi(buffer) {
    var bytes = new Uint8Array(buffer);
    var dv = new DataView(buffer);
    function ascii(at, n) { var s = ''; for (var i = 0; i < n; i++) s += String.fromCharCode(bytes[at + i]); return s; }
    if (bytes.length < 14 || ascii(0, 4) !== 'MThd') throw new Error('This file is not a MIDI file.');

    var headerLen = dv.getUint32(4);
    var nTracks = dv.getUint16(10);
    var division = dv.getUint16(12);
    if (division & 0x8000) throw new Error('SMPTE-timed MIDI files are not supported.');
    var ppq = division || 480;

    var p = 8 + headerLen;
    var tempos = [], timeSigs = [], programs = [], raw = [];
    var lastTick = 0, trackIndex = 0;

    function vlq() {
      var v = 0, b, guard = 0;
      do { b = bytes[p++]; v = (v * 128) + (b & 0x7f); guard++; } while ((b & 0x80) && p < bytes.length && guard < 5);
      return v;
    }

    while (p + 8 <= bytes.length && trackIndex < nTracks) {
      var id = ascii(p, 4);
      var len = dv.getUint32(p + 4);
      p += 8;
      var end = Math.min(p + len, bytes.length);
      if (id !== 'MTrk') { p = end; continue; }
      var t = trackIndex++;
      var tick = 0, status = 0;
      var open = {};

      while (p < end) {
        tick += vlq();
        var b = bytes[p];
        if (b === 0xff) {                       // meta event
          var type = bytes[p + 1]; p += 2;
          var l = vlq(), at = p; p += l;
          if (type === 0x51 && l >= 3) tempos.push({ tick: tick, uspq: (bytes[at] << 16) | (bytes[at + 1] << 8) | bytes[at + 2] });
          else if (type === 0x58 && l >= 2) timeSigs.push({ tick: tick, num: bytes[at] || 4, den: Math.pow(2, bytes[at + 1]) || 4 });
          else if (type === 0x2f) break;
          continue;
        }
        if (b === 0xf0 || b === 0xf7) { p++; var sl = vlq(); p += sl; continue; }   // sysex
        if (b & 0x80) { status = b; p++; }
        else if (!status) { p++; continue; }                                        // stray data byte
        var hi = status & 0xf0, ch = status & 0x0f;
        if (hi === 0xc0 || hi === 0xd0) {
          var d1 = bytes[p++];
          if (hi === 0xc0) programs.push({ tick: tick, ch: ch, program: d1 });
          continue;
        }
        if (hi < 0x80 || hi > 0xe0) { p++; continue; }
        var a = bytes[p++], v = bytes[p++];
        var key = ch * 128 + a;
        if (hi === 0x90 && v > 0) {
          (open[key] = open[key] || []).push({ tick: tick, vel: v });
        } else if (hi === 0x80 || hi === 0x90) {
          var q = open[key];
          if (q && q.length) { var on = q.shift(); raw.push({ track: t, ch: ch, pitch: a, vel: on.vel, on: on.tick, off: tick }); }
        }
      }
      // close notes still sounding at the end of the track
      for (var k in open) {
        open[k].forEach(function (on) {
          raw.push({ track: t, ch: Math.floor(k / 128), pitch: k % 128, vel: on.vel, on: on.tick, off: Math.max(tick, on.tick + ppq / 4) });
        });
      }
      lastTick = Math.max(lastTick, tick);
      p = end;
    }

    /* tempo map → seconds */
    tempos.sort(function (x, y) { return x.tick - y.tick; });
    var tmap = [{ tick: 0, sec: 0, spt: 0.5 / ppq }];
    tempos.forEach(function (tp) {
      var last = tmap[tmap.length - 1];
      var spt = tp.uspq / 1e6 / ppq;
      if (tp.tick === last.tick) { last.spt = spt; return; }
      tmap.push({ tick: tp.tick, sec: last.sec + (tp.tick - last.tick) * last.spt, spt: spt });
    });
    function toSec(tk) {
      var i = tmap.length - 1;
      while (i > 0 && tmap[i].tick > tk) i--;
      return tmap[i].sec + (tk - tmap[i].tick) * tmap[i].spt;
    }

    /* programs per channel */
    var progs = [];
    for (var c = 0; c < 16; c++) progs.push([]);
    programs.sort(function (x, y) { return x.tick - y.tick; }).forEach(function (pc) { progs[pc.ch].push(pc); });
    function programAt(chn, tk) {
      var list = progs[chn], val = 0;
      for (var i = 0; i < list.length && list[i].tick <= tk; i++) val = list[i].program;
      return val;
    }

    /* bar grid from time signatures */
    timeSigs.sort(function (x, y) { return x.tick - y.tick; });
    if (!timeSigs.length || timeSigs[0].tick > 0) timeSigs.unshift({ tick: 0, num: 4, den: 4 });
    var noteEnd = 0;
    raw.forEach(function (n) { if (n.off > noteEnd) noteEnd = n.off; });

    var barTicks = [], tsi = 0, tk = 0;
    var firstBar = ppq * 4 * timeSigs[0].num / timeSigs[0].den;
    var endTick = Math.max(noteEnd, Math.min(lastTick, noteEnd + firstBar));
    var tol = ppq / 8;
    while (barTicks.length < 1024) {
      while (tsi + 1 < timeSigs.length && timeSigs[tsi + 1].tick <= tk) tsi++;
      barTicks.push(tk);
      if (tk >= endTick - tol && barTicks.length > 1) break;
      var ts = timeSigs[tsi];
      var next = tk + ppq * 4 * ts.num / ts.den;
      if (tsi + 1 < timeSigs.length && timeSigs[tsi + 1].tick < next && timeSigs[tsi + 1].tick > tk) next = timeSigs[tsi + 1].tick;
      tk = next;
    }

    /* voices: one per (track, channel) that carries pitched notes */
    var voiceIds = {}, voices = [];
    var notes = raw.map(function (n) {
      var off = n.off > n.on ? n.off : n.on + Math.max(1, ppq / 16);
      var isDrum = n.ch === 9;
      var vkey = n.track + ':' + n.ch;
      if (!isDrum && !(vkey in voiceIds)) { voiceIds[vkey] = voices.length; voices.push({ sum: 0, count: 0 }); }
      var voice = isDrum ? -1 : voiceIds[vkey];
      if (!isDrum) { voices[voice].sum += n.pitch; voices[voice].count++; }
      return {
        pitch: n.pitch,
        velocity: Math.max(1, Math.min(127, n.vel)),
        startTime: toSec(n.on),
        endTime: toSec(off),
        program: programAt(n.ch, n.on),
        isDrum: isDrum,
        instrument: isDrum ? 9 : (voice % 16 === 9 ? 15 : voice % 16),
        voice: voice
      };
    }).sort(function (x, y) { return x.startTime - y.startTime || x.pitch - y.pitch; });

    /* rank voices by mean pitch, highest first, so the top line (usually the melody) draws strongest */
    var order = voices.map(function (vv, i) { return { i: i, mean: vv.count ? vv.sum / vv.count : 0 }; })
      .sort(function (x, y) { return y.mean - x.mean; });
    var rank = {};
    order.forEach(function (o, r) { rank[o.i] = r; });
    notes.forEach(function (n) { n.voiceRank = n.voice < 0 ? -1 : rank[n.voice]; });

    var minP = 127, maxP = 0, hasDrums = false;
    notes.forEach(function (n) {
      if (n.isDrum) { hasDrums = true; return; }
      if (n.pitch < minP) minP = n.pitch;
      if (n.pitch > maxP) maxP = n.pitch;
    });

    var bars = barTicks.map(toSec);
    var duration = bars[bars.length - 1];
    notes.forEach(function (n) { if (n.endTime > duration) duration = n.endTime; });

    return {
      notes: notes,
      duration: duration,
      bars: bars,
      qpm: 60e6 / (tempos.length ? tempos[0].uspq : 500000),
      minPitch: minP <= maxP ? minP : 60,
      maxPitch: minP <= maxP ? maxP : 72,
      hasDrums: hasDrums,
      voiceCount: voices.length
    };
  }

  function toNoteSequence(parsed, core) {
    var obj = {
      ticksPerQuarter: 220,
      totalTime: parsed.duration,
      tempos: [{ time: 0, qpm: parsed.qpm }],
      notes: parsed.notes.map(function (n) {
        return { pitch: n.pitch, velocity: n.velocity, startTime: n.startTime, endTime: n.endTime, program: n.program, isDrum: n.isDrum, instrument: n.instrument };
      })
    };
    try { if (core && core.NoteSequence && core.NoteSequence.create) return core.NoteSequence.create(obj); } catch (e) { /* plain object works too */ }
    return obj;
  }

  /* ------------------------------------------------------------------------
     Built-in Web Audio synth (last-resort fallback)
     ------------------------------------------------------------------------ */
  function family(program) {
    var f = Math.floor(program / 8);
    switch (f) {
      case 0: case 1: return { wave: 'triangle', attack: 0.004, decay: 0.45, sustain: 0.18, release: 0.18, bright: 6, gain: 1.0 };   // piano, chromatic perc.
      case 2: return { wave: 'square', attack: 0.01, decay: 0.2, sustain: 0.7, release: 0.08, bright: 3, gain: 0.45 };              // organ
      case 3: return { wave: 'sawtooth', attack: 0.004, decay: 0.3, sustain: 0.25, release: 0.12, bright: 4, gain: 0.55 };          // guitar
      case 4: return { wave: 'triangle', attack: 0.006, decay: 0.25, sustain: 0.6, release: 0.08, bright: 3, gain: 1.25 };          // bass
      case 5: case 6: return { wave: 'sawtooth', attack: 0.07, decay: 0.3, sustain: 0.75, release: 0.25, bright: 3.5, gain: 0.45 }; // strings, ensemble
      case 7: return { wave: 'sawtooth', attack: 0.03, decay: 0.2, sustain: 0.7, release: 0.12, bright: 4, gain: 0.45 };            // brass
      case 8: case 9: return { wave: 'square', attack: 0.03, decay: 0.2, sustain: 0.7, release: 0.12, bright: 3, gain: 0.4 };       // reed, pipe
      case 10: return { wave: 'square', attack: 0.005, decay: 0.2, sustain: 0.6, release: 0.1, bright: 5, gain: 0.4 };              // synth lead
      default: return { wave: 'triangle', attack: 0.02, decay: 0.3, sustain: 0.5, release: 0.2, bright: 4, gain: 0.8 };
    }
  }

  function BuiltinSynth() { this.ctx = null; this.timer = null; this.voices = []; this.base = 0; }
  BuiltinSynth.prototype.context = function () {
    if (!this.ctx) {
      var AC = global.AudioContext || global.webkitAudioContext;
      if (!AC) return null;
      this.ctx = new AC();
      var comp = this.ctx.createDynamicsCompressor();
      comp.threshold.value = -14; comp.ratio.value = 4;
      this.master = this.ctx.createGain(); this.master.gain.value = 0.7;
      this.master.connect(comp); comp.connect(this.ctx.destination);
      var len = this.ctx.sampleRate * 1.0, buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate), d = buf.getChannelData(0);
      for (var i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      this.noise = buf;
    }
    return this.ctx;
  };
  BuiltinSynth.prototype.unlock = function () { var c = this.context(); if (c && c.state === 'suspended') c.resume(); };
  BuiltinSynth.prototype.track = function (src, gain, stopAt) {
    var self = this, v = { src: src, gain: gain };
    this.voices.push(v);
    src.onended = function () { var i = self.voices.indexOf(v); if (i >= 0) self.voices.splice(i, 1); try { gain.disconnect(); } catch (e) {} };
    src.start(Math.max(0, stopAt[0])); src.stop(stopAt[1]);
  };
  BuiltinSynth.prototype.tone = function (n, t, tEnd) {
    var ctx = this.ctx, fam = family(n.program), vel = n.velocity / 127;
    var f = 440 * Math.pow(2, (n.pitch - 69) / 12);
    var dur = Math.max(0.04, tEnd - t);
    var osc = ctx.createOscillator(); osc.type = fam.wave; osc.frequency.value = f;
    var flt = ctx.createBiquadFilter(); flt.type = 'lowpass'; flt.frequency.value = Math.min(14000, f * fam.bright + 300); flt.Q.value = 0.4;
    var g = ctx.createGain(), peak = 0.16 * fam.gain * (0.35 + 0.65 * vel);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(peak, t + fam.attack);
    g.gain.setTargetAtTime(peak * fam.sustain, t + fam.attack, fam.decay);
    g.gain.setTargetAtTime(0.0001, t + dur, fam.release / 3);
    osc.connect(flt); flt.connect(g); g.connect(this.master);
    this.track(osc, g, [t, t + dur + fam.release * 2]);
  };
  BuiltinSynth.prototype.drum = function (n, t) {
    var ctx = this.ctx, vel = n.velocity / 127, p = n.pitch, g = ctx.createGain(), src;
    if (p === 35 || p === 36) {                       // kick
      src = ctx.createOscillator(); src.frequency.setValueAtTime(140, t); src.frequency.exponentialRampToValueAtTime(42, t + 0.12);
      g.gain.setValueAtTime(0.9 * vel, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.28);
      src.connect(g); g.connect(this.master); this.track(src, g, [t, t + 0.3]); return;
    }
    if ([41, 43, 45, 47, 48, 50].indexOf(p) >= 0) {   // toms
      src = ctx.createOscillator(); var f0 = 90 + (p - 41) * 14;
      src.frequency.setValueAtTime(f0 * 1.6, t); src.frequency.exponentialRampToValueAtTime(f0, t + 0.15);
      g.gain.setValueAtTime(0.55 * vel, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
      src.connect(g); g.connect(this.master); this.track(src, g, [t, t + 0.32]); return;
    }
    src = ctx.createBufferSource(); src.buffer = this.noise;
    var flt = ctx.createBiquadFilter(), decay = 0.08, level = 0.3;
    if (p === 38 || p === 40 || p === 37 || p === 39) { flt.type = 'bandpass'; flt.frequency.value = 1800; flt.Q.value = 0.7; decay = 0.16; level = 0.55; }
    else if (p === 42 || p === 44) { flt.type = 'highpass'; flt.frequency.value = 7500; decay = 0.045; level = 0.22; }
    else if (p === 46) { flt.type = 'highpass'; flt.frequency.value = 7000; decay = 0.26; level = 0.2; }
    else if ([49, 52, 55, 57].indexOf(p) >= 0) { flt.type = 'highpass'; flt.frequency.value = 5000; decay = 0.8; level = 0.18; }
    else if ([51, 53, 59].indexOf(p) >= 0) { flt.type = 'bandpass'; flt.frequency.value = 6500; flt.Q.value = 1.5; decay = 0.35; level = 0.16; }
    else { flt.type = 'bandpass'; flt.frequency.value = 2500 + (p % 12) * 250; flt.Q.value = 2; decay = 0.09; level = 0.25; }
    g.gain.setValueAtTime(level * vel, t); g.gain.exponentialRampToValueAtTime(0.001, t + decay);
    src.connect(flt); flt.connect(g); g.connect(this.master);
    this.track(src, g, [t, t + decay + 0.02]);
  };
  BuiltinSynth.prototype.start = function (seq, offset, onEnd) {
    this.stop();
    var ctx = this.context(), self = this;
    if (ctx.state === 'suspended') ctx.resume();
    var notes = seq.notes.filter(function (n) { return n.endTime > offset + 0.01; });
    var i = 0;
    this.base = ctx.currentTime + 0.05 - offset;
    this.total = seq.totalTime;
    function pump() {
      var horizon = ctx.currentTime + 0.3;
      while (i < notes.length && self.base + notes[i].startTime < horizon) {
        var n = notes[i++], t = Math.max(self.base + n.startTime, ctx.currentTime + 0.005);
        if (n.isDrum) { if (n.startTime >= offset - 0.005) self.drum(n, t); }
        else self.tone(n, t, self.base + n.endTime);
      }
      if (ctx.currentTime >= self.base + self.total) { self.stop(); if (onEnd) onEnd(); }
    }
    pump();
    this.timer = setInterval(pump, 40);
  };
  BuiltinSynth.prototype.stop = function () {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (!this.ctx) return;
    var now = this.ctx.currentTime;
    this.voices.slice().forEach(function (v) {
      try { v.gain.gain.cancelScheduledValues(now); v.gain.gain.setTargetAtTime(0.0001, now, 0.015); v.src.stop(now + 0.08); } catch (e) {}
    });
  };
  BuiltinSynth.prototype.position = function () { return this.ctx ? this.ctx.currentTime - this.base : 0; };

  /* ------------------------------------------------------------------------
     Audio engine: Magenta soundfont → Magenta synth → built-in synth
     ------------------------------------------------------------------------ */
  var SOURCES = [
    ['https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js'],
    ['https://unpkg.com/tone@14.7.58/build/Tone.js', 'https://unpkg.com/@magenta/music@1.23.1/es6/core.js'],
    // some builds of the core bundle expect TensorFlow.js as a global
    ['https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js', 'https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/core.js']
  ];

  function loadScript(src, ms) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      var timer = setTimeout(function () { reject(new Error('Timed out loading ' + src)); }, ms);
      s.src = src; s.async = false;
      s.onload = function () { clearTimeout(timer); resolve(); };
      s.onerror = function () { clearTimeout(timer); reject(new Error('Could not load ' + src)); };
      document.head.appendChild(s);
    });
  }
  function withTimeout(promise, ms, label) {
    return new Promise(function (resolve, reject) {
      var timer = setTimeout(function () { reject(new Error(label + ' timed out')); }, ms);
      promise.then(function (v) { clearTimeout(timer); resolve(v); }, function (e) { clearTimeout(timer); reject(e); });
    });
  }

  function AudioEngine(options) {
    this.soundfont = options.soundfont;
    this.mode = null;            // 'soundfont' | 'synth' | 'builtin'
    this.core = null;
    this.corePromise = null;
    this.builtin = new BuiltinSynth();
    this.session = 0;
    this.playerSession = -1;
    this.playing = false;
    this.t0 = 0;
    this.onModeChange = options.onModeChange || function () {};
  }

  AudioEngine.prototype.loadCore = function () {
    var self = this;
    if (this.corePromise) return this.corePromise;
    this.corePromise = (async function () {
      if (global.core && global.core.SoundFontPlayer) return global.core;
      for (var i = 0; i < SOURCES.length; i++) {
        try {
          for (var j = 0; j < SOURCES[i].length; j++) await loadScript(SOURCES[i][j], 20000);
          if (global.core && global.core.SoundFontPlayer) return global.core;
        } catch (e) { /* try the next mirror */ }
      }
      throw new Error('Magenta.js is unavailable');
    })().then(function (core) { self.core = core; return core; });
    return this.corePromise;
  };

  /* Call synchronously inside the click handler so browsers allow audio. */
  AudioEngine.prototype.unlock = function () {
    try { if (global.Tone && global.Tone.context && global.Tone.context.state !== 'running') global.Tone.context.resume(); } catch (e) {}
    try { if (this.core && this.core.Player && this.core.Player.tone) this.core.Player.tone.context.resume(); } catch (e) {}
    if (this.mode === 'builtin' || !this.core) this.builtin.unlock();
  };

  AudioEngine.prototype.callbacks = function () {
    var self = this;
    return {
      run: function (note) {
        if (self.playerSession !== self.session) { try { self.activePlayer().stop(); } catch (e) {} return; }
        // keep the on-screen playhead locked to what is actually sounding
        var est = (performance.now() - self.t0) / 1000;
        if (note && typeof note.startTime === 'number' && Math.abs(est - note.startTime) > 0.12) self.t0 = performance.now() - note.startTime * 1000;
      },
      stop: function () {}
    };
  };

  AudioEngine.prototype.activePlayer = function () { return this.mode === 'soundfont' ? this.sfPlayer : this.synthPlayer; };

  AudioEngine.prototype.setMode = function (mode) { if (this.mode !== mode) { this.mode = mode; this.onModeChange(mode); } };

  /* Load whatever the sequence needs. Resolves once playback can start. */
  AudioEngine.prototype.prepare = async function (seq) {
    if (this.mode === null) {
      try {
        var core = await this.loadCore();
        this.sfPlayer = new core.SoundFontPlayer(this.soundfont, undefined, undefined, undefined, this.callbacks());
        this.setMode('soundfont');
      } catch (e) {
        this.setMode('builtin');
      }
    }
    if (this.mode === 'soundfont') {
      try {
        await withTimeout(this.sfPlayer.loadSamples(seq), 30000, 'Soundfont');
      } catch (e) {
        try { this.synthPlayer = new this.core.Player(false, this.callbacks()); this.setMode('synth'); }
        catch (e2) { this.setMode('builtin'); }
      }
    }
    try { if (this.core && this.core.Player && this.core.Player.tone) await this.core.Player.tone.context.resume(); } catch (e) {}
  };

  AudioEngine.prototype.lookAhead = function () {
    try { var la = this.core.Player.tone.context.lookAhead; return typeof la === 'number' ? la : 0.1; } catch (e) { return 0.1; }
  };

  /* Start playback at `offset` seconds; `onEnd` fires only on natural end. */
  AudioEngine.prototype.start = function (seq, offset, onEnd) {
    var self = this;
    this.stop();
    var my = ++this.session;
    this.playing = true;
    if (this.mode === 'builtin') {
      this.builtin.start(seq, offset, function () { if (my === self.session) { self.playing = false; onEnd && onEnd(); } });
      return;
    }
    var player = this.activePlayer();
    this.playerSession = my;
    this.t0 = performance.now() - offset * 1000 + this.lookAhead() * 1000;
    var finished = false;
    function finish() {
      if (finished || my !== self.session) return;
      finished = true;
      self.stop();
      if (onEnd) onEnd();
    }
    var run;
    try { run = player.start(seq, undefined, offset); }
    catch (e) {
      // the player refused (e.g. still busy); fall back for this and later plays
      this.setMode('builtin');
      this.builtin.start(seq, offset, function () { if (my === self.session) { self.playing = false; onEnd && onEnd(); } });
      return;
    }
    // Magenta schedules its own stop at the full length even when starting mid-way,
    // so end the play ourselves once the remaining time has elapsed.
    clearTimeout(this.endTimer);
    this.endTimer = setTimeout(finish, (seq.totalTime - offset + this.lookAhead()) * 1000 + 120);
    Promise.resolve(run).then(finish, function () {
      try { player.stop(); } catch (e) {}
      finish();
    });
  };

  AudioEngine.prototype.stop = function () {
    this.session++;
    clearTimeout(this.endTimer);
    this.playing = false;
    [this.sfPlayer, this.synthPlayer].forEach(function (pl) {
      try { if (pl && pl.isPlaying()) pl.stop(); } catch (e) {}
    });
    this.builtin.stop();
  };

  AudioEngine.prototype.position = function () {
    if (!this.playing) return 0;
    if (this.mode === 'builtin') return this.builtin.position();
    return Math.max(0, (performance.now() - this.t0) / 1000);
  };

  global.CF2MIDI = { parseMidi: parseMidi, toNoteSequence: toNoteSequence, AudioEngine: AudioEngine };
})(window);
