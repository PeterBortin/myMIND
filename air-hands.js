// Air: руки, жести й голос для канваса мислення.
// Портовано з myDASH (MediaPipe tasks-vision): щіпка з гістерезисом, фільтр
// фантомних рук, hold-жести, Z-глибина за розміром кисті, погляд з ірисів.
(function () {
  const VER = '0.10.14';
  const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + VER;
  const MODELS = 'https://storage.googleapis.com/mediapipe-models';

  // Правдоподібність руки: ріже фантомні «руки» на обличчі
  function plausible(lm, face) {
    const sc = Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y);
    if (sc < 0.045) return false;
    if (face) {
      let inx = 0;
      for (const p of lm) if (p.x >= face.minx && p.x <= face.maxx && p.y >= face.miny && p.y <= face.maxy) inx++;
      const fw = Math.max(face.maxx - face.minx, face.maxy - face.miny);
      if (inx >= lm.length * 0.9 && sc < fw * 0.55) return false;
    }
    return true;
  }
  const ema = (o, v, k) => o + (v - o) * (k === undefined ? 0.3 : k);

  class AirHands {
    constructor() {
      this.hands = []; this.gestures = []; this.ready = false; this.running = false;
      this.error = null; this.HCONN = null; this.videoSize = { w: 0, h: 0 };
      this.mirror = true; this.gaze = null; this.faceBB = null;
      this.slots = [{ on: false, cnt: 0, z: 0 }, { on: false, cnt: 0, z: 0 }];
      this.hold = { name: '', since: 0, fired: 0 };
      this.two = null; this.shake = { last: 0, dir: 0, hits: 0, t: 0 };
      this.listeners = {}; this.wantFace = true; this.voice = null;
      this.transcript = ''; this.interim = '';
    }
    on(k, fn) { (this.listeners[k] || (this.listeners[k] = [])).push(fn); return this; }
    emit(k, d) { (this.listeners[k] || []).forEach(f => { try { f(d); } catch (e) {} }); }

    async start(opts) {
      if (this.running) return;
      this.video = opts.video;
      this.wantFace = opts.face !== false;
      try {
        const mp = await import(CDN + '/vision_bundle.mjs');
        const { FilesetResolver, GestureRecognizer, FaceLandmarker } = mp;
        const fsv = await FilesetResolver.forVisionTasks(CDN + '/wasm');
        this.HCONN = GestureRecognizer.HAND_CONNECTIONS;
        const mk = async (Cls, url, extra) => {
          let last = null;
          for (const d of ['GPU', 'CPU']) {
            try { return await Cls.createFromOptions(fsv, { baseOptions: { modelAssetPath: url, delegate: d }, runningMode: 'VIDEO', ...extra }); }
            catch (e) { last = e; }
          }
          throw last;
        };
        const st = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
        this.tracks = st.getVideoTracks();
        this.video.srcObject = st;
        this.video.muted = true;
        await this.video.play();
        this.GR = await mk(GestureRecognizer, MODELS + '/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
          { numHands: 2, minHandDetectionConfidence: 0.7, minHandPresenceConfidence: 0.7, minTrackingConfidence: 0.6 });
        this.ready = true; this.running = true;
        this.emit('ready', {});
        this.loop();
        if (this.wantFace) {
          mk(FaceLandmarker, MODELS + '/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            { numFaces: 1, outputFaceBlendshapes: false }).then(f => {
              this.FL = f;
              this.EYE = { r: idx(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE), l: idx(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE), ri: idx(FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS), li: idx(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS) };
            }).catch(() => {});
        }
      } catch (e) {
        this.error = (e && e.name) || 'помилка';
        this.emit('error', this.error);
        throw e;
      }
      function idx(conns) { const s = new Set(); for (const c of conns) { s.add(c.start); s.add(c.end); } return [...s]; }
    }
    stop() {
      this.running = false; this.ready = false;
      if (this.tracks) for (const t of this.tracks) { try { t.stop(); } catch (e) {} }
      this.tracks = null; this.hands = []; this.gestures = []; this.gaze = null;
      this.slots.forEach(s => { s.on = false; s.cnt = 0; });
      this.two = null;
      this.stopVoice();
      clearTimeout(this.watchdog);
      this.pending = false;
      if (this.video) this.video.srcObject = null;
    }
    loop() {
      if (!this.running) return;
      const step = () => {
        if (!this.running) return;
        const v = this.video;
        if (v && v.readyState >= 2 && v.videoWidth) {
          this.videoSize = { w: v.videoWidth, h: v.videoHeight };
          const t = performance.now();
          this.frame = (this.frame || 0) + 1;
          try {
            if (this.FL && this.frame % 3 === 0) this.postFace(this.FL.detectForVideo(v, t));
            else this.postGest(this.GR.recognizeForVideo(v, t), t);
          } catch (e) {}
        }
        // rVFC не спрацьовує на схованому/тротленому відео — тримаємо страховку rAF
        this.pending = true;
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
          this.video.requestVideoFrameCallback(() => { this.pending = false; step(); });
          clearTimeout(this.watchdog);
          this.watchdog = setTimeout(() => { if (this.pending && this.running) { this.pending = false; step(); } }, 400);
        } else requestAnimationFrame(() => { this.pending = false; step(); });
      };
      step();
    }
    postFace(res) {
      const lms = res && res.faceLandmarks;
      if (!lms || !lms.length) { this.gaze = null; this.faceBB = null; return; }
      const lm = lms[0];
      let a = 1, b = 0, c = 1, d = 0;
      for (const p of lm) { if (p.x < a) a = p.x; if (p.x > b) b = p.x; if (p.y < c) c = p.y; if (p.y > d) d = p.y; }
      this.faceBB = { minx: a, maxx: b, miny: c, maxy: d };
      if (!this.EYE) return;
      const eye = (ei, ii) => {
        let mnx = 1, mxx = 0, mny = 1, mxy = 0;
        for (const i of ei) { const p = lm[i]; if (p.x < mnx) mnx = p.x; if (p.x > mxx) mxx = p.x; if (p.y < mny) mny = p.y; if (p.y > mxy) mxy = p.y; }
        let cx = 0, cy = 0;
        for (const i of ii) { cx += lm[i].x; cy += lm[i].y; }
        return { ex: (mnx + mxx) / 2, ey: (mny + mxy) / 2, cx: cx / ii.length, cy: cy / ii.length, w: mxx - mnx };
      };
      const R = eye(this.EYE.r, this.EYE.ri), L = eye(this.EYE.l, this.EYE.li);
      const w = Math.max(1e-4, (R.w + L.w) / 2);
      const dx = ((R.cx - R.ex) + (L.cx - L.ex)) / 2 / w;
      const dy = ((R.cy - R.ey) + (L.cy - L.ey)) / 2 / w;
      const fx = (a + b) / 2, fy = (c + d) / 2;
      const gx = Math.min(1, Math.max(0, fx + dx * 1.9));
      const gy = Math.min(1, Math.max(0, fy + dy * 2.4));
      this.gaze = this.gaze
        ? { x: ema(this.gaze.x, gx, 0.25), y: ema(this.gaze.y, gy, 0.25) }
        : { x: gx, y: gy };
    }
    postGest(res, now) {
      const raw = (res && res.landmarks) || [];
      const hands = [], gests = [];
      for (let i = 0; i < raw.length; i++) {
        if (plausible(raw[i], this.faceBB)) { hands.push(raw[i]); gests.push((res.gestures && res.gestures[i]) || []); }
      }
      this.hands = hands; this.gestures = gests;
      const named = i => (gests[i] && gests[i][0] && gests[i][0].score > 0.6) ? gests[i][0].categoryName : '';
      // щіпка з гістерезисом і дебаунсом (2 кадри)
      for (let i = 0; i < 2; i++) {
        const s = this.slots[i], lm = hands[i];
        if (!lm || named(i) === 'Closed_Fist') {
          if (s.on) this.emit('pinchEnd', { slot: i, x: s.x, y: s.y });
          s.on = false; s.cnt = 0; continue;
        }
        const scale = Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) || 1e-6;
        const r = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / scale;
        const want = s.on ? r < 0.45 : r < 0.35;
        const px = this.mirror ? 1 - (lm[4].x + lm[8].x) / 2 : (lm[4].x + lm[8].x) / 2;
        const py = (lm[4].y + lm[8].y) / 2;
        s.z = ema(s.z, Math.max(-0.6, Math.min(1.2, (0.12 / scale - 1) * 0.9)));
        if (want) {
          s.cnt++;
          if (s.cnt >= 2 && !s.on) { s.on = true; s.x = px; s.y = py; this.emit('pinchStart', { slot: i, x: px, y: py, z: s.z }); }
          else if (s.on) { s.x = px; s.y = py; this.emit('pinchMove', { slot: i, x: px, y: py, z: s.z }); }
        } else {
          if (s.on) this.emit('pinchEnd', { slot: i, x: px, y: py });
          s.on = false; s.cnt = 0;
        }
      }
      // вказівний палець = курсор; Pointing_Up = малювання нитки
      if (hands.length) {
        const lm = hands[0];
        const tip = lm[8];
        const x = this.mirror ? 1 - tip.x : tip.x;
        this.point = { x, y: tip.y, drawing: named(0) === 'Pointing_Up' };
        this.emit('point', this.point);
      } else this.point = null;
      // дві руки: розвести — углиб, звести — назад
      if (hands.length >= 2) {
        const p = i => { const t = hands[i][8]; return { x: this.mirror ? 1 - t.x : t.x, y: t.y }; };
        const a = p(0), b = p(1);
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        if (!this.two) this.two = { d0: d, fired: 0, t: now };
        else {
          const ratio = d / (this.two.d0 || 1e-6);
          if (now - this.two.fired > 900) {
            if (ratio > 1.55) { this.two.fired = now; this.two.d0 = d; this.emit('spread', mid); }
            else if (ratio < 0.62) { this.two.fired = now; this.two.d0 = d; this.emit('pinchTogether', mid); }
          }
        }
        if (named(0) === 'Open_Palm' && named(1) === 'Open_Palm' && now - (this.palms2 || 0) > 1400) {
          this.palms2 = now; this.emit('twoPalms', mid);
        }
      } else this.two = null;
      // hold-жести: долоня / victory 1.2с
      let top = '';
      for (let i = 0; i < hands.length; i++) { const n = named(i); if (n) { top = n; break; } }
      if (top === 'Open_Palm' || top === 'Victory') {
        if (this.hold.name !== top) { this.hold.name = top; this.hold.since = now; }
        else if (now - this.hold.since > 1200 && now - this.hold.fired > 1800) {
          this.hold.fired = now;
          this.emit(top === 'Open_Palm' ? 'palmHold' : 'victoryHold', {});
        }
      } else this.hold.name = '';
      // кулак і потрясти
      const fi = hands.findIndex((_, i) => named(i) === 'Closed_Fist');
      if (fi >= 0) {
        const c = hands[fi][0];
        const x = this.mirror ? 1 - c.x : c.x;
        const sh = this.shake;
        if (sh.px !== undefined) {
          const dx = x - sh.px;
          if (Math.abs(dx) > 0.02) {
            const dir = dx > 0 ? 1 : -1;
            if (dir !== sh.dir) { sh.dir = dir; sh.hits++; sh.t = now; }
            if (sh.hits >= 4 && now - sh.last > 2200) { sh.last = now; sh.hits = 0; this.emit('fistShake', {}); }
          }
          if (now - sh.t > 900) sh.hits = 0;
        }
        sh.px = x;
      } else { this.shake.px = undefined; this.shake.hits = 0; }
    }
    // ── Голос ──
    startVoice(lang) {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) { this.emit('voiceError', 'not-supported'); return false; }
      if (this.voice) return true;
      const r = new SR();
      r.lang = lang || 'uk-UA'; r.continuous = true; r.interimResults = true;
      r.onresult = e => {
        let fin = '', intm = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const q = e.results[i];
          if (q.isFinal) fin += q[0].transcript; else intm += q[0].transcript;
        }
        this.interim = intm;
        if (fin) this.transcript = (this.transcript + ' ' + fin).trim();
        this.emit('speech', { final: fin.trim(), interim: intm, text: (this.transcript + ' ' + intm).trim() });
      };
      r.onend = () => { if (this.voice === r) { try { r.start(); } catch (e) {} } };
      r.onerror = () => {};
      try { r.start(); } catch (e) {}
      this.voice = r;
      return true;
    }
    stopVoice() {
      const r = this.voice; this.voice = null;
      if (r) { r.onend = null; try { r.stop(); } catch (e) {} }
      this.transcript = ''; this.interim = '';
    }
    resetVoice() { this.transcript = ''; this.interim = ''; }
    // cover-мапінг кадру на полотно
    cover(W, H) {
      const vw = this.videoSize.w || 640, vh = this.videoSize.h || 480;
      const s = Math.max(W / vw, H / vh), dw = vw * s, dh = vh * s;
      return { s, dw, dh, ox: (W - dw) / 2, oy: (H - dh) / 2 };
    }
    toScreen(nx, ny, W, H) {
      const c = this.cover(W, H);
      return { x: c.ox + nx * c.dw, y: c.oy + ny * c.dh };
    }
  }
  window.AirHands = new AirHands();
})();
