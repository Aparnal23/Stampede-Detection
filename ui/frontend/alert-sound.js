/**
 * Alert chimes: Web Audio (if unlocked via user gesture) with HTMLAudioElement + WAV fallback.
 * Browsers block audio until a user gesture; call unlockFromCheckbox() on the "Alert sound" checkbox.
 */
(function () {
  const AC = window.AudioContext || window.webkitAudioContext;

  let webCtx = null;
  let htmlAudio = null;

  function scheduleWebChime(ctx) {
    const t0 = ctx.currentTime;
    function beep(start, freq) {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = "sine";
      o.frequency.value = freq;
      g.gain.setValueAtTime(0, start);
      g.gain.linearRampToValueAtTime(0.12, start + 0.02);
      g.gain.linearRampToValueAtTime(0.0001, start + 0.24);
      o.connect(g);
      g.connect(ctx.destination);
      o.start(start);
      o.stop(start + 0.24);
    }
    beep(t0, 784);
    beep(t0 + 0.18, 988);
  }

  function buildAlertWavBuffer() {
    const sampleRate = 22050;
    const n = Math.floor(sampleRate * 0.5);
    const samples = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / sampleRate;
      let v = 0;
      if (t < 0.2) {
        v =
          Math.sin(2 * Math.PI * 784 * t) *
          0.28 *
          Math.min(1, t * 30) *
          Math.min(1, (0.2 - t) * 40);
      } else if (t > 0.22 && t < 0.45) {
        const t2 = t - 0.22;
        v =
          Math.sin(2 * Math.PI * 988 * t2) *
          0.28 *
          Math.min(1, t2 * 35) *
          Math.min(1, (0.23 - t2) * 35);
      }
      samples[i] = v;
    }
    const buffer = new ArrayBuffer(44 + n * 2);
    const view = new DataView(buffer);
    const w = (o, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
    };
    w(0, "RIFF");
    view.setUint32(4, 36 + n * 2, true);
    w(8, "WAVE");
    w(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    w(36, "data");
    view.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return buffer;
  }

  function getHtmlAudio() {
    if (!htmlAudio) {
      const blob = new Blob([buildAlertWavBuffer()], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      htmlAudio = new Audio(url);
      htmlAudio.preload = "auto";
    }
    return htmlAudio;
  }

  async function playHtml5() {
    const a = getHtmlAudio();
    a.currentTime = 0;
    await a.play();
  }

  async function playWeb() {
    if (!webCtx) throw new Error("no web context");
    await webCtx.resume();
    if (webCtx.state !== "running") throw new Error("suspended");
    scheduleWebChime(webCtx);
  }

  async function play() {
    if (webCtx) {
      try {
        await playWeb();
        return;
      } catch (_) {
        /* fall through to HTML5 */
      }
    }
    await playHtml5();
  }

  function unlockFromCheckbox(checkbox) {
    if (!checkbox) return;
    checkbox.addEventListener("change", () => {
      if (!checkbox.checked) return;
      void (async () => {
        if (AC) {
          if (!webCtx) webCtx = new AC();
          try {
            await webCtx.resume();
            scheduleWebChime(webCtx);
            return;
          } catch (_) {
            /* try HTML5 in same gesture */
          }
        }
        try {
          await playHtml5();
        } catch (e) {
          console.warn("AlertSound: could not unlock", e);
        }
      })();
    });
  }

  window.AlertSound = { unlockFromCheckbox, play };
})();
