/**
 * Animated shader backdrop for the glass chrome.
 *
 * WHY THIS EXISTS
 * The glass surfaces (topbar, history rail) are `backdrop-filter` over a pure
 * black page. That is the weakest case for glass: blurring flat black yields
 * flat black, so the surfaces read as translucent rectangles with nothing
 * behind them. Something has to be there for the blur to actually blur.
 *
 * The original approach was drifting CSS colour blobs. This replaces them
 * with one full-screen fragment shader, which gives the chrome a live,
 * slowly-varying field to sample.
 *
 * WHY RAW WEBGL
 * The reference implementation this is derived from used THREE.js to draw a
 * single full-screen quad. That is a dependency (and a CDN one, which
 * `script-src 'self'` blocks outright) for a job that is one triangle and one
 * fragment shader. No build step, no vendored library, no network.
 *
 * WHY IT IS COLOURED
 * The app's chrome is achromatic — provider identity is carried by labels, not
 * hue — but the backdrop is not, and those are separate decisions. The glass
 * recipe depends on the distinction: --glass-blur pairs blur with saturate()
 * in both themes, and that saturation exists precisely to pull colour out of
 * whatever sits behind the surface. Point it at a greyscale field and the
 * saturate() half of the recipe is dead weight.
 *
 * So the backdrop carries the colour while the chrome stays neutral. The two
 * never mix legibly on screen: the glass blurs this by 12-24px, so what
 * reaches the topbar is a soft wash, not readable hue.
 *
 * WHY IT FOLLOWS THE THEME
 * A dark field behind white glass would read as dirt smudged under the
 * surface, so the shader renders two mappings of the same drifting pattern:
 * the tone-mapped dark field, and an airy inverse of it for white mode. The
 * `lightness` uniform blends between them, eased over ~0.6s so toggling the
 * theme dissolves instead of cutting. Scaling the dark field (rather than
 * re-shaping it) keeps the hue variation pastel and the motion identical in
 * both themes. Which mapping is live follows `data-theme`, falling back to
 * the OS preference — watched live, so every page's toggle is honoured with
 * no per-page wiring.
 *
 * The shader is GLSL ES 1.00 (`attribute`, `gl_FragColor`), which both WebGL 1
 * and WebGL 2 accept, so one source works on either context.
 */

/* ---------- Shader source ---------- */

/* One oversized triangle covers clip space with no index buffer and no seam
   down the diagonal that a two-triangle quad would have. */
const VERT = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const FRAG = `
precision highp float;

uniform vec2  resolution;
uniform float time;
uniform float xScale;
uniform float yScale;
uniform float distortion;
uniform float intensity;
uniform float chroma;
/* 0 renders the dark field, 1 the white-mode field. Driven by the page theme
   (see themeIsLight), eased per frame so toggling cross-fades instead of
   flashing. */
uniform float lightness;

float lum3(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

/* 1/abs() of a sine gives the thin bright line the pattern is built from, but
   dividing by the raw value spikes to near-white every time the wave crosses
   zero — a hard grid of blown-out lines, far too aggressive behind glass.
   The epsilon floors that spike into a soft glow instead. */
float line(float y) {
  return 0.05 / (abs(y) + 0.035);
}

void main() {
  vec2 p = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);

  /* Radial lens term: pinches the bands into a bulge at the centre. */
  float d = length(p) * distortion;

  /* Three wavefronts at slightly different radii. Keeping all three is what
     makes them interfere; giving each its own channel is what turns the
     moire into chromatic fringes. Collapsing these to a single luminance
     value loses that, and leaves the glass's own saturate() with nothing to
     amplify — the two are the same decision, not two. */
  float rx = p.x * (1.0 + d);
  float gx = p.x;
  float bx = p.x * (1.0 - d);

  float r = line(p.y + sin((rx + time) * xScale) * yScale);
  float g = line(p.y + sin((gx + time) * xScale) * yScale);
  float b = line(p.y + sin((bx + time) * xScale) * yScale);

  vec3 v = vec3(r, g, b) * intensity;

  /* Pull the field back toward its own luminance.
     This is the single most important control in the whole file, and it is
     about legibility rather than taste. Saturated colour sitting behind small
     grey text is unreadable even when the WCAG luminance ratio looks fine: a
     vivid yellow and a mid grey at the same luminance do not read as the same
     weight, and the ratio formula cannot see that. Measured with full chroma,
     the composer hint sat at 1.93:1 against a [117,122,35] fringe — the field
     was dim enough and still killed it.
     Blending toward luminance keeps the colour visible as a tint while
     restoring the luminance separation the text depends on. */
  v = mix(vec3(lum3(v)), v, chroma);

  /* Reinhard tone map, per channel. Rolls off toward 1 instead of clipping,
     so peaks stay readable as bright fringes rather than saturating to flat
     white — which would wash the hue out exactly where it is strongest. */
  vec3 dark = v / (1.0 + v);

  /* White-mode field: the same drifting pattern, inverted into an airy range.
     Scaling the dark field (rather than re-shaping it) keeps the hue variation
     pastel and the motion identical across themes, while the bright floor
     keeps dark text contrast high behind white glass. */
  vec3 light = 1.0 - dark * 0.55;

  gl_FragColor = vec4(mix(dark, light, lightness), 1.0);
}
`;

/* ---------- Setup helpers ---------- */

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}`);
  }
  return sh;
}

function link(gl, vsSrc, fsSrc) {
  const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  // The shaders are reference-counted by the program; drop our handles so they
  // can be collected once attached.
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`program link failed: ${log}`);
  }
  return prog;
}

/* ---------- Public API ---------- */

/** Numeric knobs, and nothing else, may be set from the query string. */
const TUNABLE = ['xScale', 'yScale', 'distortion', 'intensity', 'chroma', 'speed', 'renderScale', 'maxDpr'];

function defaults() {
  return {
    /* These are NOT the source sample's values, and the difference is
       measured, not aesthetic. Sampled off real screenshots of the composed
       page — pixels inside the topbar, after the glass — because the canvas
       reading lies: a field can carry plenty of chroma in the canvas and still
       deliver grey to the surface, since `blur(24px)` averages neighbouring
       R/G/B fringes into one another.

       Source sample (xScale 1.0, distortion 0.05, intensity 1) measured 0.75
       mean channel spread inside the topbar: no colour at all.

       The fix is distortion, not frequency. Widening the bands was tried
       first and made things strictly worse (0.63), because `distortion` is
       what separates the channels in the first place — `d = length(p) *
       distortion` is the only thing pushing red and blue to different radii,
       so shrinking it collapses all three onto the same value. Raising it to
       0.45 while leaving xScale at the original 1.0 reached 23.35.

       xScale stays at the source value: every attempt to widen the bands
       (0.5, 0.25, 0.15) scored below leaving it alone. */
    xScale: 1.0,
    yScale: 0.8,
    distortion: 0.45,
    /* How bright the field renders before the glass bounds it. Raised from 0.5
       — the field read dim behind the glass. The dark scrim in theme.css was
       densified (0.58 -> 0.7) to match, because the composer hint had only
       4.65:1 of headroom at 0.5 and this level otherwise drops it under 4.5. */
    intensity: 0.6,
    /* How much of the field's own hue survives. 0 is greyscale (which is what
       this started as, and which killed the glass's saturate() for nothing),
       1 is the fully saturated fringe pattern. Raised from 0.5 for a more
       vivid field. Measured against real pixels, the composer hint's ratio
       does not fall as chroma rises — saturated fringes move off the mid-grey
       luminance that competes with grey text — so this is limited by
       legibility by eye, not by the ratio formula. 0.8 is visibly richer and
       still leaves the glass's saturate() something to work with. */
    chroma: 0.8,
    speed: 0.35,
    /* The glass blurs this at 20-24px, so rendering below 1:1 is free
       visually and cuts fragment cost roughly quadratically. */
    renderScale: 0.6,
    maxDpr: 2,
  };
}

/**
 * Optional per-load overrides from the query string, e.g.
 * `?fx=xScale:0.3,distortion:0.028,intensity:2.6`
 *
 * This exists because the right values cannot be chosen by eye. A field can
 * carry plenty of chroma in the raw canvas and still come out grey behind the
 * glass, so tuning it means measuring the composited result — which needs a
 * way to sweep parameters without editing source and reloading every time.
 */
function overridesFromQuery() {
  const out = {};
  let raw;
  try {
    raw = new URLSearchParams(location.search).get('fx');
  } catch {
    return out;
  }
  if (!raw) return out;
  for (const pair of raw.split(',')) {
    const i = pair.indexOf(':');
    if (i < 1) continue;
    const key = pair.slice(0, i).trim();
    const num = Number(pair.slice(i + 1));
    // This string is user-controlled and goes straight into a uniform, so it
    // is filtered to known keys and finite numbers.
    if (TUNABLE.includes(key) && Number.isFinite(num)) out[key] = num;
  }
  return out;
}

/**
 * Whether the page is currently in its white (light) theme. An explicit
 * `data-theme` choice wins; otherwise the OS preference decides. Unknown
 * means the dark field, which preserves the original look.
 */
function themeIsLight() {
  const explicit = document.documentElement?.dataset?.theme;
  if (explicit === 'light') return true;
  if (explicit === 'dark') return false;
  try {
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ?? false;
  } catch {
    return false;
  }
}

/**
 * Mounts the animated backdrop on `canvas`.
 *
 * Degrades rather than throws: no WebGL, a failed compile, or a lost context
 * all leave the page in its normal state (pure black, glass over nothing),
 * which is exactly how it looked before this existed.
 *
 * Precedence: defaults < `?fx=` query string < explicit `opts`.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} [opts]
 * @returns {{destroy: () => void, renderOnce: () => void}|null}
 */
export function initShaderBackdrop(canvas, opts = {}) {
  const {
    xScale,
    yScale,
    distortion,
    intensity,
    chroma,
    speed,
    renderScale,
    maxDpr,
  } = { ...defaults(), ...overridesFromQuery(), ...opts };

  const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');

  const gl =
    canvas.getContext('webgl', { alpha: false, antialias: false, depth: false, stencil: false, powerPreference: 'low-power' }) ||
    canvas.getContext('experimental-webgl', { alpha: false, antialias: false, depth: false, stencil: false });

  if (!gl) {
    console.warn('[shader] no WebGL context; glass will sit on flat black');
    return null;
  }

  let program;
  try {
    program = link(gl, VERT, FRAG);
  } catch (err) {
    console.warn('[shader]', err.message);
    return null;
  }

  // Theme drives the dark/light mapping (see the `lightness` uniform): an
  // explicit page choice wins, otherwise the OS preference decides.
  let lightCur = themeIsLight() ? 1 : 0;
  let lightTarget = lightCur;

  // Attribute/uniform locations and static values, (re)bound here and after a
  // context restore — every GL handle dies with a lost context, including the
  // buffer, so this recreates all of them rather than reusing stale ones.
  let buf = null;
  let aPos, uResolution, uTime, uXScale, uYScale, uDistortion, uIntensity, uChroma, uLight;
  function bindAll() {
    gl.useProgram(program);
    buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    // Single oversized triangle: 3 verts x 2 floats.
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    aPos = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    uResolution = gl.getUniformLocation(program, 'resolution');
    uTime = gl.getUniformLocation(program, 'time');
    uXScale = gl.getUniformLocation(program, 'xScale');
    uYScale = gl.getUniformLocation(program, 'yScale');
    uDistortion = gl.getUniformLocation(program, 'distortion');
    uIntensity = gl.getUniformLocation(program, 'intensity');
    uChroma = gl.getUniformLocation(program, 'chroma');
    uLight = gl.getUniformLocation(program, 'lightness');
    gl.uniform1f(uXScale, xScale);
    gl.uniform1f(uYScale, yScale);
    gl.uniform1f(uDistortion, distortion);
    gl.uniform1f(uIntensity, intensity);
    gl.uniform1f(uChroma, chroma);
    gl.uniform1f(uLight, lightCur);
  }
  bindAll();

  let width = 0;
  let height = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, maxDpr) * renderScale;
    // Round rather than floor: sub-pixel sizes make some drivers pick a
    // different internal format, which shows up as a flicker on resize.
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (w === width && h === height) return false;
    width = w;
    height = h;
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
    return true;
  }

  function draw(time) {
    gl.uniform2f(uResolution, width, height);
    gl.uniform1f(uTime, time);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Seconds of shader time. Driven by elapsed wall time rather than a fixed
  // per-frame increment, so the pattern moves at the same speed on a 60Hz and
  // a 144Hz display instead of running 2.4x fast on the latter.
  let clock = 0;
  let start = null;
  let raf = 0;
  let running = false;

  let last = null;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (start === null) start = now;
    // Clamped: edge cases (a restart racing a timestamp jump) could otherwise
    // hand the theme easing a seconds-long dt and snap it in one frame.
    const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.1);
    last = now;
    clock = (now - start) / 1000;
    // Theme cross-fade: ~0.6s so toggling feels like a dissolve, not a cut.
    if (lightCur !== lightTarget) {
      const step = dt / 0.6;
      lightCur += Math.sign(lightTarget - lightCur) * Math.min(Math.abs(lightTarget - lightCur), step);
      gl.uniform1f(uLight, lightCur);
    }
    resize();
    draw(clock * speed);
  }

  function startLoop() {
    if (running || reduceMotion?.matches) return;
    running = true;
    raf = requestAnimationFrame(frame);
  }

  function stopLoop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  /** Draw exactly one frame. Used for the reduced-motion static state. */
  function renderOnce() {
    // Static contexts don't ease: snap straight to the target theme.
    lightCur = lightTarget;
    gl.uniform1f(uLight, lightCur);
    resize();
    draw(0);
  }

  // Pausing on a hidden tab matters more than usual here: a full-screen
  // fragment shader on a hidden tab is pure wasted battery, and browsers do
  // not throttle requestAnimationFrame for you on every platform.
  const onVisibility = () => {
    if (document.hidden) {
      stopLoop();
    } else {
      // Restart resets the clock so the pattern does not jump forward by the
      // whole time the tab was hidden.
      start = null;
      last = null;
      startLoop();
    }
  };
  document.addEventListener('visibilitychange', onVisibility);

  // A lost context leaves every GL object invalid. Rather than trying to
  // resurrect the old ones, tear down and rebuild from scratch.
  const onLost = (e) => {
    e.preventDefault();
    stopLoop();
  };
  const onRestored = () => {
    // Re-link: the old program and buffer handles died with the context.
    try {
      program = link(gl, VERT, FRAG);
    } catch {
      return;
    }
    bindAll();
    width = 0;
    height = 0;
    start = null;
    last = null;
    if (!document.hidden) startLoop();
  };
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  // Resize coalesced to one call per frame, so a drag-resize does not queue a
  // reallocation per intermediate width.
  let resizeQueued = false;
  const onResize = () => {
    if (resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => {
      resizeQueued = false;
      if (resize()) {
        // Keep the pattern filling the screen while the loop is stopped.
        if (!running) renderOnce();
      }
    });
  };
  window.addEventListener('resize', onResize);

  // Reduced motion: one static frame, and stop. Also react if the user flips
  // the OS setting while the page is open.
  const onMotionChange = () => {
    if (reduceMotion?.matches) {
      stopLoop();
      renderOnce();
    } else {
      start = null;
      last = null;
      startLoop();
    }
  };
  reduceMotion?.addEventListener?.('change', onMotionChange);

  // Follow the page theme live. Every page's toggle writes `data-theme`
  // (chat, settings, image studio alike), so one observer covers them all with
  // no per-page wiring; the OS preference is followed only while no explicit
  // choice is set.
  const themeObs = new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.attributeName === 'data-theme') {
        lightTarget = themeIsLight() ? 1 : 0;
        break;
      }
    }
  });
  themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  let osWatcher = null;
  try {
    osWatcher = window.matchMedia?.('(prefers-color-scheme: light)') ?? null;
  } catch {
    osWatcher = null;
  }
  const onOsTheme = () => {
    // An explicit choice always wins; only follow the OS while unset.
    if (document.documentElement?.dataset?.theme) return;
    lightTarget = themeIsLight() ? 1 : 0;
  };
  osWatcher?.addEventListener?.('change', onOsTheme);

  if (reduceMotion?.matches) {
    renderOnce();
  } else {
    startLoop();
  }

  return {
    renderOnce,
    destroy() {
      stopLoop();
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('resize', onResize);
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      reduceMotion?.removeEventListener?.('change', onMotionChange);
      themeObs.disconnect();
      osWatcher?.removeEventListener?.('change', onOsTheme);
      gl.deleteProgram(program);
      gl.deleteBuffer(buf);
    },
  };
}
