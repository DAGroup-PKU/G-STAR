'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- Text-to-image table (paper Table 1) ---------- */
// [model, comp, params, GenEval, DPG, PRISM, LongText, Qwen-Image, group]
const T2I = [
  ['Kolors 2.0', '—', '—', null, null, null, 0.258, null, 'prop'],
  ['GPT Image 1', '—', '—', 0.84, 85.15, 8.07, 0.956, 54.07, 'prop'],
  ['Seedream 3.0', '—', '—', 0.84, 88.27, 7.62, 0.896, null, 'prop'],
  ['HiDream-I1-Full', '16×', '17B', 0.83, 85.89, 7.26, 0.543, 43.08, 'diff'],
  ['FLUX.1 [dev]', '16×', '12B', 0.66, 83.84, 6.85, 0.607, 21.58, 'diff'],
  ['BAGEL', '16×', '14B', 0.88, 85.07, 6.20, 0.373, 42.04, 'diff'],
  ['Z-Image', '16×', '6B', 0.84, 88.14, 7.56, 0.935, 49.10, 'diff'],
  ['Qwen-Image', '16×', '20B', 0.87, 88.32, 7.41, 0.943, 49.23, 'diff'],
  ['OmniGen2', '16×', '4B', 0.86, 83.57, 6.48, 0.561, 40.70, 'diff'],
  ['SANA-1.5', '32×', '4.8B', 0.81, 84.70, 6.51, 0.050, 35.14, 'diff'],
  ['Janus-Pro-7B', '16×', '7B', 0.80, 84.19, 6.21, 0.019, 23.53, 'ar'],
  ['GLM-Image', '16×', '16B', 0.72, 84.78, 7.32, 0.952, 48.19, 'ar'],
  ['Emu3.5', '16×', '34.1B', 0.86, 88.26, 7.52, 0.976, 49.89, 'ar'],
  ['BitDance', '16×', '14B', 0.86, 88.28, 7.19, 0.837, 45.19, 'ar'],
  ['G-STAR-Pro-8B (ours)', '32×', '8B', 0.84, 83.38, 6.98, 0.935, 50.07, 'pro'],
  ['G-STAR-Flash-1.7B (ours)', '64×', '1.7B', 0.76, 80.18, 6.81, 0.853, 46.88, 'flash'],
];
const GROUP_LABEL = { prop: 'Proprietary', diff: 'Diffusion', ar: 'Autoregressive', ours: 'Ours' };
const DIGITS = [0, 0, 0, 2, 2, 2, 3, 2];
const table = { filter: 'all', sort: null };

function renderT2I() {
  const tbody = $('#t2i tbody');
  let rows = T2I.filter(r => {
    const g = r[8];
    if (g === 'pro' || g === 'flash') return true;
    return table.filter === 'all' || table.filter === g;
  });
  if (table.sort != null) {
    const c = table.sort;
    rows = [...rows].sort((a, b) => c === 0 ? a[0].localeCompare(b[0]) : (b[c] ?? -1) - (a[c] ?? -1));
  }
  // best value per score column among visible rows
  const best = {};
  for (let c = 3; c <= 7; c++) best[c] = Math.max(...rows.map(r => r[c] ?? -Infinity));

  tbody.replaceChildren();
  let lastGroup = null;
  for (const r of rows) {
    const g = (r[8] === 'pro' || r[8] === 'flash') ? 'ours' : r[8];
    if (table.sort == null && g !== lastGroup) {
      const tr = document.createElement('tr');
      tr.className = 'group';
      const td = document.createElement('td');
      td.colSpan = 8; td.textContent = GROUP_LABEL[g];
      tr.append(td); tbody.append(tr); lastGroup = g;
    }
    const tr = document.createElement('tr');
    if (r[8] === 'pro' || r[8] === 'flash') tr.className = r[8];
    r.slice(0, 8).forEach((v, i) => {
      const cell = document.createElement(i === 0 ? 'th' : 'td');
      if (i === 0) cell.scope = 'row';
      cell.textContent = v == null ? '—' : (i >= 3 ? v.toFixed(DIGITS[i]) : v);
      if (i >= 3 && v != null && v === best[i]) cell.classList.add('best');
      tr.append(cell);
    });
    tbody.append(tr);
  }
  $$('#t2i thead th').forEach((th, i) => th.classList.toggle('sorted', table.sort === i));
}
$$('[data-filter]').forEach(b => b.addEventListener('click', () => {
  table.filter = b.dataset.filter;
  $$('[data-filter]').forEach(o => o.setAttribute('aria-pressed', String(o === b)));
  renderT2I();
}));
$$('#t2i thead th[data-sort]').forEach(th => {
  th.tabIndex = 0;
  const go = () => { const c = +th.dataset.sort; table.sort = table.sort === c ? null : c; renderT2I(); };
  th.addEventListener('click', go);
  th.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
});
renderT2I();

/* ---------- radio-style segmented controls ---------- */
function segmented(container, attr, onChange) {
  const buttons = $$(`[data-${attr}]`, container);
  const pick = b => {
    buttons.forEach(o => o.setAttribute('aria-checked', String(o === b)));
    onChange(b.dataset[attr]);
  };
  buttons.forEach(b => b.addEventListener('click', () => pick(b)));
  return pick;
}

/* ---------- Co-scaling explorer ---------- */
(() => {
  const root = $('#coscale');
  const canvas = $('#cs-canvas');
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.src = 'assets/gen-panda.webp';
  const CODE_COLORS = ['#f4d27a', '#bfe3b4', '#5b9bd5'];   // yellow / green / blue, as in paper Fig. 2
  let s = 32, cursor = 0, timer = null;

  function draw() {
    const W = canvas.width, n = 1024 / s, cell = W / n;
    ctx.clearRect(0, 0, W, W);
    if (img.complete) ctx.drawImage(img, 0, 0, W, W);
    ctx.fillStyle = 'rgba(6,7,10,.35)'; ctx.fillRect(0, 0, W, W);
    // already-generated positions brighten
    const done = cursor % (n * n);
    ctx.save(); ctx.beginPath();
    for (let k = 0; k < done; k++) ctx.rect((k % n) * cell, Math.floor(k / n) * cell, cell, cell);
    ctx.clip();
    if (img.complete) ctx.drawImage(img, 0, 0, W, W);
    ctx.restore();
    // grid
    ctx.strokeStyle = 'rgba(255,255,255,.28)'; ctx.lineWidth = n > 40 ? .5 : 1;
    ctx.beginPath();
    for (let i = 1; i < n; i++) { const p = Math.round(i * cell) + .5; ctx.moveTo(p, 0); ctx.lineTo(p, W); ctx.moveTo(0, p); ctx.lineTo(W, p); }
    ctx.stroke();
    // current position
    ctx.strokeStyle = '#6cb6ff'; ctx.lineWidth = 3;
    ctx.strokeRect((done % n) * cell + 1.5, Math.floor(done / n) * cell + 1.5, cell - 3, cell - 3);
  }
  function tick() {
    const n = 1024 / s;
    // sweep the whole grid in ~6 s regardless of s, so the difference in step count is visible
    cursor = (cursor + Math.max(1, Math.round(n * n / 180))) % (n * n);
    draw();
  }
  function set(val) {
    s = +val; cursor = 0;
    const steps = (1024 / s) ** 2, G = 8 * (s / 32) ** 2;
    $('#cs-steps').textContent = steps.toLocaleString('en-US');
    $('#cs-g').textContent = G; $('#cs-g2').textContent = G;
    const codes = $('#cs-codes'); codes.replaceChildren();
    for (let g = 0; g < G; g++) {
      const i = document.createElement('i');
      i.style.background = CODE_COLORS[Math.floor(g / Math.max(1, G / 4)) % 3];
      i.style.animationDelay = `${g * 15}ms`;
      codes.append(i);
    }
    draw();
  }
  img.onload = draw;
  segmented(root, 's', set);
  set(32);
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !reduceMotion && !timer) timer = setInterval(tick, 33);
    if (!e.isIntersecting && timer) { clearInterval(timer); timer = null; }
  });
  io.observe(canvas);
})();

/* ---------- Prediction-head explorer (paper Table 4) ---------- */
(() => {
  const HEADS = {
    ar:  { lat: 6.44, ge: '0.8386', dpg: '82.48', pr: '5.46', note: 'G full-head passes, one after another: accurate, but slow.' },
    par: { lat: 1.77, ge: '0.7800', dpg: '79.98', pr: '4.72', note: 'One pass emits all codes at once: fast, but codes can collide.' },
    sar: { lat: 3.41, ge: '0.8253', dpg: '82.60', pr: '5.46', note: 'One parallel pass, then lightweight one-layer causal steps.' },
  };
  const G = 16, MAX = 6.44;
  const tl = $('#hd-timeline');
  const pct = t => `${(t / MAX) * 96 + 2}%`;

  function block(x, w, y, h, c, d) {
    const b = document.createElement('div');
    b.className = 'blk';
    b.style.cssText = `--x:${pct(x)};--w:calc(${(w / MAX) * 96}% - 2px);--y:${y}px;--h:${h}px;--c:${c};--d:${d}ms`;
    tl.append(b);
  }
  function render(kind) {
    const h = HEADS[kind];
    $('#hd-lat').textContent = `${h.lat.toFixed(2)} s`;
    $('#hd-ge').textContent = h.ge; $('#hd-dpg').textContent = h.dpg; $('#hd-pr').textContent = h.pr;
    $('#hd-note').textContent = h.note;
    tl.replaceChildren();
    const codeY = 104;
    if (kind === 'ar') {
      const w = h.lat / G;
      for (let g = 0; g < G; g++) {
        block(g * w, w, 28, 52, '#5b9bd5', g * 45);
        block((g + 1) * w - 0.07, 0.07, codeY, 22, '#f4d27a', g * 45 + 120);
      }
    } else if (kind === 'par') {
      block(0, h.lat, 28, 52, '#ffa02e', 0);
      for (let g = 0; g < G; g++) block(h.lat - 0.07, 0.07, codeY - 4 + (g % 4) * 3, 18, g % 2 ? '#e5484d' : '#f4d27a', 250);
    } else {
      const trunk = 1.77, step = (h.lat - trunk) / G;      // trunk ≈ one parallel pass; rest split over G light steps
      block(0, trunk, 28, 52, '#7fcf8f', 0);
      for (let g = 0; g < G; g++) {
        block(trunk + g * step, step, 40, 28, '#5b9bd5', 250 + g * 40);
        block(trunk + (g + 1) * step - 0.05, 0.05, codeY, 22, '#f4d27a', 290 + g * 40);
      }
    }
    const end = document.createElement('div');
    end.className = 'end';
    end.style.cssText = `--x:${pct(h.lat)};--d:${kind === 'ar' ? 760 : kind === 'sar' ? 900 : 300}ms`;
    end.innerHTML = `<span>${h.lat.toFixed(2)} s</span>`;
    if (h.lat / MAX > .8) end.querySelector('span').style.cssText = 'left:auto;right:6px';
    tl.append(end);
  }
  segmented($('#heads'), 'h', render);
  render('sar');
})();

/* ---------- Speed bars + counters animate on view ---------- */
const onView = (el, fn) => {
  const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { fn(); io.disconnect(); } }, { threshold: .35 });
  io.observe(el);
};
onView($('#speed'), () => $('#speed').classList.add('on'));
if (!reduceMotion) {
  $$('.metrics strong[data-count]').forEach(el => {
    const target = +el.dataset.count, dec = +el.dataset.dec;
    const unit = el.querySelector('small')?.outerHTML ?? '';
    el.innerHTML = (0).toFixed(dec) + unit;
    onView(el, () => {
      const t0 = performance.now();
      const step = now => {
        const p = Math.min(1, (now - t0) / 1100), e = 1 - (1 - p) ** 3;
        el.innerHTML = (target * e).toFixed(dec) + unit;
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  });
}

/* ---------- Before/after sliders ---------- */
$$('.compare input').forEach(input => input.addEventListener('input', () =>
  input.closest('.compare').style.setProperty('--split', `${input.value}%`)));

/* ---------- Lightbox ---------- */
const lb = $('#lightbox');
$$('[data-img]').forEach(b => b.addEventListener('click', () => {
  $('img', lb).src = b.dataset.img;
  $('img', lb).alt = $('img', b).alt;
  $('p', lb).textContent = b.dataset.cap;
  lb.showModal();
}));
$('#lb-close').addEventListener('click', () => lb.close());
lb.addEventListener('click', e => { if (e.target === lb) lb.close(); });

/* ---------- Copy BibTeX ---------- */
$('#copy-bib').addEventListener('click', async () => {
  const btn = $('#copy-bib');
  try { await navigator.clipboard.writeText($('#bibtex').textContent); btn.textContent = 'Copied'; }
  catch { btn.textContent = 'Select & copy'; }
  setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
});

/* ---------- Highlight the current section in the nav ---------- */
const navLinks = $$('.topbar nav a');
const spy = new IntersectionObserver(entries => {
  for (const e of entries) if (e.isIntersecting)
    navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`));
}, { rootMargin: '-45% 0px -50% 0px' });
navLinks.forEach(a => { const t = $(a.getAttribute('href')); if (t) spy.observe(t); });
