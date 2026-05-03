// Dashboard interactivity. Vanilla JS, no dependencies.
// 1. Tab switching for the reading panel.
// 2. Showcase: 3-stop dopant slider + microstructure/scattering view toggle.
//    Looks for these images in assets/figures/:
//      showcase_low_microstructure.png    showcase_low_scattering.png
//      showcase_medium_microstructure.png showcase_medium_scattering.png
//      showcase_high_microstructure.png   showcase_high_scattering.png
// 3. Grain-growth animation: scrubbable timeline + play/pause over the
//    snapshot frames declared in #growth-animation[data-frames].

(function () {

  // ---------- 1. Tabs ----------
  const tabs = Array.from(document.querySelectorAll('#tabstrip button[role="tab"]'));
  const panels = tabs
    .map((t) => document.getElementById(t.getAttribute('aria-controls')))
    .filter(Boolean);

  const activateTab = (tab) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    panels.forEach((p) => {
      const on = p.id === tab.getAttribute('aria-controls');
      p.classList.toggle('active', on);
      if (on) p.removeAttribute('hidden');
      else    p.setAttribute('hidden', '');
    });
    // MathJax may need to typeset the panel the first time it appears.
    if (window.MathJax && window.MathJax.typesetPromise) {
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      if (panel) window.MathJax.typesetPromise([panel]);
    }
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', (e) => {
      // Arrow-key tablist navigation.
      const i = tabs.indexOf(tab);
      let next = null;
      if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
      if (e.key === 'ArrowLeft')  next = tabs[(i - 1 + tabs.length) % tabs.length];
      if (next) { e.preventDefault(); next.focus(); activateTab(next); }
    });
  });

  // ---------- 2. Showcase: slider + view toggle ----------
  const SHOWCASE_CASES = [
    { label: 'low',    C: 0.05, D: 15.5, mu: '2.15&times;10<sup>&minus;5</sup>',
      micro: 'assets/figures/showcase_low_microstructure.png',
      scat:  'assets/figures/showcase_low_scattering.png' },
    { label: 'medium', C: 0.20, D: 12.0, mu: '2.93&times;10<sup>&minus;5</sup>',
      micro: 'assets/figures/showcase_medium_microstructure.png',
      scat:  'assets/figures/showcase_medium_scattering.png' },
    { label: 'high',   C: 0.40, D: 7.9,  mu: '4.16&times;10<sup>&minus;5</sup>',
      micro: 'assets/figures/showcase_high_microstructure.png',
      scat:  'assets/figures/showcase_high_scattering.png' }
  ];

  const showcaseImg = document.getElementById('showcase-img');
  const showcaseCap = document.getElementById('showcase-caption');
  const dopantRange = document.getElementById('dopant-range');
  const stopLabels  = document.querySelectorAll('.dopant-slider .stop');
  const toggleBtns  = document.querySelectorAll('#showcase-interactive .toggle-btn');

  let currentIdx = 0;
  let currentView = 'micro';

  // Pre-load all showcase variants so swaps are instant.
  SHOWCASE_CASES.forEach((c) => {
    [c.micro, c.scat].forEach((src) => { const i = new Image(); i.src = src; });
  });

  const renderShowcase = () => {
    if (!showcaseImg) return;
    const c = SHOWCASE_CASES[currentIdx];
    showcaseImg.src = currentView === 'micro' ? c.micro : c.scat;
    showcaseImg.alt = currentView === 'micro'
      ? `Microstructure for the ${c.label}-dopant case (C_bulk = ${c.C}).`
      : `Scattering field for the ${c.label}-dopant case (C_bulk = ${c.C}).`;
    if (showcaseCap) {
      showcaseCap.innerHTML =
        `<strong>C_bulk = ${c.C}</strong> &middot; ` +
        `&lang;D&rang; = ${c.D} &middot; ` +
        `&mu;_eff = ${c.mu}`;
    }
    stopLabels.forEach((s, i) => s.classList.toggle('active', i === currentIdx));
    if (dopantRange) dopantRange.value = String(currentIdx);
  };

  if (dopantRange) {
    dopantRange.addEventListener('input', (e) => {
      currentIdx = parseInt(e.target.value, 10) || 0;
      renderShowcase();
    });
  }
  stopLabels.forEach((s, i) => s.addEventListener('click', () => {
    currentIdx = i; renderShowcase();
  }));
  toggleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      currentView = btn.dataset.view;
      toggleBtns.forEach((b) => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      renderShowcase();
    });
  });
  if (showcaseImg) renderShowcase();

  // ---------- 3. Grain-growth animation ----------
  const growth = document.getElementById('growth-animation');
  if (!growth) return;
  const frames = (growth.dataset.frames || '')
    .split(',')
    .map((s) => s.split('|'))
    .filter((p) => p.length === 2)
    .map(([src, mcs]) => ({ src: src.trim(), mcs: parseInt(mcs, 10) }));
  if (frames.length === 0) return;

  const gImg   = document.getElementById('growth-img');
  const gCap   = document.getElementById('growth-caption');
  const gRange = document.getElementById('growth-range');
  const gPlay  = document.getElementById('growth-play');
  const gStep  = document.getElementById('growth-step');
  if (!gImg || !gRange) return;

  frames.forEach((f) => { const i = new Image(); i.src = f.src; });
  gRange.max = String(frames.length - 1);

  let idx = 0, playing = false, timer = null;
  const showFrame = (i) => {
    idx = ((i % frames.length) + frames.length) % frames.length;
    const f = frames[idx];
    gImg.src = f.src;
    if (gCap)  gCap.textContent  = `MCS = ${f.mcs}`;
    if (gStep) gStep.textContent = `step ${idx + 1} of ${frames.length}`;
    gRange.value = String(idx);
  };
  const stop = () => {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
    if (gPlay) gPlay.innerHTML = '&#9654; Play';
  };
  const play = () => {
    playing = true;
    if (gPlay) gPlay.innerHTML = '&#10074;&#10074; Pause';
    timer = setInterval(() => {
      showFrame(idx + 1);
      if (idx === frames.length - 1) {
        setTimeout(() => { if (playing) showFrame(0); }, 400);
      }
    }, 1100);
  };
  gRange.addEventListener('input', (e) => {
    stop(); showFrame(parseInt(e.target.value, 10) || 0);
  });
  if (gPlay) gPlay.addEventListener('click', () => playing ? stop() : play());
  showFrame(0);

})();
