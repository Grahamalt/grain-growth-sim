// All page interactivity. Vanilla JS, no dependencies.
//
// Features:
//   1. Hamburger toggle for the mobile nav.
//   2. IntersectionObserver scroll-spy that adds .active to the nav link
//      whose section is topmost in the viewport. Native CSS smooth scroll
//      handles anchor navigation (html { scroll-behavior: smooth } in
//      style.css).
//   3. Showcase: 3-stop dopant slider + microstructure / scattering toggle
//      that swap the showcase image source. The image grid is six PNGs:
//        showcase_low_microstructure.png      showcase_low_scattering.png
//        showcase_medium_microstructure.png   showcase_medium_scattering.png
//        showcase_high_microstructure.png     showcase_high_scattering.png
//   4. Grain-growth timeline: scrubbable + auto-play over snapshot frames
//      declared in the data-frames attribute on #growth-animation.

(function () {

  // ---------- 1. Hamburger ----------
  const nav = document.getElementById('site-nav');
  const toggle = document.getElementById('nav-toggle');
  const navLinks = Array.from(document.querySelectorAll('#site-nav a[href^="#"]'));

  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    navLinks.forEach((a) => {
      a.addEventListener('click', () => {
        nav.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // ---------- 2. Scroll-spy ----------
  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter((s) => s !== null);

  if ('IntersectionObserver' in window && sections.length > 0) {
    const visible = new Set();
    const setActive = () => {
      if (visible.size === 0) return;
      let bestId = null, bestTop = Infinity;
      visible.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const top = el.getBoundingClientRect().top;
        if (top < bestTop) { bestTop = top; bestId = id; }
      });
      navLinks.forEach((a) => a.classList.toggle(
        'active', a.getAttribute('href') === '#' + bestId
      ));
    };
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) visible.add(e.target.id);
        else visible.delete(e.target.id);
      });
      setActive();
    }, { rootMargin: '-20% 0px -60% 0px', threshold: 0 });
    sections.forEach((s) => obs.observe(s));
  }

  // ---------- 3. Showcase: slider + toggle ----------
  const SHOWCASE_CASES = [
    {
      label: 'low',
      C: 0.05,
      D: 15.5,
      mu: '2.15×10⁻⁵',
      micro: 'assets/figures/showcase_low_microstructure.png',
      scat:  'assets/figures/showcase_low_scattering.png'
    },
    {
      label: 'medium',
      C: 0.20,
      D: 12.0,
      mu: '2.93×10⁻⁵',
      micro: 'assets/figures/showcase_medium_microstructure.png',
      scat:  'assets/figures/showcase_medium_scattering.png'
    },
    {
      label: 'high',
      C: 0.40,
      D: 7.9,
      mu: '4.16×10⁻⁵',
      micro: 'assets/figures/showcase_high_microstructure.png',
      scat:  'assets/figures/showcase_high_scattering.png'
    }
  ];

  const showcaseImg = document.getElementById('showcase-img');
  const showcaseCap = document.getElementById('showcase-caption');
  const dopantRange = document.getElementById('dopant-range');
  const stopLabels  = document.querySelectorAll('.dopant-slider .stop');
  const toggleBtns  = document.querySelectorAll('#showcase-interactive .toggle-btn');

  let currentIdx = 0;
  let currentView = 'micro';

  // Pre-load the other 5 images so transitions feel instant.
  SHOWCASE_CASES.forEach((c) => {
    [c.micro, c.scat].forEach((src) => { const i = new Image(); i.src = src; });
  });

  const renderShowcase = () => {
    if (!showcaseImg) return;
    const c = SHOWCASE_CASES[currentIdx];
    const src = currentView === 'micro' ? c.micro : c.scat;
    showcaseImg.src = src;
    showcaseImg.alt = currentView === 'micro'
      ? `Grain microstructure for the ${c.label}-dopant case (C_bulk = ${c.C}).`
      : `Scattering intensity field for the ${c.label}-dopant case (C_bulk = ${c.C}).`;

    if (showcaseCap) {
      showcaseCap.innerHTML =
        `<strong>C_bulk = ${c.C}</strong> &middot; ` +
        `&lang;D&rang; = ${c.D} lattice units &middot; ` +
        `&mu;_eff = ${c.mu} (1/lattice unit, illustrative)`;
    }

    stopLabels.forEach((s, i) => s.classList.toggle('active', i === currentIdx));
    if (dopantRange) dopantRange.setAttribute('aria-valuenow', String(currentIdx));
  };

  if (dopantRange) {
    dopantRange.addEventListener('input', (e) => {
      currentIdx = parseInt(e.target.value, 10) || 0;
      renderShowcase();
    });
  }

  // Allow clicking a stop label to jump there.
  stopLabels.forEach((s, i) => {
    s.style.cursor = 'pointer';
    s.addEventListener('click', () => {
      currentIdx = i;
      if (dopantRange) dopantRange.value = String(i);
      renderShowcase();
    });
  });

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

  // ---------- 4. Grain-growth animation timeline ----------
  const growth = document.getElementById('growth-animation');
  if (growth) {
    const frames = (growth.dataset.frames || '')
      .split(',')
      .map((s) => s.split('|'))
      .filter((p) => p.length === 2)
      .map(([src, mcs]) => ({ src: src.trim(), mcs: parseInt(mcs, 10) }));

    const gImg   = document.getElementById('growth-img');
    const gCap   = document.getElementById('growth-caption');
    const gRange = document.getElementById('growth-range');
    const gPlay  = document.getElementById('growth-play');
    const gStep  = document.getElementById('growth-step');

    if (frames.length > 0 && gImg && gRange) {
      // Pre-load all frames.
      frames.forEach((f) => { const i = new Image(); i.src = f.src; });

      gRange.max = String(frames.length - 1);
      let idx = 0;
      let playing = false;
      let timer = null;

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
        if (gPlay) gPlay.innerHTML = '▶ Play';
      };
      const play = () => {
        playing = true;
        if (gPlay) gPlay.innerHTML = '❚❚ Pause';
        timer = setInterval(() => {
          showFrame(idx + 1);
          if (idx === frames.length - 1) {
            // Pause briefly on the last frame, then loop.
            setTimeout(() => { if (playing) showFrame(0); }, 400);
          }
        }, 1100);
      };

      gRange.addEventListener('input', (e) => {
        stop();
        showFrame(parseInt(e.target.value, 10) || 0);
      });
      if (gPlay) gPlay.addEventListener('click', () => playing ? stop() : play());

      showFrame(0);
    }
  }

})();
