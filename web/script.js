// Hamburger toggle and scroll-spy active-section highlighting.

(function () {
  const nav = document.getElementById('site-nav');
  const toggle = document.getElementById('nav-toggle');
  const links = Array.from(document.querySelectorAll('#site-nav a[href^="#"]'));

  // Mobile nav: hamburger toggles the .open class on #site-nav.
  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // Close the menu after clicking any link.
    links.forEach((a) => {
      a.addEventListener('click', () => {
        nav.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // Scroll-spy: highlight the nav link whose target section is closest to
  // the top of the viewport (just under the sticky nav).
  const sections = links
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter((s) => s !== null);

  if (!('IntersectionObserver' in window) || sections.length === 0) return;

  const linkFor = (id) =>
    links.find((a) => a.getAttribute('href') === '#' + id);

  // Track currently visible sections; pick the topmost one.
  const visible = new Set();
  const setActive = () => {
    if (visible.size === 0) return;
    let best = null;
    let bestTop = Infinity;
    visible.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const top = el.getBoundingClientRect().top;
      if (top < bestTop) { bestTop = top; best = id; }
    });
    links.forEach((a) => a.classList.toggle(
      'active', a.getAttribute('href') === '#' + best
    ));
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) visible.add(entry.target.id);
      else visible.delete(entry.target.id);
    });
    setActive();
  }, {
    rootMargin: '-20% 0px -60% 0px',
    threshold: 0
  });

  sections.forEach((s) => observer.observe(s));
})();
