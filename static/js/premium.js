/* DiaCare Premium Interactions */

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initReveal();
  initTabs();
  initRiskGauge();
});

function initNav() {
  const nav = document.querySelector('.premium-nav');
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');

  if (nav) {
    window.addEventListener('scroll', () => {
      nav.classList.toggle('scrolled', window.scrollY > 20);
    }, { passive: true });
  }

  if (toggle && links) {
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
      toggle.innerHTML = links.classList.contains('open')
        ? '<i class="fas fa-times"></i>'
        : '<i class="fas fa-bars"></i>';
    });

    links.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        links.classList.remove('open');
        toggle.innerHTML = '<i class="fas fa-bars"></i>';
      });
    });
  }

  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
}

function initReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  els.forEach(el => observer.observe(el));
}

function initTabs() {
  document.querySelectorAll('.tabs-premium').forEach(tabBar => {
    const tabs = tabBar.querySelectorAll('.tab-premium');
    const container = tabBar.nextElementSibling;
    if (!container) return;

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.target;
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        container.querySelectorAll('.tab-pane-premium').forEach(pane => {
          pane.classList.toggle('active', pane.id === target);
        });
      });
    });
  });
}

function initRiskGauge() {
  const gauge = document.querySelector('.risk-gauge-fill');
  if (!gauge) return;

  const target = parseFloat(gauge.dataset.value) || 0;
  const circumference = 2 * Math.PI * 90;
  gauge.style.strokeDasharray = circumference;
  gauge.style.strokeDashoffset = circumference;

  requestAnimationFrame(() => {
    setTimeout(() => {
      gauge.style.strokeDashoffset = circumference - (target / 100) * circumference;
    }, 300);
  });
}

function animateCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const duration = 1200;
    const start = performance.now();

    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = (target * eased).toFixed(target % 1 ? 1 : 0) + suffix;
      if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
  });
}

if (document.querySelector('[data-count]')) {
  document.addEventListener('DOMContentLoaded', animateCounters);
}
