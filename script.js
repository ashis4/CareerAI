/* ══════════════════════════════════════════════
   CareerAI — script.js
══════════════════════════════════════════════ */

// Store API response globally for roadmap lookup
let globalTop3 = [];
let globalRoadmaps = {};

function showStep1() {
  document.getElementById('step-select').style.display = 'block';
  document.getElementById('step-roadmap').style.display = 'none';
}

function showRoadmap(role) {
  const roadmap = globalRoadmaps[role] || 'Roadmap coming soon!';
  document.getElementById('result-career').textContent = role;
  document.getElementById('result-roadmap').textContent = roadmap;
  document.getElementById('step-select').style.display = 'none';
  document.getElementById('step-roadmap').style.display = 'block';

  // Scroll modal back to top
  const modal = document.getElementById('result-modal');
  modal.scrollTop = 0;
}

document.addEventListener('DOMContentLoaded', () => {

  /* ── 1. SKILL TAG CLICK ── */
  const skillTags = document.querySelectorAll('.skill-tag');
  skillTags.forEach(tag => {
    tag.addEventListener('click', () => {
      skillTags.forEach(t => t.classList.remove('active'));
      tag.classList.add('active');
    });
  });

  /* ── 2. SCROLL-REVEAL ── */
  const revealTargets = document.querySelectorAll('.feature-card, .step, .outcome-card');
  revealTargets.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  });
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  revealTargets.forEach(el => revealObserver.observe(el));

  /* ── 3. NAVBAR SCROLL SHADOW ── */
  const nav = document.querySelector('nav');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
      nav.style.boxShadow = '0 4px 32px rgba(0,0,0,0.4)';
      nav.style.background = 'rgba(8,11,20,0.97)';
    } else {
      nav.style.boxShadow = 'none';
      nav.style.background = 'rgba(8,11,20,0.85)';
    }
  }, { passive: true });

  /* ── 4. PREDICT BUTTON ── */
  const predictBtn  = document.querySelector('.predict-btn');
  const ctaBtn      = document.querySelector('.cta-btn');
  const searchInput = document.querySelector('.search-wrap input');
  const modal       = document.getElementById('result-modal');
  const resultTop3  = document.getElementById('result-top3');

  function shakeInput() {
    if (!searchInput) return;
    searchInput.style.transition = 'transform 0.1s ease';
    searchInput.style.transform = 'translateX(6px)';
    setTimeout(() => { searchInput.style.transform = 'translateX(-6px)'; }, 100);
    setTimeout(() => { searchInput.style.transform = 'translateX(4px)';  }, 200);
    setTimeout(() => { searchInput.style.transform = 'translateX(0)';    }, 300);
  }

  async function handlePredict() {
    const skills = searchInput ? searchInput.value.trim() : '';
    if (!skills) { shakeInput(); return; }

    if (predictBtn) {
      predictBtn.textContent = '⏳ Predicting...';
      predictBtn.disabled = true;
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skills })
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();

      // Save globally
      globalTop3 = data.top3 || [];
      globalRoadmaps = data.roadmaps || {};

      // Build top 3 cards — clickable to show roadmap
      if (resultTop3 && globalTop3.length) {
        const colors = ['rgba(108,99,255,0.25)', 'rgba(108,99,255,0.15)', 'rgba(108,99,255,0.08)'];
        resultTop3.innerHTML = globalTop3.map((r, i) =>
          `<div onclick="showRoadmap('${r.role.replace(/'/g, "\\'")}')"
            style="display:flex;justify-content:space-between;align-items:center;
                   padding:16px 20px;background:${colors[i]};border-radius:12px;
                   margin-bottom:10px;cursor:pointer;border:1px solid rgba(108,99,255,${0.4 - i*0.1});
                   transition:transform 0.15s ease, background 0.15s ease;"
            onmouseover="this.style.transform='translateX(4px)';this.style.background='rgba(108,99,255,${0.35 - i*0.05})'"
            onmouseout="this.style.transform='translateX(0)';this.style.background='${colors[i]}'">
            <div style="display:flex;align-items:center;gap:12px;">
              <span style="background:rgba(108,99,255,0.3);color:#a78bfa;width:28px;height:28px;
                           border-radius:50%;display:flex;align-items:center;justify-content:center;
                           font-size:13px;font-weight:700;flex-shrink:0;">${i + 1}</span>
              <span style="color:#e8eaf6;font-size:15px;font-weight:500;">${r.role}</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="color:#a78bfa;font-size:14px;font-weight:700;">${r.score}%</span>
              <span style="color:#6c63ff;font-size:18px;">→</span>
            </div>
          </div>`
        ).join('');
      }

      // Show modal at step 1
      showStep1();
      modal.style.display = 'flex';

    } catch (err) {
      console.error('Prediction error:', err);
      alert('⚠️ Could not connect to the server. Make sure Flask is running.');
    } finally {
      if (predictBtn) {
        predictBtn.textContent = '✦ Predict Career';
        predictBtn.disabled = false;
      }
    }
  }

  if (predictBtn) predictBtn.addEventListener('click', handlePredict);
  if (ctaBtn) ctaBtn.addEventListener('click', () => {
    document.querySelector('.hero').scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => { if (searchInput) searchInput.focus(); }, 600);
  });
  if (searchInput) {
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handlePredict();
    });
  }

  // Close on backdrop click
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.style.display = 'none';
    });
  }

  /* ── 5. SMOOTH NAV LINKS ── */
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
    });
  });

});
