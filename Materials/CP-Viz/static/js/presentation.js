(function() {
  'use strict';

  var branchOrigin = -1; // Mainline slide index before entering a branch.

  function byId(id) {
    return document.getElementById(id);
  }

  function first(selector, root) {
    return (root || document).querySelector(selector);
  }

  function setBackButtonVisible(visible) {
    var backButton = first('.btn-back');
    if (backButton) backButton.style.display = visible ? 'inline-flex' : 'none';
  }

  function currentSlideIndex() {
    return window.ws && typeof window.ws.currentSlideI_ === 'number'
      ? window.ws.currentSlideI_
      : -1;
  }

  async function renderMermaid() {
    if (!window.mermaid || typeof window.mermaid.run !== 'function') return;

    try {
      await window.mermaid.run({ querySelector: '.mermaid' });
    } catch (error) {
      console.error('mermaid error:', error);
    }
  }

  function initWebSlides() {
    if (window.ws || typeof window.WebSlides !== 'function') return;
    window.ws = new window.WebSlides();
  }

  function openLightbox(src) {
    var lightbox = byId('lightbox');
    var image = byId('lightbox-img');
    if (!lightbox || !image || !src) return;

    image.src = src;
    lightbox.classList.add('active');
    lightbox.setAttribute('aria-hidden', 'false');
  }

  function closeLightbox() {
    var lightbox = byId('lightbox');
    if (!lightbox) return;

    lightbox.classList.remove('active');
    lightbox.setAttribute('aria-hidden', 'true');
  }

  function jumpToBranch(branchId) {
    var webslides = byId('webslides');
    var container = byId('branch-container');
    var branch = byId(branchId);
    if (!webslides || !container || !branch) return;

    var activeBranch = first('.branch-slide.active', container);
    if (activeBranch && activeBranch !== branch) activeBranch.classList.remove('active');

    branchOrigin = currentSlideIndex();
    webslides.style.display = 'none';
    container.classList.add('active');
    branch.classList.add('active');
    setBackButtonVisible(true);
  }

  function backFromBranch() {
    var webslides = byId('webslides');
    var container = byId('branch-container');
    if (!webslides || !container) return;

    webslides.style.display = '';
    container.classList.remove('active');

    var activeBranch = first('.branch-slide.active', container);
    if (activeBranch) activeBranch.classList.remove('active');

    setBackButtonVisible(false);
    if (branchOrigin >= 0 && window.ws && typeof window.ws.goToSlide === 'function') {
      window.ws.goToSlide(branchOrigin);
    }
    branchOrigin = -1;
  }

  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
      closeLightbox();
    }
  });

  // Keep existing inline handlers working while containing implementation details.
  window.openLightbox = openLightbox;
  window.closeLightbox = closeLightbox;
  window.jumpToBranch = jumpToBranch;
  window.backFromBranch = backFromBranch;

  renderMermaid().then(initWebSlides);
})();
