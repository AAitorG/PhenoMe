"""
JavaScript code for the phenotyping HTML report.

This module contains all JavaScript functionality for the standalone HTML
report:

- Collapsible sections
- Smooth scrolling and scrollspy
- Column-sortable tables
- Keyboard-aware image lightbox (ESC to close, arrow keys to navigate)
- Floating back-to-top button
- Mobile-friendly navigation drawer (hamburger)
- Section anchor link copy to clipboard
"""

REPORT_JS = """
(function () {
    'use strict';

    // ---------- Utilities ----------
    const $  = (sel, root) => (root || document).querySelector(sel);
    const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

    function showToast(message, timeout) {
        const toast = $('#toast');
        if (!toast) return;
        toast.textContent = message;
        toast.classList.add('visible');
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => {
            toast.classList.remove('visible');
        }, timeout || 1800);
    }

    // ---------- Collapsible sections ----------
    $$('.collapsible').forEach(button => {
        button.addEventListener('click', () => {
            button.classList.toggle('active');
            const content = button.nextElementSibling;
            if (content) content.classList.toggle('active');
        });
    });

    // ---------- Smooth scroll for nav links ----------
    $$('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href') || '';
            if (!href.startsWith('#')) return;
            const target = document.getElementById(href.substring(1));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                closeMobileNav();
                if (history.replaceState) {
                    history.replaceState(null, '', href);
                }
            }
        });
    });

    // ---------- Sortable tables ----------
    $$('th[data-sortable]').forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
            const table = th.closest('table');
            const tbody = table && table.querySelector('tbody');
            if (!tbody) return;
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const colIndex = th.cellIndex;
            const isNumeric = th.dataset.type === 'number';

            const sorted = rows.sort((a, b) => {
                let aVal = (a.cells[colIndex] && a.cells[colIndex].textContent) || '';
                let bVal = (b.cells[colIndex] && b.cells[colIndex].textContent) || '';
                if (isNumeric) {
                    aVal = parseFloat(String(aVal).replace(/,/g, '')) || 0;
                    bVal = parseFloat(String(bVal).replace(/,/g, '')) || 0;
                    return aVal - bVal;
                }
                return aVal.localeCompare(bVal);
            });

            const dir = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
            if (dir === 'desc') sorted.reverse();
            th.dataset.sortDir = dir;

            // Clear sibling direction indicators so only the active column shows a caret
            const headerRow = th.parentElement;
            if (headerRow) {
                Array.from(headerRow.children).forEach(sib => {
                    if (sib !== th) delete sib.dataset.sortDir;
                });
            }

            const frag = document.createDocumentFragment();
            sorted.forEach(row => frag.appendChild(row));
            tbody.appendChild(frag);
        });
    });

    // ---------- Image lightbox ----------
    const Lightbox = (function () {
        let overlay = null;
        let imgEl = null;
        let captionEl = null;
        let counterEl = null;
        let currentList = [];
        let currentIndex = 0;
        let zoom = 1;

        function build() {
            overlay = document.createElement('div');
            overlay.className = 'lightbox';
            overlay.setAttribute('role', 'dialog');
            overlay.setAttribute('aria-modal', 'true');

            overlay.innerHTML = `
                <div class="lightbox__counter" aria-hidden="true"></div>
                <button class="lightbox__close" aria-label="Close (Esc)">&times;</button>
                <button class="lightbox__prev" aria-label="Previous image (Left arrow)">&#10094;</button>
                <img class="lightbox__image" alt="">
                <button class="lightbox__next" aria-label="Next image (Right arrow)">&#10095;</button>
                <div class="lightbox__caption"></div>
            `;
            imgEl     = overlay.querySelector('.lightbox__image');
            captionEl = overlay.querySelector('.lightbox__caption');
            counterEl = overlay.querySelector('.lightbox__counter');

            overlay.querySelector('.lightbox__close').addEventListener('click', close);
            overlay.querySelector('.lightbox__prev').addEventListener('click', (e) => {
                e.stopPropagation();
                show(currentIndex - 1);
            });
            overlay.querySelector('.lightbox__next').addEventListener('click', (e) => {
                e.stopPropagation();
                show(currentIndex + 1);
            });
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) close();
            });
            imgEl.addEventListener('click', (e) => e.stopPropagation());
            imgEl.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                zoom = Math.min(4, Math.max(0.5, zoom + delta));
                imgEl.style.transform = 'scale(' + zoom + ')';
            }, { passive: false });
        }

        function show(index) {
            if (!currentList.length) return;
            currentIndex = (index + currentList.length) % currentList.length;
            const src = currentList[currentIndex];
            zoom = 1;
            imgEl.style.transform = 'scale(1)';
            imgEl.src = src.src;
            imgEl.alt = src.caption || '';
            captionEl.textContent = src.meta ? src.caption + ' \\u2014 ' + src.meta : (src.caption || '');
            counterEl.textContent = (currentIndex + 1) + ' / ' + currentList.length;
            const single = currentList.length <= 1;
            overlay.querySelector('.lightbox__prev').style.display = single ? 'none' : '';
            overlay.querySelector('.lightbox__next').style.display = single ? 'none' : '';
        }

        function open(list, index) {
            if (!overlay) build();
            currentList = list;
            document.body.appendChild(overlay);
            document.addEventListener('keydown', onKey);
            show(index);
        }

        function close() {
            if (overlay && overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
            document.removeEventListener('keydown', onKey);
            currentList = [];
        }

        function onKey(e) {
            if (e.key === 'Escape') close();
            else if (e.key === 'ArrowRight') show(currentIndex + 1);
            else if (e.key === 'ArrowLeft')  show(currentIndex - 1);
            else if (e.key === '+' || e.key === '=') {
                zoom = Math.min(4, zoom + 0.25);
                imgEl.style.transform = 'scale(' + zoom + ')';
            } else if (e.key === '-') {
                zoom = Math.max(0.5, zoom - 0.25);
                imgEl.style.transform = 'scale(' + zoom + ')';
            } else if (e.key === '0') {
                zoom = 1;
                imgEl.style.transform = 'scale(1)';
            }
        }

        return { open: open };
    })();

    $$('[data-gallery]').forEach(gallery => {
        const imgs = $$('img', gallery);
        const descriptors = imgs.map(img => ({
            src: img.src,
            caption: img.dataset.caption || img.alt || '',
            meta: img.dataset.meta || ''
        }));
        imgs.forEach((img, index) => {
            img.addEventListener('click', () => Lightbox.open(descriptors, index));
        });
    });

    // Fallback: any stray .image-card img not inside [data-gallery]
    $$('.image-card img').forEach(img => {
        if (img.closest('[data-gallery]')) return;
        img.addEventListener('click', () => {
            Lightbox.open([{ src: img.src, caption: img.alt || '' }], 0);
        });
    });

    // ---------- Tooltips (kept from previous behaviour) ----------
    $$('[data-tooltip]').forEach(el => {
        el.addEventListener('mouseenter', () => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = el.dataset.tooltip;
            tooltip.style.cssText = [
                'position:absolute',
                'background:var(--bg-card)',
                'color:var(--text-primary)',
                'padding:0.5rem 0.75rem',
                'border-radius:4px',
                'font-size:0.8rem',
                'z-index:1000',
                'pointer-events:none',
                'border:1px solid var(--border)',
                'box-shadow:var(--shadow-sm)'
            ].join(';');
            document.body.appendChild(tooltip);
            const rect = el.getBoundingClientRect();
            tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)) + 'px';
            tooltip.style.top  = (rect.top - tooltip.offsetHeight - 8) + 'px';
            el._tooltip = tooltip;
        });
        el.addEventListener('mouseleave', () => {
            if (el._tooltip && el._tooltip.parentNode) {
                el._tooltip.parentNode.removeChild(el._tooltip);
                el._tooltip = null;
            }
        });
    });

    // ---------- Scrollspy ----------
    const sectionsEls = $$('.section');
    const navLinks = $$('.nav-link');
    if (sectionsEls.length && navLinks.length && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                const targetId = entry.target.id;
                navLinks.forEach(link => {
                    link.classList.toggle(
                        'active',
                        link.getAttribute('href') === '#' + targetId
                    );
                });
            });
        }, { root: null, rootMargin: '0px 0px -60% 0px', threshold: 0 });
        sectionsEls.forEach(section => observer.observe(section));
    }

    // ---------- Back-to-top ----------
    const backToTop = $('#backToTop');
    if (backToTop) {
        const onScroll = () => {
            backToTop.classList.toggle('visible', window.scrollY > 400);
        };
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // ---------- Mobile nav drawer ----------
    const navToggle = $('#navToggle');
    const navContainer = $('#reportNav');

    function closeMobileNav() {
        if (!navContainer || !navToggle) return;
        navContainer.classList.remove('open');
        navToggle.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
        document.body.classList.remove('nav-open');
    }

    if (navToggle && navContainer) {
        navToggle.addEventListener('click', () => {
            const isOpen = navContainer.classList.toggle('open');
            navToggle.classList.toggle('open', isOpen);
            navToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            document.body.classList.toggle('nav-open', isOpen);
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeMobileNav();
        });
    }

    // ---------- Section anchor copy ----------
    $$('.section-anchor').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            const href = anchor.getAttribute('href') || '';
            const url = location.origin + location.pathname + href;
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            if (history.replaceState) {
                history.replaceState(null, '', href);
            }
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(url).then(
                    () => showToast('Link copied'),
                    () => showToast('Link: ' + href)
                );
            } else {
                showToast('Link: ' + href);
            }
        });
    });
})();
"""
