"""
CSS styles for the phenotyping HTML report.

This module contains all CSS styling for the standalone HTML report.
Theming is driven by ``[data-theme="dark"]`` (default) and
``[data-theme="light"]`` attributes on the ``<html>`` element (set at report
generation time). Includes responsive design, mobile-friendly navigation, a
floating back-to-top button, a keyboard-aware image lightbox, and print styles.
"""

REPORT_CSS = """
/* ---------- Theme tokens ---------- */
:root,
html[data-theme="dark"] {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #10b981;
    --accent: #f59e0b;
    --danger: #ef4444;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #334155;
    --overlay-strong: rgba(0, 0, 0, 0.9);
    --overlay-soft: rgba(0, 0, 0, 0.2);
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    --shadow-sm: 0 4px 15px rgba(0, 0, 0, 0.2);
    --focus-ring: 0 0 0 3px rgba(99, 102, 241, 0.35);
}

html[data-theme="light"] {
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --secondary: #059669;
    --accent: #d97706;
    --danger: #dc2626;
    --bg-dark: #f8fafc;
    --bg-card: #ffffff;
    --bg-card-hover: #eef2ff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --overlay-strong: rgba(15, 23, 42, 0.85);
    --overlay-soft: rgba(15, 23, 42, 0.04);
    --gradient-1: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    --gradient-2: linear-gradient(135deg, #ec4899 0%, #f97316 100%);
    --gradient-3: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
    --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    --shadow-sm: 0 2px 10px rgba(15, 23, 42, 0.06);
    --focus-ring: 0 0 0 3px rgba(79, 70, 229, 0.25);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    /* Use web fonts when available, but fall back to robust system stacks so
     * the report remains legible offline (i.e. without Google Fonts). */
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Consolas',
        ui-monospace, Menlo, monospace;
    background: var(--bg-dark);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
    transition: background-color 0.25s ease, color 0.25s ease;
}

:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
    border-radius: 4px;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'SF Pro Display', system-ui,
        -apple-system, 'Segoe UI', Roboto, sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.5rem;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

h2 {
    font-size: 1.75rem;
    color: var(--text-primary);
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    scroll-margin-top: 1rem;
}

h2::before {
    content: '';
    width: 4px;
    height: 1.5rem;
    background: var(--primary);
    border-radius: 2px;
    flex-shrink: 0;
}

.section-title-text {
    flex: 1;
    min-width: 0;
}

.section-anchor {
    color: var(--text-muted);
    text-decoration: none;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 400;
    opacity: 0;
    transition: opacity 0.15s ease, color 0.15s ease;
    padding: 0 0.25rem;
}

h2:hover .section-anchor,
.section-anchor:focus-visible {
    opacity: 1;
}

.section-anchor:hover {
    color: var(--primary);
}

h3 {
    font-size: 1.25rem;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem;
}

h4 {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 1rem 0 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

p {
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

a {
    color: var(--primary);
}

/* ---------- Layout ---------- */
.container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 2rem;
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 2.5rem;
    align-items: start;
}

@media (max-width: 1024px) {
    .container {
        grid-template-columns: 1fr;
        padding: 1rem;
    }
}

.header {
    text-align: center;
    padding: 3rem 0;
    margin-bottom: 2rem;
    position: relative;
    grid-column: 1 / -1;
}

.header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 3px;
    background: var(--gradient-1);
    border-radius: 2px;
}

.subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* ---------- Mobile nav toggle ---------- */
.nav-toggle {
    position: fixed;
    top: 1rem;
    left: 1rem;
    z-index: 1100;
    display: none;
    flex-direction: column;
    gap: 5px;
    width: 40px;
    height: 40px;
    padding: 10px 8px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
}

.nav-toggle span {
    display: block;
    height: 2px;
    width: 100%;
    background: var(--text-primary);
    border-radius: 2px;
    transition: transform 0.2s ease, opacity 0.2s ease;
}

.nav-toggle.open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
.nav-toggle.open span:nth-child(2) { opacity: 0; }
.nav-toggle.open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

@media (max-width: 1024px) {
    .nav-toggle { display: flex; }
}

/* ---------- Navigation ---------- */
.nav-container {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    position: sticky;
    top: 2rem;
    z-index: 100;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    grid-column: 1;
    max-height: calc(100vh - 4rem);
    overflow-y: auto;
}

.nav-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.nav-links {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    transition: all 0.2s ease;
    background: transparent;
    border: 1px solid transparent;
    display: block;
}

.nav-link:hover, .nav-link.active {
    background: var(--bg-card-hover);
    color: var(--primary);
    border-color: var(--border);
}

/* Mobile drawer behaviour */
@media (max-width: 1024px) {
    .nav-container {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        max-height: 100vh;
        width: min(320px, 85vw);
        border-radius: 0;
        margin: 0;
        transform: translateX(-105%);
        transition: transform 0.25s ease;
        box-shadow: var(--shadow);
        z-index: 1050;
    }
    .nav-container.open {
        transform: translateX(0);
    }
    body.nav-open {
        overflow: hidden;
    }
}

/* ---------- Main content ---------- */
.content-wrapper {
    grid-column: 2;
    min-width: 0; /* Prevents overflow in CSS grid */
}

@media (max-width: 1024px) {
    .content-wrapper { grid-column: 1; }
}

/* ---------- Stats cards ---------- */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}

.stat-card.primary::before { background: var(--primary); }
.stat-card.secondary::before { background: var(--secondary); }
.stat-card.accent::before { background: var(--accent); }
.stat-card.danger::before { background: var(--danger); }

.stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.stat-card.primary .stat-value { color: var(--primary); }
.stat-card.secondary .stat-value { color: var(--secondary); }
.stat-card.accent .stat-value { color: var(--accent); }
.stat-card.danger .stat-value { color: var(--danger); }

.stat-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- Section ---------- */
.section {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: visible; /* Allows Plotly tooltips to display properly */
    scroll-margin-top: 1rem;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
}

.section-badge {
    background: var(--primary);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ---------- Subsection grid ---------- */
.subsection-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}

.subsection {
    background: var(--overlay-soft);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border);
}

/* ---------- Tables ---------- */
.table-container {
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin: 1rem 0;
}

.table-scroll {
    max-height: 420px;
    overflow-y: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

thead {
    background: linear-gradient(to bottom, var(--bg-card-hover), var(--bg-card));
    position: sticky;
    top: 0;
}

th {
    text-align: left;
    padding: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 2px solid var(--primary);
    white-space: nowrap;
}

th[data-sortable] {
    user-select: none;
}

th[data-sortable]::after {
    content: ' \\2195';
    color: var(--text-muted);
    font-size: 0.75rem;
}

th[data-sort-dir="asc"]::after { content: ' \\2191'; color: var(--primary); }
th[data-sort-dir="desc"]::after { content: ' \\2193'; color: var(--primary); }

td {
    padding: 0.875rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
}

tr:hover td {
    background: rgba(99, 102, 241, 0.05);
}

tr:last-child td {
    border-bottom: none;
}

/* Code/value highlight */
.value {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    background: rgba(99, 102, 241, 0.12);
    padding: 0.125rem 0.5rem;
    border-radius: 4px;
    color: var(--primary);
    font-size: 0.85em;
}

/* ---------- Plot containers ---------- */
.plot-container {
    background: var(--overlay-soft);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid var(--border);
}

.plot-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.plot-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5rem;
}

/* ---------- Info boxes ---------- */
.info-box {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid var(--primary);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    color: var(--text-secondary);
}

.info-box.warning {
    background: rgba(245, 158, 11, 0.1);
    border-color: var(--accent);
}

.info-box.error {
    background: rgba(239, 68, 68, 0.1);
    border-color: var(--danger);
}

.info-box.success {
    background: rgba(16, 185, 129, 0.1);
    border-color: var(--secondary);
}

/* ---------- Collapsible ---------- */
.collapsible {
    background: var(--bg-card-hover);
    border: none;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    width: 100%;
    text-align: left;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--text-primary);
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.5rem;
    transition: background 0.2s;
}

.collapsible:hover {
    background: var(--border);
}

.collapsible::after {
    content: '\\25BC';
    font-size: 0.75rem;
    transition: transform 0.3s;
}

.collapsible.active::after {
    transform: rotate(180deg);
}

.collapsible-content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease-out;
    background: var(--overlay-soft);
    border-radius: 0 0 8px 8px;
}

.collapsible-content.active {
    max-height: 2000px;
    padding: 1rem;
}

/* ---------- Progress bar ---------- */
.progress-bar {
    width: 100%;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.5rem;
}

.progress-fill {
    height: 100%;
    background: var(--gradient-1);
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* ---------- Feature tags ---------- */
.feature-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
}

.feature-tag {
    background: rgba(99, 102, 241, 0.15);
    color: var(--primary);
    padding: 0.375rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

/* ---------- Image gallery ---------- */
.image-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.image-card {
    background: var(--bg-card-hover);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    transition: transform 0.2s, box-shadow 0.2s;
}

.image-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}

.image-card img {
    display: block;
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: contain;
    background: var(--overlay-soft);
    cursor: zoom-in;
}

.image-card-info {
    padding: 0.75rem;
}

.image-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.image-card-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* ---------- Cluster colors ---------- */
.cluster-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

.cluster-0 { background: rgba(99, 102, 241, 0.2); color: #818cf8; }
.cluster-1 { background: rgba(16, 185, 129, 0.2); color: #34d399; }
.cluster-2 { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
.cluster-3 { background: rgba(239, 68, 68, 0.2); color: #f87171; }
.cluster-4 { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
.cluster-5 { background: rgba(6, 182, 212, 0.2); color: #22d3ee; }
.cluster-6 { background: rgba(251, 146, 60, 0.2); color: #fb923c; }
.cluster-7 { background: rgba(244, 114, 182, 0.2); color: #f472b6; }

/* ---------- Image lightbox ---------- */
.lightbox {
    position: fixed;
    inset: 0;
    background: var(--overlay-strong);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2000;
    animation: fadeInOverlay 0.15s ease;
}

@keyframes fadeInOverlay {
    from { opacity: 0; }
    to { opacity: 1; }
}

.lightbox__image {
    max-width: 92vw;
    max-height: 86vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    transition: transform 0.15s ease;
    user-select: none;
}

.lightbox__close,
.lightbox__prev,
.lightbox__next {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(15, 23, 42, 0.7);
    color: #f1f5f9;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    width: 48px;
    height: 48px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    cursor: pointer;
    transition: background 0.15s ease, transform 0.15s ease;
}

.lightbox__close:hover,
.lightbox__prev:hover,
.lightbox__next:hover {
    background: rgba(99, 102, 241, 0.85);
}

.lightbox__close {
    top: 1.25rem;
    right: 1.25rem;
    transform: none;
}

.lightbox__prev { left: 1.25rem; }
.lightbox__next { right: 1.25rem; }

.lightbox__caption {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 1.25rem;
    text-align: center;
    color: #f1f5f9;
    font-size: 0.9rem;
    padding: 0 4rem;
    pointer-events: none;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
}

.lightbox__counter {
    position: absolute;
    top: 1.25rem;
    left: 1.25rem;
    color: #f1f5f9;
    background: rgba(15, 23, 42, 0.6);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
}

/* ---------- Back-to-top ---------- */
.back-to-top {
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    width: 44px;
    height: 44px;
    border-radius: 999px;
    background: var(--primary);
    color: #fff;
    border: none;
    box-shadow: var(--shadow);
    cursor: pointer;
    opacity: 0;
    visibility: hidden;
    transform: translateY(6px);
    transition: opacity 0.2s ease, transform 0.2s ease, visibility 0.2s;
    z-index: 900;
    font-size: 1.25rem;
    line-height: 44px;
}

.back-to-top.visible {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

.back-to-top:hover {
    background: var(--primary-dark);
}

/* ---------- Toast ---------- */
.toast {
    position: fixed;
    bottom: 1.5rem;
    left: 50%;
    transform: translate(-50%, 20px);
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 0.6rem 1.1rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    font-size: 0.85rem;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease, transform 0.2s ease;
    z-index: 2100;
}

.toast.visible {
    opacity: 1;
    transform: translate(-50%, 0);
}

/* ---------- Footer ---------- */
.footer {
    grid-column: 1 / -1;
    text-align: center;
    padding: 3rem 0;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    h1 { font-size: 1.75rem; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .plot-grid { grid-template-columns: 1fr; }
    .image-gallery { grid-template-columns: repeat(2, 1fr); }
    .section { padding: 1.5rem; }
    .back-to-top { right: 1rem; bottom: 1rem; }
    .lightbox__caption { padding: 0 1rem; }
}

/* ---------- Print ---------- */
@media print {
    :root, html[data-theme="dark"], html[data-theme="light"] {
        --bg-dark: #ffffff;
        --bg-card: #ffffff;
        --bg-card-hover: #f8fafc;
        --text-primary: #0f172a;
        --text-secondary: #1e293b;
        --text-muted: #475569;
        --border: #e2e8f0;
        --overlay-soft: rgba(15, 23, 42, 0.04);
    }
    body { background: #fff; color: #0f172a; }
    .nav-container,
    .nav-toggle,
    .back-to-top,
    .toast,
    .lightbox { display: none !important; }
    .container {
        grid-template-columns: 1fr;
        padding: 0;
        max-width: 100%;
    }
    .section {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #cbd5e1;
    }
    .plot-container { page-break-inside: avoid; }
    a { color: inherit; text-decoration: underline; }
}

/* ---------- Animations ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.section, .stats-grid {
    animation: fadeIn 0.5s ease forwards;
    opacity: 0;
}

.stats-grid { animation-delay: 0.05s; }
.section:nth-child(2) { animation-delay: 0.1s; }
.section:nth-child(3) { animation-delay: 0.2s; }
.section:nth-child(4) { animation-delay: 0.3s; }
.section:nth-child(5) { animation-delay: 0.4s; }
.section:nth-child(6) { animation-delay: 0.5s; }
.section:nth-child(7) { animation-delay: 0.6s; }
.section:nth-child(8) { animation-delay: 0.7s; }
.section:nth-child(9) { animation-delay: 0.8s; }
.section:nth-child(10) { animation-delay: 0.9s; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }
}
"""
