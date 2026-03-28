"""
CSS styles for the phenotyping HTML report.

This module contains all CSS styling for the standalone HTML report,
including dark theme, responsive design, and animations.
"""

REPORT_CSS = """
:root {
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
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    --shadow-sm: 0 4px 15px rgba(0, 0, 0, 0.2);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg-dark);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'SF Pro Display', system-ui, sans-serif;
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
}

h2::before {
    content: '';
    width: 4px;
    height: 1.5rem;
    background: var(--primary);
    border-radius: 2px;
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

/* Layout */
.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    padding: 3rem 0;
    margin-bottom: 2rem;
    position: relative;
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

/* Navigation */
.nav-container {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 2rem;
    position: sticky;
    top: 1rem;
    z-index: 100;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}

.nav-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.nav-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    transition: all 0.2s ease;
    background: transparent;
    border: 1px solid transparent;
}

.nav-link:hover {
    background: var(--bg-card-hover);
    color: var(--primary);
    border-color: var(--primary);
}

/* Stats Cards */
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

/* Section */
.section {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
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

/* Subsection Grid */
.subsection-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}

.subsection {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--border);
}

/* Tables */
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

/* Code/Value highlight */
.value {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(99, 102, 241, 0.1);
    padding: 0.125rem 0.5rem;
    border-radius: 4px;
    color: var(--primary);
    font-size: 0.85em;
}

/* Plot containers */
.plot-container {
    background: rgba(0, 0, 0, 0.2);
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

/* Info box */
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

/* Collapsible */
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
    background: rgba(0, 0, 0, 0.1);
    border-radius: 0 0 8px 8px;
}

.collapsible-content.active {
    max-height: 2000px;
    padding: 1rem;
}

/* Progress bar for percentages */
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

/* Feature list */
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

/* Image gallery */
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
    width: 100%;
    height: 200px;
    object-fit: contain;
    background: rgba(0, 0, 0, 0.3);
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

/* Cluster colors */
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

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* Responsive */
@media (max-width: 768px) {
    .container { padding: 1rem; }
    h1 { font-size: 1.75rem; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .plot-grid { grid-template-columns: 1fr; }
    .nav-links { flex-direction: column; }
    .image-gallery { grid-template-columns: repeat(2, 1fr); }
}

/* Print styles */
@media print {
    body { background: white; color: black; }
    .nav-container { display: none; }
    .section { break-inside: avoid; }
    .plot-container { page-break-inside: avoid; }
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.section {
    animation: fadeIn 0.5s ease forwards;
}

/* Sections are 4th+ children of container (after header, nav, stats) */
.section:nth-child(4) { animation-delay: 0.1s; }
.section:nth-child(5) { animation-delay: 0.2s; }
.section:nth-child(6) { animation-delay: 0.3s; }
.section:nth-child(7) { animation-delay: 0.4s; }
.section:nth-child(8) { animation-delay: 0.5s; }
.section:nth-child(9) { animation-delay: 0.6s; }
.section:nth-child(10) { animation-delay: 0.7s; }
.section:nth-child(11) { animation-delay: 0.8s; }
"""
