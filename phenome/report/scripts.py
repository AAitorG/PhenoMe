"""
JavaScript code for the phenotyping HTML report.

This module contains all JavaScript functionality for the standalone HTML report,
including collapsible sections, smooth scrolling, and table sorting.
"""

REPORT_JS = """
// Collapsible sections
document.querySelectorAll('.collapsible').forEach(button => {
    button.addEventListener('click', () => {
        button.classList.toggle('active');
        const content = button.nextElementSibling;
        content.classList.toggle('active');
    });
});

// Smooth scrolling for navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const target = document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Table sorting (simple implementation)
document.querySelectorAll('th[data-sortable]').forEach(th => {
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const colIndex = th.cellIndex;
        const isNumeric = th.dataset.type === 'number';

        const sorted = rows.sort((a, b) => {
            let aVal = a.cells[colIndex].textContent || '';
            let bVal = b.cells[colIndex].textContent || '';
            if (isNumeric) {
                // Strip thousand separators before parsing (e.g. "1,234" -> 1234)
                aVal = parseFloat(String(aVal).replace(/,/g, '')) || 0;
                bVal = parseFloat(String(bVal).replace(/,/g, '')) || 0;
                return aVal - bVal;
            }
            return aVal.localeCompare(bVal);
        });

        if (th.dataset.sortDir === 'asc') {
            sorted.reverse();
            th.dataset.sortDir = 'desc';
        } else {
            th.dataset.sortDir = 'asc';
        }

        sorted.forEach(row => tbody.appendChild(row));
    });
});

// Image modal/lightbox functionality
document.querySelectorAll('.image-card img').forEach(img => {
    img.addEventListener('click', () => {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            cursor: pointer;
        `;

        const largeImg = document.createElement('img');
        largeImg.src = img.src;
        largeImg.style.cssText = `
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            border-radius: 8px;
        `;

        modal.appendChild(largeImg);
        document.body.appendChild(modal);

        modal.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    });
});

// Initialize tooltips on hover
document.querySelectorAll('[data-tooltip]').forEach(el => {
    el.addEventListener('mouseenter', (e) => {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = el.dataset.tooltip;
        tooltip.style.cssText = `
            position: absolute;
            background: var(--bg-card);
            color: var(--text-primary);
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            z-index: 1000;
            pointer-events: none;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        `;
        document.body.appendChild(tooltip);

        const rect = el.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';

        el._tooltip = tooltip;
    });

    el.addEventListener('mouseleave', () => {
        if (el._tooltip) {
            document.body.removeChild(el._tooltip);
            el._tooltip = null;
        }
    });
});
"""
