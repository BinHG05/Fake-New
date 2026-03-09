/**
 * Tech Stack Reference — Interactive Script
 * Handles: navigation highlighting, scroll animations, back-to-top, code highlighting
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize highlight.js
    hljs.highlightAll();

    // === Navigation Active State ===
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.category-section');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Update active nav on scroll
    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
    };

    const navObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const cat = entry.target.dataset.category;
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.dataset.category === cat);
                });
            }
        });
    }, observerOptions);

    sections.forEach(section => navObserver.observe(section));

    // === Scroll Reveal for Cards ===
    const cards = document.querySelectorAll('.tech-card');

    const cardObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Stagger animation
                setTimeout(() => {
                    entry.target.classList.add('revealed');
                }, index * 80);
                cardObserver.unobserve(entry.target);
            }
        });
    }, {
        root: null,
        rootMargin: '0px 0px -50px 0px',
        threshold: 0.05
    });

    cards.forEach(card => cardObserver.observe(card));

    // === Back to Top Button ===
    const backToTopBtn = document.getElementById('backToTop');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 500) {
            backToTopBtn.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
        }
    }, { passive: true });

    backToTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // === Count Technologies ===
    const techCount = document.querySelectorAll('.tech-card').length;
    const countEl = document.getElementById('tech-count');
    if (countEl) {
        countEl.textContent = techCount;
    }

    // === Click to expand/collapse code examples ===
    const exampleHeaders = document.querySelectorAll('.example-header');
    exampleHeaders.forEach(header => {
        header.style.cursor = 'pointer';
        // Expand is default
        header.addEventListener('click', () => {
            const pre = header.parentElement.querySelector('pre');
            if (pre) {
                if (pre.style.display === 'none') {
                    pre.style.display = 'block';
                    header.querySelector('.example-badge').textContent = '📂 Ví dụ từ project';
                } else {
                    pre.style.display = 'none';
                    header.querySelector('.example-badge').textContent = '📂 Ví dụ từ project (click để mở)';
                }
            }
        });
    });

    // === Smooth parallax for background glows ===
    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                const scrollY = window.scrollY;
                const glows = document.querySelectorAll('.bg-glow');
                glows.forEach((glow, i) => {
                    const speed = 0.02 + (i * 0.01);
                    glow.style.transform = `translateY(${scrollY * speed}px)`;
                });
                ticking = false;
            });
            ticking = true;
        }
    }, { passive: true });
});
