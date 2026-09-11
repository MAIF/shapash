function initReportInteractions() {
    let scrollFrame = null;
    const boundScrollRoots = new WeakSet();

    function collectRoots() {
        const roots = [document];
        const pending = [document];
        const seen = new WeakSet();
        seen.add(document);

        while (pending.length > 0) {
            const currentRoot = pending.pop();
            currentRoot.querySelectorAll('*').forEach(element => {
                if (element.shadowRoot && !seen.has(element.shadowRoot)) {
                    seen.add(element.shadowRoot);
                    roots.push(element.shadowRoot);
                    pending.push(element.shadowRoot);
                }
            });
        }

        return roots;
    }

    function queryAllRoots(selector) {
        return collectRoots().flatMap(root => Array.from(root.querySelectorAll(selector)));
    }

    function queryByIdAcrossRoots(id) {
        for (const root of collectRoots()) {
            if (typeof root.getElementById === 'function') {
                const match = root.getElementById(id);
                if (match) {
                    return match;
                }
            }
        }
        return null;
    }

    function clearActive(navItems, navChildren, navGroupTitles) {
        navItems.forEach(element => element.classList.remove('active'));
        navChildren.forEach(element => element.classList.remove('active'));
        navGroupTitles.forEach(element => element.classList.remove('active'));
    }

    function bindScrollListeners() {
        collectRoots().forEach(root => {
            if (!boundScrollRoots.has(root)) {
                root.addEventListener('scroll', queueScrollUpdate, true);
                boundScrollRoots.add(root);
            }
        });
    }

    function bindPanelSelectors() {
        queryAllRoots('.js-panel-select[data-panel-group]').forEach(select => {
            if (select.dataset.reportBound === 'true') {
                return;
            }

            function updatePanels() {
                const panelGroup = select.getAttribute('data-panel-group');
                const panels = queryAllRoots(`.section-block[data-panel-group="${panelGroup}"]`);

                panels.forEach(panel => {
                    panel.style.display = 'none';
                });

                const selectedPanel = queryByIdAcrossRoots(select.value);
                if (selectedPanel) {
                    selectedPanel.style.display = 'block';
                }

                queueScrollUpdate();
            }

            select.addEventListener('change', updatePanels);
            select.dataset.reportBound = 'true';
            updatePanels();
        });
    }

    function onScroll() {
        const sections = queryAllRoots('.scroll-anchor[id]');
        const navItems = queryAllRoots('.nav-item:not(.nav-group-title):not(.nav-child)');
        const navChildren = queryAllRoots('.nav-child');
        const navGroupTitles = queryAllRoots('.nav-group-title');
        const navCurrentValue = queryAllRoots('.nav-current-value')[0] || null;
        const sectionPositions = sections
            .map(section => ({
                section,
                top: section.getBoundingClientRect().top,
            }))
            .sort((left, right) => left.top - right.top);
        let currentId = '';

        sectionPositions.forEach(({ section, top }) => {
            if (top <= 120) {
                currentId = section.getAttribute('id');
            }
        });

        if (!currentId && sectionPositions.length > 0) {
            const firstVisibleSection = sectionPositions.find(({ top }) => top > 0);
            currentId = firstVisibleSection ? firstVisibleSection.section.getAttribute('id') : '';
        }

        clearActive(navItems, navChildren, navGroupTitles);
        let matchedLabel = 'Top of report';

        navItems.forEach(item => {
            if (item.getAttribute('href') === '#' + currentId) {
                item.classList.add('active');
                matchedLabel = item.textContent.trim();
            }
        });

        let childMatched = false;
        navChildren.forEach(child => {
            if (child.getAttribute('href') === '#' + currentId) {
                child.classList.add('active');
                childMatched = true;
                matchedLabel = child.textContent.trim();
                const group = child.closest('.nav-group');
                if (group) {
                    const parentTitle = group.querySelector('.nav-group-title');
                    if (parentTitle) {
                        parentTitle.classList.add('active');
                    }
                }
            }
        });

        if (!childMatched) {
            navGroupTitles.forEach(title => {
                if (title.getAttribute('href') === '#' + currentId) {
                    title.classList.add('active');
                    matchedLabel = title.textContent.trim();
                }
            });
        }

        if (navCurrentValue) {
            navCurrentValue.textContent = matchedLabel || 'Top of report';
        }
    }

    function queueScrollUpdate() {
        if (scrollFrame !== null) {
            return;
        }

        scrollFrame = window.requestAnimationFrame(() => {
            scrollFrame = null;
            bindScrollListeners();
            bindPanelSelectors();
            onScroll();
        });
    }

    window.addEventListener('resize', queueScrollUpdate);
    window.addEventListener('hashchange', queueScrollUpdate);
    queueScrollUpdate();

    let attempts = 0;
    function refreshUntilReady() {
        queueScrollUpdate();
        attempts += 1;
        if (attempts >= 120) {
            return;
        }

        const hasNavigation = queryAllRoots('.nav-item').length > 0;
        const hasSections = queryAllRoots('.scroll-anchor[id]').length > 0;
        if (!hasNavigation || !hasSections) {
            window.requestAnimationFrame(refreshUntilReady);
        }
    }

    refreshUntilReady();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initReportInteractions);
} else {
    initReportInteractions();
}
