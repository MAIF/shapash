document.addEventListener('DOMContentLoaded', function () {
    const sections = document.querySelectorAll('.scroll-section[id]');
    const navItems = document.querySelectorAll('.nav-item:not(.nav-group-title):not(.nav-child)');
    const navChildren = document.querySelectorAll('.nav-child');
    const navGroupTitles = document.querySelectorAll('.nav-group-title');
    const panelSelectors = document.querySelectorAll('.js-panel-select[data-panel-group]');

    function clearActive() {
        navItems.forEach(el => el.classList.remove('active'));
        navChildren.forEach(el => el.classList.remove('active'));
        navGroupTitles.forEach(el => el.classList.remove('active'));
    }

    function onScroll() {
        let currentId = '';
        sections.forEach(section => {
            if (window.scrollY >= (section.offsetTop - 150)) {
                currentId = section.getAttribute('id');
            }
        });
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 5) {
            if (sections.length > 0) {
                currentId = sections[sections.length - 1].getAttribute('id');
            }
        }

        clearActive();

        navItems.forEach(item => {
            if (item.getAttribute('href') === '#' + currentId) {
                item.classList.add('active');
            }
        });

        let childMatched = false;
        navChildren.forEach(child => {
            if (child.getAttribute('href') === '#' + currentId) {
                child.classList.add('active');
                childMatched = true;
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
                }
            });
        }
    }

    window.addEventListener('scroll', onScroll);
    onScroll();

    panelSelectors.forEach(select => {
        const panelGroup = select.getAttribute('data-panel-group');
        const panels = document.querySelectorAll(`.section-block[data-panel-group="${panelGroup}"]`);

        function updatePanels() {
            panels.forEach(panel => {
                panel.style.display = 'none';
            });

            const selectedPanel = document.getElementById(select.value);
            if (selectedPanel) {
                selectedPanel.style.display = 'block';
            }
        }

        select.addEventListener('change', updatePanels);
        updatePanels();
    });
});
