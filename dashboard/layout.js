(() => {
  "use strict";
  const links = Array.from(document.querySelectorAll("a[data-section]"));
  const sections = ["overview", "training", "models", "encounters", "metrics"]
    .map(id => document.getElementById(id)).filter(Boolean);
  let queued = false;
  function markCurrent() {
    queued = false;
    const threshold = Math.min(window.innerHeight * .25, 180);
    let current = sections[0]?.id;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= threshold) current = section.id;
    }
    for (const link of links) {
      if (link.dataset.section === current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  }
  function schedule() {
    if (!queued) { queued = true; window.requestAnimationFrame(markCurrent); }
  }
  window.addEventListener("scroll", schedule, {passive: true});
  window.addEventListener("resize", schedule);
  window.addEventListener("hashchange", schedule);
  if (typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(schedule);
    sections.forEach(section => observer.observe(section));
  }
  schedule();
})();
