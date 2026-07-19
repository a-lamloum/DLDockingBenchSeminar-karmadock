(() => {
  "use strict";
  const dialog = document.getElementById("search-dialog");
  const trigger = document.getElementById("search-trigger");
  const closeButton = document.getElementById("search-close");
  const input = document.getElementById("search-input");
  const status = document.getElementById("search-status");
  const list = document.getElementById("search-results");
  const payload = window.KD_SEARCH_INDEX;
  if (!payload || payload.lang !== document.documentElement.lang || !Array.isArray(payload.entries)) {
    throw new Error("Search index language mismatch");
  }
  const index = payload.entries;
  let priorFocus = null;
  let matches = [];
  let active = -1;

  const normalize = (value) => value.toLocaleLowerCase().normalize("NFKD")
    .replace(/\p{M}/gu, "").replace(/ـ/g, "").replace(/[أإآ]/g, "ا").trim().replace(/\s+/g, " ");
  const fixtures = Array.isArray(window.KD_NORMALIZATION_FIXTURES) ? window.KD_NORMALIZATION_FIXTURES : [];
  fixtures.forEach((fixture) => {
    if (normalize(fixture.input) !== fixture.expected) throw new Error("Search normalization fixture mismatch");
  });
  const editable = (target) => target instanceof HTMLElement &&
    (target.matches("input, textarea, select") || target.isContentEditable);

  const open = () => {
    priorFocus = document.activeElement;
    if (!dialog.open) dialog.showModal();
    input.focus();
    input.select();
  };
  const close = () => {
    if (dialog.open) dialog.close();
    if (priorFocus instanceof HTMLElement) priorFocus.focus();
  };

  const draw = () => {
    list.replaceChildren();
    matches.forEach((entry, position) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      const title = document.createElement("span");
      const excerpt = document.createElement("span");
      link.href = entry.url;
      link.dataset.active = String(position === active);
      link.tabIndex = position === active ? 0 : -1;
      title.className = "result-title";
      excerpt.className = "result-excerpt";
      title.textContent = `${entry.title} — ${entry.heading}`;
      excerpt.textContent = entry.excerpt;
      link.append(title, excerpt);
      item.append(link);
      list.append(item);
    });
    if (active >= 0) list.querySelectorAll("a")[active]?.focus();
  };

  const search = () => {
    const query = normalize(input.value);
    active = -1;
    if (!query) {
      matches = [];
      status.textContent = dialog.dataset.empty;
      draw();
      return;
    }
    const terms = query.split(/\s+/);
    matches = index.filter((entry) => terms.every((term) => entry.text.includes(term))).slice(0, 24);
    status.textContent = matches.length
      ? `${matches.length} ${matches.length === 1 ? dialog.dataset.resultSingular : dialog.dataset.resultPlural}.`
      : dialog.dataset.none;
    draw();
  };

  trigger.addEventListener("click", open);
  closeButton.addEventListener("click", close);
  input.addEventListener("input", search);
  dialog.addEventListener("click", (event) => { if (event.target === dialog) close(); });
  dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
  input.addEventListener("keydown", (event) => {
    if (!matches.length || !["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowDown") active = (active + 1) % matches.length;
    if (event.key === "ArrowUp") active = (active - 1 + matches.length) % matches.length;
    if (event.key === "Enter") {
      const choice = matches[active >= 0 ? active : 0];
      if (choice) window.location.href = choice.url;
      return;
    }
    draw();
  });
  list.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowDown") active = (active + 1) % matches.length;
    if (event.key === "ArrowUp") active = (active - 1 + matches.length) % matches.length;
    if (event.key === "Enter" && active >= 0) window.location.href = matches[active].url;
    draw();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && dialog.open) { event.preventDefault(); close(); return; }
    if ((event.key === "/" && !editable(event.target)) || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k")) {
      event.preventDefault();
      open();
    }
  });
  search();
})();
