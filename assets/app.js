(() => {
  "use strict";
  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  const menuButton = document.getElementById("menu-toggle");
  const languageToggle = document.getElementById("language-toggle");

  let savedLanguage = null;
  try { savedLanguage = localStorage.getItem("karmadock-language"); } catch (_) { savedLanguage = null; }
  const counterpart = root.dataset.counterpart || "";
  if (counterpart && (savedLanguage === "en" || savedLanguage === "ar") && savedLanguage !== root.lang) {
    window.location.replace(counterpart + window.location.hash);
    return;
  }
  if (languageToggle) {
    languageToggle.href = languageToggle.getAttribute("href") + window.location.hash;
    languageToggle.addEventListener("click", () => {
      try { localStorage.setItem("karmadock-language", languageToggle.dataset.language); } catch (_) { /* storage is optional */ }
    });
  }

  let savedTheme = null;
  try { savedTheme = localStorage.getItem("karmadock-theme"); } catch (_) { savedTheme = null; }
  if (savedTheme === "light" || savedTheme === "dark") root.dataset.theme = savedTheme;

  const effectiveDark = () => root.dataset.theme === "dark" ||
    (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const updateThemeButton = () => {
    const dark = effectiveDark();
    themeButton.setAttribute("aria-pressed", String(dark));
    themeButton.setAttribute("aria-label", dark ? themeButton.dataset.labelLight : themeButton.dataset.labelDark);
  };
  updateThemeButton();
  themeButton.addEventListener("click", () => {
    const next = effectiveDark() ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("karmadock-theme", next); } catch (_) { /* storage is optional */ }
    updateThemeButton();
  });

  menuButton.addEventListener("click", () => {
    const open = document.body.classList.toggle("nav-open");
    menuButton.setAttribute("aria-expanded", String(open));
  });
  document.querySelectorAll(".sidebar a").forEach((link) => link.addEventListener("click", () => {
    document.body.classList.remove("nav-open");
    menuButton.setAttribute("aria-expanded", "false");
  }));
})();
