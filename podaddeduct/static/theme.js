/* podaddeduct theme switcher: System / Light / Dark, remembered per browser. */
(function () {
  var KEY = "podaddeduct-theme";

  function current() {
    try {
      return localStorage.getItem(KEY) || "system";
    } catch (e) {
      return "system";
    }
  }

  function paint(choice) {
    document.documentElement.setAttribute("data-theme", choice);
    document.querySelectorAll("[data-theme-btn]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-theme-btn") === choice ? "true" : "false");
    });
  }

  function choose(choice) {
    try {
      localStorage.setItem(KEY, choice);
    } catch (e) {}
    paint(choice);
  }

  // Paint ASAP (also set inline in <head> to avoid a flash).
  paint(current());

  document.addEventListener("DOMContentLoaded", function () {
    paint(current());
    document.querySelectorAll("[data-theme-btn]").forEach(function (b) {
      b.addEventListener("click", function () { choose(b.getAttribute("data-theme-btn")); });
    });
  });

  // Follow OS changes while "System" is selected.
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: light)");
    var onChange = function () { if (current() === "system") paint("system"); };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }
})();
