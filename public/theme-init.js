/* Runs synchronously in <head>, before the first paint, so the stored theme is
   already on <html> when the stylesheets resolve. The page modules call
   applyTheme() too, but they are deferred and run after first paint — which is
   a black→light flash on every navigation for a light-theme user. This is a
   classic (non-module) script precisely so it blocks the first paint.

   Key and values mirror app.js / settings.js / studio.js. Any storage error
   (private mode, disabled storage) leaves the CSS default in place. */
(function () {
  try {
    var theme = localStorage.getItem('llm-chat-theme');
    if (theme === 'dark' || theme === 'light') {
      document.documentElement.dataset.theme = theme;
    }
  } catch (e) {
    /* no storage access: keep the default dark surface */
  }
})();
