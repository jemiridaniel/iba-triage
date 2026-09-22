/** Removes the inline launch screen (index.html #launch) once the app has mounted. */

export const FADE_MS = 250;

export function dismissLaunchScreen(): void {
  clearTimeout(window.__ibaLaunchTimer);
  const el = document.getElementById("launch");
  if (!el) return;

  const reduced =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) {
    el.remove(); // no fade
    return;
  }

  const remove = () => el.remove();
  el.classList.add("iba-hide"); // CSS transitions opacity to 0
  el.addEventListener("transitionend", remove, { once: true });
  setTimeout(remove, FADE_MS + 100); // in case transitionend never fires
}
