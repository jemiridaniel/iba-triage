/** Removes the inline launch screen (index.html #launch) once the app has mounted.
 *
 * First load of a browser session: hold for at least LAUNCH_MIN_MS from first paint (or until
 * the app mounts, whichever is later), so the logo assembly animation completes rather than
 * flickering. If the app is still not ready by then, the dot keeps doing an idle hop.
 * Later loads in the same session: remove as soon as the app mounts.
 * A tap anywhere on the launch screen dismisses it at once (index.html handles taps that
 * happen before this bundle loads).
 */

// The assembly animation in index.html runs ~1,400 ms (stem 0-450, dot 380-1400, wordmark
// 1150-1400); hold a little longer so it never gets cut off mid-bounce.
export const LAUNCH_MIN_MS = 1550;
export const FADE_MS = 300;

const SESSION_KEY = "iba.launched";

function seenThisSession(): boolean {
  try {
    return sessionStorage.getItem(SESSION_KEY) === "1";
  } catch {
    return false; // private mode / storage blocked: treat as a first load
  }
}

function markSeen(): void {
  try {
    sessionStorage.setItem(SESSION_KEY, "1");
  } catch {
    /* ignore */
  }
}

function prefersReducedMotion(): boolean {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function remove(el: HTMLElement): void {
  el.remove();
}

function fadeOut(el: HTMLElement): void {
  if (!el.isConnected) return; // already dismissed by a tap or the safety timeout
  if (prefersReducedMotion()) {
    remove(el);
    return;
  }
  el.classList.add("iba-hide"); // CSS transitions opacity to 0 over FADE_MS
  el.addEventListener("transitionend", () => remove(el), { once: true });
  setTimeout(() => remove(el), FADE_MS + 100); // in case transitionend never fires
}

export function dismissLaunchScreen(): void {
  clearTimeout(window.__ibaLaunchTimer);
  const el = document.getElementById("launch");
  if (!el) {
    markSeen();
    return;
  }
  el.addEventListener("click", () => remove(el), { once: true });
  el.classList.add("iba-mounted"); // stops the idle hop; the sequence itself still finishes

  const firstLoad = !seenThisSession();
  markSeen();
  if (!firstLoad) {
    fadeOut(el);
    return;
  }
  const painted = window.__ibaLaunchStart ? Date.now() - window.__ibaLaunchStart : 0;
  setTimeout(() => fadeOut(el), Math.max(0, LAUNCH_MIN_MS - painted));
}
