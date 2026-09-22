import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FADE_MS, LAUNCH_MIN_MS, dismissLaunchScreen } from "./launch";

const SESSION_KEY = "iba.launched";

function mountLaunchScreen(paintedMsAgo = 0): HTMLElement {
  document.body.innerHTML = '<div id="root"></div><div id="launch">Ibà</div>';
  window.__ibaLaunchStart = Date.now() - paintedMsAgo;
  return document.getElementById("launch")!;
}

function setReducedMotion(reduced: boolean): void {
  window.matchMedia = ((query: string) => ({
    matches: reduced && query.includes("reduce"),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  })) as unknown as typeof window.matchMedia;
}

const present = () => document.getElementById("launch") !== null;

describe("dismissLaunchScreen", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setReducedMotion(false);
    sessionStorage.clear();
  });
  afterEach(() => vi.useRealTimers());

  describe("first load of a session", () => {
    it("holds for LAUNCH_MIN_MS from first paint, then fades", () => {
      const el = mountLaunchScreen();
      dismissLaunchScreen();

      vi.advanceTimersByTime(LAUNCH_MIN_MS - 1);
      expect(el.classList.contains("iba-hide")).toBe(false); // still held

      vi.advanceTimersByTime(1);
      expect(el.classList.contains("iba-hide")).toBe(true); // fading
      expect(present()).toBe(true);

      el.dispatchEvent(new Event("transitionend"));
      expect(present()).toBe(false);
    });

    it("counts time already spent painting, so a slow mount waits no longer", () => {
      const el = mountLaunchScreen(LAUNCH_MIN_MS + 500); // app mounted after the minimum
      dismissLaunchScreen();
      vi.advanceTimersByTime(0);
      expect(el.classList.contains("iba-hide")).toBe(true); // fades at once
    });

    it("removes it after the fade even if transitionend never fires", () => {
      mountLaunchScreen();
      dismissLaunchScreen();
      vi.advanceTimersByTime(LAUNCH_MIN_MS + FADE_MS + 100);
      expect(present()).toBe(false);
    });
  });

  it("removes it as soon as the app mounts on later loads in the same session", () => {
    sessionStorage.setItem(SESSION_KEY, "1");
    const el = mountLaunchScreen();
    dismissLaunchScreen();
    expect(el.classList.contains("iba-hide")).toBe(true); // no minimum hold
    vi.advanceTimersByTime(FADE_MS + 100);
    expect(present()).toBe(false);
  });

  it("dismisses immediately when the launch screen is tapped", () => {
    const el = mountLaunchScreen();
    dismissLaunchScreen();
    el.dispatchEvent(new MouseEvent("click"));
    expect(present()).toBe(false); // gone before the minimum hold elapses

    vi.advanceTimersByTime(LAUNCH_MIN_MS + FADE_MS + 100); // late timers must not throw
    expect(present()).toBe(false);
  });

  it("still holds under prefers-reduced-motion, but removes without fading", () => {
    setReducedMotion(true);
    const el = mountLaunchScreen();
    dismissLaunchScreen();

    vi.advanceTimersByTime(LAUNCH_MIN_MS - 1);
    expect(present()).toBe(true);

    vi.advanceTimersByTime(1);
    expect(present()).toBe(false);
    expect(el.classList.contains("iba-hide")).toBe(false); // no fade class
  });

  it("stops the idle hop as soon as the app mounts", () => {
    const el = mountLaunchScreen();
    dismissLaunchScreen();
    expect(el.classList.contains("iba-mounted")).toBe(true); // CSS: .iba-mounted -> no idle
    expect(el.classList.contains("iba-hide")).toBe(false); // sequence still finishing
  });

  it("holds long enough for the assembly animation to finish", () => {
    expect(LAUNCH_MIN_MS).toBeGreaterThanOrEqual(1400 + 100); // sequence + margin
    expect(LAUNCH_MIN_MS).toBeLessThanOrEqual(2000); // still feels like a launch, not a wait
  });

  it("clears the 4s safety timeout set in index.html", () => {
    mountLaunchScreen();
    const clear = vi.spyOn(globalThis, "clearTimeout");
    window.__ibaLaunchTimer = setTimeout(() => {}, 4000);
    dismissLaunchScreen();
    expect(clear).toHaveBeenCalledWith(window.__ibaLaunchTimer);
  });

  it("treats blocked sessionStorage as a first load", () => {
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    const el = mountLaunchScreen();
    expect(() => dismissLaunchScreen()).not.toThrow();
    vi.advanceTimersByTime(LAUNCH_MIN_MS);
    expect(el.classList.contains("iba-hide")).toBe(true);
    getItem.mockRestore();
    setItem.mockRestore();
  });

  it("does nothing when the launch screen is already gone", () => {
    document.body.innerHTML = '<div id="root"></div>';
    expect(() => dismissLaunchScreen()).not.toThrow();
    expect(sessionStorage.getItem(SESSION_KEY)).toBe("1");
  });
});
