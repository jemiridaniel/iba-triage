import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FADE_MS, dismissLaunchScreen } from "./launch";

function mountLaunchScreen(): HTMLElement {
  document.body.innerHTML = '<div id="root"></div><div id="launch">Ibà</div>';
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

describe("dismissLaunchScreen", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setReducedMotion(false);
  });
  afterEach(() => vi.useRealTimers());

  it("fades out, then removes the launch screen on transitionend", () => {
    const el = mountLaunchScreen();
    dismissLaunchScreen();
    expect(el.classList.contains("iba-hide")).toBe(true);
    expect(document.getElementById("launch")).not.toBeNull(); // still fading

    el.dispatchEvent(new Event("transitionend"));
    expect(document.getElementById("launch")).toBeNull();
  });

  it("removes it even if transitionend never fires", () => {
    mountLaunchScreen();
    dismissLaunchScreen();
    vi.advanceTimersByTime(FADE_MS + 100);
    expect(document.getElementById("launch")).toBeNull();
  });

  it("removes it immediately without animation under prefers-reduced-motion", () => {
    setReducedMotion(true);
    const el = mountLaunchScreen();
    dismissLaunchScreen();
    expect(document.getElementById("launch")).toBeNull();
    expect(el.classList.contains("iba-hide")).toBe(false);
  });

  it("clears the 4s safety timeout set in index.html", () => {
    mountLaunchScreen();
    const clear = vi.spyOn(globalThis, "clearTimeout");
    window.__ibaLaunchTimer = setTimeout(() => {}, 4000);
    dismissLaunchScreen();
    expect(clear).toHaveBeenCalledWith(window.__ibaLaunchTimer);
  });

  it("does nothing when the launch screen is already gone", () => {
    document.body.innerHTML = '<div id="root"></div>';
    expect(() => dismissLaunchScreen()).not.toThrow();
  });
});
