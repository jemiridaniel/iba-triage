import { defineConfig } from "vitest/config";

// Separate from vite.config.ts: vitest ships its own (older) vite types, which clash with
// the plugin types used for the build.
export default defineConfig({
  test: { environment: "jsdom", include: ["src/**/*.test.ts"] },
});
