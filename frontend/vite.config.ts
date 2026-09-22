import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      // public/manifest.webmanifest is hand-written and checked by tests/test_pwa.py.
      manifest: false,
      includeAssets: ["icon.svg", "apple-touch-icon.png", "manifest.webmanifest"],
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png,webmanifest}"],
        // Launch images are large and only used by iOS at launch: served, not precached.
        globIgnores: ["**/splash/**"],
        navigateFallback: "/index.html",
        // Triage always needs the network; never serve API responses from cache.
        navigateFallbackDenylist: [/^\/triage/, /^\/meta/, /^\/health/, /^\/docs/],
        runtimeCaching: [],
      },
    }),
  ],
  server: {
    proxy: { "/triage": api, "/meta": api, "/health": api },
  },
  build: { target: "es2019", sourcemap: false },
});
