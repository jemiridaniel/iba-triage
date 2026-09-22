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
      includeAssets: ["icon.svg"],
      manifest: {
        name: "Iba: fever triage support",
        short_name: "Iba",
        description: "Fever-triage decision support for primary health care workers in Nigeria.",
        theme_color: "#115e59",
        background_color: "#ffffff",
        display: "standalone",
        start_url: "/",
        icons: [
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
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
