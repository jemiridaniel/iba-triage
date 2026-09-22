/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

declare global {
  interface Window {
    /** Safety timeout set inline in index.html; cleared when the app mounts. */
    __ibaLaunchTimer?: ReturnType<typeof setTimeout>;
    /** Date.now() at first paint, set inline in index.html. */
    __ibaLaunchStart?: number;
  }
}

export {};
