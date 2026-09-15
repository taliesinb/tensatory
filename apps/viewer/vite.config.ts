import { defineConfig } from "vite";

// Fixed port so the app can be saved to the Dock / added as a PWA at a stable URL.
export const PORT = 5180;

export default defineConfig({
  server: { port: PORT, strictPort: true, host: "127.0.0.1" },
  preview: { port: PORT, strictPort: true, host: "127.0.0.1" },
  build: { target: "es2022", sourcemap: true },
});
