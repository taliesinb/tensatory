import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, type Plugin } from "vite";

// Fixed port so the app can be saved to the Dock / added as a PWA at a stable URL.
export const PORT = 5180;

/** where the dev server writes the clients' logs (`clientLog`): one file per day, gitignored */
export const CLIENT_LOG_DIR = join(import.meta.dirname, ".logs");

/**
 * Client log sink (dev server only): the viewer POSTs its console / status / error lines to `/__tensatory/log`
 * (`apps/viewer/src/log.ts`) and they are appended to `.logs/client-<date>.log`, each prefixed with the server
 * time and the client's session tag. A Safari web app (Dock) or another browser has no console we can read; this
 * file is the console of every client of this server: `tail -f apps/viewer/.logs/client-*.log`.
 */
function clientLog(): Plugin {
  return {
    name: "tensatory-client-log",
    configureServer(server) {
      mkdirSync(CLIENT_LOG_DIR, { recursive: true });
      server.middlewares.use("/__tensatory/log", (req, res) => {
        if (req.method !== "POST") { res.statusCode = 405; res.end(); return; }
        let body = "";
        req.on("data", (c: Buffer) => { body += c; });
        req.on("end", () => {
          // local time (the user's clock), as the shell's `date` would print it
          const now = new Date(), pad = (n: number, w = 2) => String(n).padStart(w, "0");
          const stamp = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}.${pad(now.getMilliseconds(), 3)}`;
          const file = join(CLIENT_LOG_DIR, `client-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}.log`);
          const lines = body.split("\n").filter(Boolean).map((l) => `${stamp} ${l}\n`).join("");
          try { appendFileSync(file, lines); } catch (e) { console.error(e); }
          res.statusCode = 204; res.end();
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [clientLog()],
  server: { port: PORT, strictPort: true, host: "127.0.0.1" },
  preview: { port: PORT, strictPort: true, host: "127.0.0.1" },
  build: { target: "es2022", sourcemap: true },
});
