import { defineConfig, type Plugin } from "vite";

// Fixed port, distinct from the viewer's 5180.
export const PORT = 5181;

/** Inline every emitted JS chunk and CSS asset into index.html so dist/index.html is one self-contained file
 *  that opens via file:// (module scripts loaded from separate files do not, because of CORS). */
function singleFile(): Plugin {
  return {
    name: "single-file",
    enforce: "post",
    generateBundle(_opts, bundle) {
      for (const [name, out] of Object.entries(bundle)) {
        if (out.type !== "asset" || !name.endsWith(".html")) continue;
        let html = String(out.source);
        html = html.replace(/<script type="module"[^>]*src="\.?\/?([^"]+)"[^>]*><\/script>/g, (m, file: string) => {
          const chunk = bundle[file];
          if (!chunk || chunk.type !== "chunk") return m;
          delete bundle[file];
          return `<script type="module">\n${chunk.code}\n</script>`;
        });
        html = html.replace(/<link rel="stylesheet"[^>]*href="\.?\/?([^"]+)"[^>]*>/g, (m, file: string) => {
          const asset = bundle[file];
          if (!asset || asset.type !== "asset") return m;
          delete bundle[file];
          return `<style>\n${String(asset.source)}\n</style>`;
        });
        out.source = html;
      }
    },
  };
}

export default defineConfig({
  base: "./",
  server: { port: PORT, strictPort: true, host: "127.0.0.1" },
  preview: { port: PORT, strictPort: true, host: "127.0.0.1" },
  build: { target: "es2022", sourcemap: false, modulePreload: false, cssCodeSplit: false, assetsInlineLimit: 1e9 },
  plugins: [singleFile()],
});
