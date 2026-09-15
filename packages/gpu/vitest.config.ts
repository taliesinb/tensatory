import { defineConfig } from "vitest/config";

// One Dawn (native WebGPU) instance per run, created in beforeAll and destroyed
// in afterAll; keep test files sequential so they share it cleanly.
export default defineConfig({
  test: { pool: "threads", fileParallelism: false, testTimeout: 60000, hookTimeout: 60000 },
});
