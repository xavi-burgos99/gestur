import { readFile } from "node:fs/promises";
import { createApp } from "./app.mjs";
const token =
  process.env.GESTUR_ADMIN_TOKEN ||
  (await readFile(
    process.env.GESTUR_TOKEN_FILE || "/etc/gestur/portal-token",
    "utf8",
  )
    .then((value) => value.trim())
    .catch((error) => {
      if (error.code !== "ENOENT") throw error;
      return undefined; // Fresh devices use onboarding; configured devices use a hash.
    }));
const app = await createApp({
  configPath: process.env.GESTUR_CONFIG || "/var/lib/gestur/config.json",
  modelsDir: process.env.GESTUR_MODELS_DIR || "/var/lib/gestur/models",
  token,
});
await app.listen({
  port: Number(process.env.PORT || 3000),
  host: process.env.HOST || "0.0.0.0",
});
console.log(
  "GESTUR portal listening on port",
  Number(process.env.PORT || 3000),
);
for (const signal of ["SIGINT", "SIGTERM"])
  process.on(signal, async () => {
    await app.close();
    process.exit(0);
  });
