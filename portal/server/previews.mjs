import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash, randomUUID } from "node:crypto";
import { readFile, mkdir, stat, realpath, rename, rm } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";
import { ApiError, repository } from "./store.mjs";
const execute = promisify(execFile);

export function createPreviews({ store, modelsDir }) {
  // One disposable renderer at a time, with identical requests sharing the job.
  let queue = Promise.resolve();
  const pending = new Map();
  return async function preview(id) {
    const { models } = await store.catalog();
    const model = models.find((entry) => entry.id === id);
    if (!model) throw new ApiError(404, "El modelo ya no está disponible.");
    const root = await realpath(modelsDir);
    const filename = await realpath(path.join(root, model.id));
    if (!filename.startsWith(root + path.sep))
      throw new ApiError(404, "Modelo no válido.");
    const info = await stat(filename);
    const key = createHash("sha256")
      .update(
        JSON.stringify([
          "v2",
          model.id,
          model.orientation,
          info.size,
          info.mtimeMs,
        ]),
      )
      .digest("hex");
    const folder = path.join(path.dirname(root), ".previews");
    const output = path.join(folder, key + ".png");
    try {
      return await readFile(output);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    if (!pending.has(key)) {
      const task = queue
        .catch(() => {})
        .then(async () => {
          await mkdir(folder, { recursive: true, mode: 0o2770 });
          const temporary = path.join(folder, randomUUID() + ".png");
          try {
            const python =
              process.env.GESTUR_PYTHON ||
              (existsSync("/opt/gestur/.venv/bin/python")
                ? "/opt/gestur/.venv/bin/python"
                : path.join(repository, ".venv/bin/python"));
            await execute(
              python,
              [
                path.join(repository, "scripts/render_preview.py"),
                filename,
                temporary,
                JSON.stringify(model.orientation),
              ],
              {
                timeout: 40000,
                maxBuffer: 65536,
                env: {
                  ...process.env,
                  OPENBLAS_NUM_THREADS: "1",
                  OMP_NUM_THREADS: "1",
                },
              },
            );
            await rename(temporary, output);
            return await readFile(output);
          } catch {
            throw new ApiError(
              503,
              "No se pudo generar la vista previa del modelo.",
            );
          } finally {
            await rm(temporary, { force: true });
          }
        });
      queue = task;
      pending.set(key, task);
      task.finally(() => pending.delete(key)).catch(() => {});
    }
    return pending.get(key);
  };
}
