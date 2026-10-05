import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { existsSync } from "node:fs";
import { ApiError, repository } from "./store.mjs";
const execute = promisify(execFile);
export async function preflightModel(filename) {
  const python =
    process.env.GESTUR_PYTHON ||
    (existsSync("/opt/gestur/.venv/bin/python")
      ? "/opt/gestur/.venv/bin/python"
      : path.join(repository, ".venv/bin/python"));
  try {
    const { stdout } = await execute(
      python,
      [path.join(repository, "scripts/check_model.py"), filename],
      { timeout: 60000, maxBuffer: 65536, windowsHide: true },
    );
    const result = JSON.parse(stdout.trim());
    if (result.ok !== true) throw new Error("invalid");
    return result;
  } catch (error) {
    if (error.code === "ENOENT")
      throw new ApiError(
        503,
        "El validador 3D no está instalado. Revisa la instalación del visualizador.",
      );
    throw new ApiError(
      422,
      "El visualizador no puede abrir este modelo. Revisa su geometría, materiales y texturas y vuelve a exportarlo.",
    );
  }
}
