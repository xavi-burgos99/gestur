import {
  readFile,
  writeFile,
  rename,
  mkdir,
  readdir,
  stat,
  unlink,
} from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import Ajv from "ajv";

export const repository = path.resolve(import.meta.dirname, "../..");
export class ApiError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.statusCode = statusCode;
  }
}
export async function atomicJson(filename, data) {
  await mkdir(path.dirname(filename), { recursive: true, mode: 0o2770 });
  const temporary = `${filename}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, JSON.stringify(data, null, 2) + "\n", {
      mode: 0o660,
    });
    await rename(temporary, filename);
  } finally {
    await unlink(temporary).catch(() => {});
  }
}
export async function createStore({
  configPath,
  modelsDir,
  root = repository,
}) {
  const defaults = JSON.parse(
    await readFile(path.join(root, "config/default.json"), "utf8"),
  );
  const schema = JSON.parse(
    await readFile(path.join(root, "config/schema.json"), "utf8"),
  );
  const validate = new Ajv({ allErrors: true, strict: false }).compile(schema);
  await mkdir(modelsDir, { recursive: true, mode: 0o2770 });
  function check(config) {
    if (!validate(config))
      throw new ApiError(
        400,
        `Configuración no válida: ${validate.errors.map((e) => `${e.instancePath || "/"} ${e.message}`).join("; ")}`,
      );
    const outputs = config.controls.mappings
      .filter((m) => m.enabled)
      .map((m) => m.output);
    if (new Set(outputs).size !== outputs.length)
      throw new ApiError(
        400,
        "Cada movimiento solo puede tener un gesto activo.",
      );
    const ids = config.controls.mappings.map((m) => m.id);
    if (new Set(ids).size !== ids.length)
      throw new ApiError(400, "Los identificadores de gesto deben ser únicos.");
    for (const m of config.controls.mappings) {
      if (
        m.mode === "hybrid" &&
        !(m.left_threshold < m.center && m.center < m.right_threshold)
      )
        throw new ApiError(
          400,
          "El centro debe estar entre los umbrales izquierdo y derecho.",
        );
      if (
        m.mode === "stepped" &&
        (m.small_scale > m.large_scale ||
          m.threshold - m.hysteresis < 0 ||
          m.threshold + m.hysteresis > 1)
      )
        throw new ApiError(
          400,
          "Revisa las escalas y los umbrales del gesto por pasos.",
        );
    }
    return config;
  }
  async function listModels() {
    const entries = [];
    for (const directory of (await readdir(modelsDir)).filter((n) =>
      /^[a-f0-9-]{36}$/.test(n),
    )) {
      try {
        const metadata = JSON.parse(
          await readFile(
            path.join(modelsDir, directory, ".gestur-model.json"),
            "utf8",
          ),
        );
        if (typeof metadata.entrypoint !== "string") continue;
        const packageRoot = path.join(modelsDir, directory);
        const modelPath = path.resolve(packageRoot, metadata.entrypoint);
        if (!modelPath.startsWith(packageRoot + path.sep)) continue;
        const info = await stat(modelPath);
        if (!info.isFile()) continue;
        entries.push({
          id: `${directory}/${metadata.entrypoint}`,
          name: metadata.name,
          builtin: false,
          format: path.extname(metadata.entrypoint).slice(1).toUpperCase(),
          sourceFormat: metadata.sourceFormat,
          size: metadata.size,
          triangles: metadata.triangles,
          originalTriangles: metadata.originalTriangles,
          simplified: metadata.simplified === true,
          warnings: Array.isArray(metadata.warnings) ? metadata.warnings : [],
        });
      } catch {
        /* Incomplete imports are never listed. */
      }
    }
    return entries;
  }
  async function read() {
    try {
      const saved = JSON.parse(await readFile(configPath, "utf8"));
      // Retire only the former bundled default. User imports remain untouched.
      if (saved.schema_version === 1 && saved.active_model === "capitell.obj") {
        saved.active_model = null;
        check(saved);
        await atomicJson(configPath, saved);
      }
      return check(saved);
    } catch (error) {
      if (error.code !== "ENOENT")
        throw new ApiError(
          503,
          "La configuración guardada no es válida. Se ha conservado para poder revisarla.",
        );
      await atomicJson(configPath, defaults);
      return structuredClone(defaults);
    }
  }
  let queue = Promise.resolve();
  async function update(fn) {
    const result = queue.then(async () => {
      const next = check(await fn(await read()));
      if (
        next.active_model !== null &&
        !(await listModels()).some((m) => m.id === next.active_model)
      )
        throw new ApiError(400, "Selecciona un modelo disponible.");
      await atomicJson(configPath, next);
      return next;
    });
    queue = result.catch(() => {});
    return result;
  }
  return { read, update, listModels, defaults, schema };
}

// Check URI-bearing extensions too; loaders must never follow network references.
export function resourceUris(document) {
  const result = [];
  const pending = [document];
  while (pending.length) {
    const value = pending.pop();
    if (!value || typeof value !== "object") continue;
    for (const [key, child] of Object.entries(value)) {
      if (key === "uri") {
        if (typeof child !== "string")
          throw new ApiError(400, "El modelo contiene una URI no válida.");
        result.push(child);
      } else if (child && typeof child === "object") pending.push(child);
    }
  }
  return result;
}

// Only complete, self-contained GLB 2.0 files are accepted. Never follow external URIs.
export function validateGlb(buffer) {
  if (
    buffer.length < 20 ||
    buffer.readUInt32LE(0) !== 0x46546c67 ||
    buffer.readUInt32LE(4) !== 2 ||
    buffer.readUInt32LE(8) !== buffer.length
  )
    throw new ApiError(400, "El archivo no es un GLB 2.0 válido.");
  const length = buffer.readUInt32LE(12);
  if (
    buffer.readUInt32LE(16) !== 0x4e4f534a ||
    length % 4 ||
    20 + length > buffer.length
  )
    throw new ApiError(400, "La estructura GLB no es válida.");
  let doc;
  try {
    doc = JSON.parse(
      buffer
        .subarray(20, 20 + length)
        .toString("utf8")
        .trim(),
    );
  } catch {
    throw new ApiError(400, "El modelo contiene datos GLB no válidos.");
  }
  if (doc.asset?.version !== "2.0")
    throw new ApiError(400, "Solo se admite glTF 2.0.");
  for (const uri of resourceUris(doc)) {
    if (!/^data:/.test(uri))
      throw new ApiError(
        400,
        "Exporta el modelo como GLB con sus texturas incluidas.",
      );
  }
  let offset = 20 + length;
  while (offset < buffer.length) {
    if (offset + 8 > buffer.length)
      throw new ApiError(400, "El modelo está incompleto.");
    const size = buffer.readUInt32LE(offset);
    if (size % 4 || offset + 8 + size > buffer.length)
      throw new ApiError(400, "El modelo está incompleto.");
    offset += 8 + size;
  }
  return doc;
}
