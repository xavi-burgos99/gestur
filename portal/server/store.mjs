import {
  readFile,
  writeFile,
  rename,
  mkdir,
  readdir,
  stat,
  realpath,
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
    const libraryRoot = await realpath(modelsDir);
    const modelPattern = new RegExp(schema.properties.active_model.pattern);
    for (const directory of (await readdir(modelsDir))
      .filter((n) => /^[a-f0-9-]{36}$/.test(n))
      .sort()) {
      try {
        const metadata = JSON.parse(
          await readFile(
            path.join(modelsDir, directory, ".gestur-model.json"),
            "utf8",
          ),
        );
        if (
          typeof metadata.entrypoint !== "string" ||
          !modelPattern.test(`${directory}/${metadata.entrypoint}`)
        )
          continue;
        const packageRoot = await realpath(path.join(modelsDir, directory));
        if (!packageRoot.startsWith(libraryRoot + path.sep)) continue;
        const modelPath = await realpath(
          path.resolve(packageRoot, metadata.entrypoint),
        );
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
  // Reconciliation and explicit changes share one queue. A read must never
  // restore an older selection while an import or a user's change is saving.
  let queue = Promise.resolve();
  function serialized(fn) {
    const result = queue.then(fn);
    queue = result.catch(() => {});
    return result;
  }
  async function readCurrent(models) {
    let saved;
    let missing = false;
    let persistedSelection;
    try {
      saved = JSON.parse(await readFile(configPath, "utf8"));
      persistedSelection = saved.active_model;
      if (saved.schema_version === 1 && saved.active_model === "capitell.obj")
        saved.active_model = null;
      check(saved);
    } catch (error) {
      if (error.code !== "ENOENT")
        throw new ApiError(
          503,
          "La configuración guardada no es válida. Se ha conservado para poder revisarla.",
        );
      saved = structuredClone(defaults);
      missing = true;
    }
    const original = saved.active_model;
    if (!models.some((model) => model.id === original))
      saved.active_model = models[0]?.id ?? null;
    // Also persist the legacy migration even if the resulting library is empty.
    if (missing || persistedSelection !== saved.active_model)
      await atomicJson(configPath, saved);
    return saved;
  }
  function read() {
    return serialized(async () => readCurrent(await listModels()));
  }
  function catalog() {
    return serialized(async () => {
      const models = await listModels();
      const current = await readCurrent(models);
      return { models, active: current.active_model };
    });
  }
  function update(fn) {
    return serialized(async () => {
      const models = await listModels();
      const next = check(await fn(await readCurrent(models)));
      if (next.active_model === null && models.length)
        throw new ApiError(
          400,
          "Debe haber un modelo seleccionado mientras haya modelos disponibles.",
        );
      if (
        next.active_model !== null &&
        !models.some((m) => m.id === next.active_model)
      )
        throw new ApiError(400, "Selecciona un modelo disponible.");
      await atomicJson(configPath, next);
      return next;
    });
  }
  try {
    await read();
  } catch (error) {
    // Keep administration reachable to report malformed saved configuration;
    // never reset or overwrite it during startup reconciliation.
    if (!(error instanceof ApiError) || error.statusCode !== 503) throw error;
  }
  return { read, update, listModels, catalog, defaults, schema };
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
