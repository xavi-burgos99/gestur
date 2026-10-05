import {
  readFile,
  writeFile,
  rename,
  mkdir,
  readdir,
  stat,
  lstat,
  rm,
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
const zeroOrientation = () => ({ x: 0, y: 0, z: 0 });
function validOrientation(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 3 &&
    ["x", "y", "z"].every(
      (axis) =>
        Object.hasOwn(value, axis) &&
        Number.isInteger(value[axis]) &&
        [0, 90, 180, 270].includes(value[axis]),
    )
  );
}
function modelOrientation(value) {
  return validOrientation(value)
    ? { x: value.x, y: value.y, z: value.z }
    : zeroOrientation();
}
function validateModelUrl(value) {
  if (value === null) return null;
  const invalid = () =>
    new ApiError(
      400,
      "La URL debe ser HTTP o HTTPS, sin credenciales, y ocupar como máximo 2048 bytes.",
    );
  if (
    typeof value !== "string" ||
    /[\u0000-\u001f\u007f-\u009f\ud800-\udfff]/u.test(value)
  )
    throw invalid();
  const url = value.trim();
  if (!url) return null;
  if (
    url.length > 2048 ||
    Buffer.byteLength(url, "utf8") > 2048 ||
    /\s|\\/.test(url) ||
    !/^https?:\/\//i.test(url)
  )
    throw invalid();
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw invalid();
  }
  const authority = url.split("://", 2)[1].split(/[/?#]/, 1)[0];
  if (
    !authority ||
    !parsed.hostname ||
    parsed.username ||
    parsed.password ||
    authority.includes("@")
  )
    throw invalid();
  return url;
}
function modelContent(value = null) {
  const empty = { title: "", description: "", placement: "bottom" };
  if (value === null || value === undefined) return empty;
  if (
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).length !== 3 ||
    !["top", "bottom"].includes(value.placement) ||
    ![
      ["title", 160],
      ["description", 1200],
    ].every(
      ([key, limit]) =>
        typeof value[key] === "string" &&
        value[key].length <= limit &&
        !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\ud800-\udfff]/u.test(
          value[key],
        ),
    )
  )
    throw new ApiError(
      400,
      "El contenido requiere un título de hasta 160 caracteres, una descripción de hasta 1200 y una posición válida.",
    );
  return {
    title: value.title.trim(),
    description: value.description.trim(),
    placement: value.placement,
  };
}
function storedContent(value) {
  try {
    return modelContent(value);
  } catch {
    return modelContent();
  }
}
function modelUrl(value) {
  try {
    return validateModelUrl(value ?? null);
  } catch {
    return null;
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
    config = structuredClone(config);
    if (config?.schema_version === 1) {
      if (!Object.hasOwn(config, "screen"))
        config.screen = structuredClone(defaults.screen);
      for (const [section, key, value] of [
        ["render", "ambient_light", "none"],
        ["render", "exposure", 50],
        ["controls", "idle_mode", "return"],
      ]) {
        if (
          config[section] &&
          typeof config[section] === "object" &&
          !Array.isArray(config[section]) &&
          !Object.hasOwn(config[section], key)
        )
          config[section][key] = value;
      }
      const presets = new Map([
        ["soft", "studio"],
        ["warm", "sunset"],
        ["cool", "gallery"],
        ["contrast", "rim"],
      ]);
      const preset = config.render?.ambient_light;
      if (typeof preset === "string" && presets.has(preset))
        config.render.ambient_light = presets.get(preset);
    }
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
          orientation: modelOrientation(metadata.orientation),
          url: modelUrl(metadata.url),
          content: storedContent(metadata.content),
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
  function validateParameters(parameters) {
    const sections = ["tracking", "render", "controls"];
    if (
      !parameters ||
      typeof parameters !== "object" ||
      Array.isArray(parameters) ||
      Object.keys(parameters).length !== sections.length ||
      !sections.every((key) => Object.hasOwn(parameters, key))
    )
      throw new ApiError(
        400,
        "El preset debe contener seguimiento, renderizado y controles completos.",
      );
    const checked = check({ ...defaults, ...parameters });
    return Object.fromEntries(sections.map((key) => [key, checked[key]]));
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
    let persistedConfig;
    try {
      saved = JSON.parse(await readFile(configPath, "utf8"));
      persistedConfig = JSON.stringify(saved);
      if (saved.schema_version === 1 && saved.active_model === "capitell.obj")
        saved.active_model = null;
      saved = check(saved);
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
    // Persist schema-v1 field migrations and selection reconciliation together.
    if (missing || persistedConfig !== JSON.stringify(saved))
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
  async function locateModel(id, models) {
    if (typeof id !== "string" || !models.some((model) => model.id === id))
      throw new ApiError(404, "El modelo ya no está disponible.");
    const libraryRoot = await realpath(modelsDir);
    const packageName = id.split("/")[0];
    if (!/^[a-f0-9-]{36}$/.test(packageName))
      throw new ApiError(400, "El identificador del modelo no es válido.");
    const packagePath = path.join(libraryRoot, packageName);
    const metadataPath = path.join(packagePath, ".gestur-model.json");
    const info = await lstat(packagePath);
    if (
      !info.isDirectory() ||
      info.isSymbolicLink() ||
      (await realpath(packagePath)) !== packagePath ||
      (await lstat(metadataPath)).isSymbolicLink()
    )
      throw new ApiError(400, "La carpeta del modelo no es válida.");
    const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
    if (`${packageName}/${metadata.entrypoint}` !== id)
      throw new ApiError(
        409,
        "El modelo ha cambiado. Actualiza la biblioteca.",
      );
    return { packagePath, packageName, metadataPath, metadata };
  }
  function validateMutation(body, editing) {
    const allowed = editing
      ? ["id", "name", "orientation", "url", "content"]
      : ["id"];
    if (
      !body ||
      typeof body !== "object" ||
      Array.isArray(body) ||
      typeof body.id !== "string" ||
      !body.id ||
      Object.keys(body).some((key) => !allowed.includes(key)) ||
      (editing &&
        !Object.hasOwn(body, "name") &&
        !Object.hasOwn(body, "orientation") &&
        !Object.hasOwn(body, "url") &&
        !Object.hasOwn(body, "content"))
    )
      throw new ApiError(
        400,
        "Indica el modelo y los cambios que quieres guardar.",
      );
    if (
      Object.hasOwn(body, "name") &&
      (typeof body.name !== "string" ||
        body.name.trim().length < 1 ||
        body.name.trim().length > 100 ||
        /[\u0000-\u001f\u007f]/.test(body.name))
    )
      throw new ApiError(
        400,
        "El nombre debe tener entre 1 y 100 caracteres en una sola línea.",
      );
    if (
      Object.hasOwn(body, "orientation") &&
      !validOrientation(body.orientation)
    )
      throw new ApiError(
        400,
        "La orientación debe indicar X, Y y Z con giros de 0, 90, 180 o 270 grados.",
      );
    if (Object.hasOwn(body, "url")) validateModelUrl(body.url);
    if (Object.hasOwn(body, "content")) modelContent(body.content);
  }
  function updateModel(body) {
    validateMutation(body, true);
    return serialized(async () => {
      const models = await listModels();
      const located = await locateModel(body.id, models);
      const config = await readCurrent(models);
      const metadata = { ...located.metadata };
      if (Object.hasOwn(body, "name")) metadata.name = body.name.trim();
      if (Object.hasOwn(body, "orientation"))
        metadata.orientation = modelOrientation(body.orientation);
      if (Object.hasOwn(body, "url")) metadata.url = validateModelUrl(body.url);
      if (Object.hasOwn(body, "content"))
        metadata.content = modelContent(body.content);
      await atomicJson(located.metadataPath, metadata);
      // The renderer stats this directory once a second instead of reading all
      // metadata every frame. A rename within a package does not change it.
      try {
        await atomicJson(path.join(modelsDir, ".catalog-revision.json"), {
          revision: randomUUID(),
        });
      } catch (error) {
        await atomicJson(located.metadataPath, located.metadata);
        throw error;
      }
      const model = {
        ...models.find((entry) => entry.id === body.id),
        name: metadata.name,
        orientation: modelOrientation(metadata.orientation),
        url: modelUrl(metadata.url),
        content: storedContent(metadata.content),
      };
      return { model, config };
    });
  }
  function deleteModel(body) {
    validateMutation(body, false);
    return serialized(async () => {
      const models = await listModels();
      const located = await locateModel(body.id, models);
      const current = await readCurrent(models);
      const remaining = models.filter((model) => model.id !== body.id);
      const config = {
        ...current,
        active_model:
          current.active_model === body.id
            ? (remaining[0]?.id ?? null)
            : current.active_model,
      };
      const trash = path.join(modelsDir, `.trash-${randomUUID()}`);
      // Hide the complete package atomically. Do not destroy any assets until
      // the replacement selection has been saved successfully.
      await rename(located.packagePath, trash);
      try {
        await atomicJson(configPath, check(config));
      } catch (error) {
        await rename(trash, located.packagePath);
        throw error;
      }
      // Deletion is committed. A failed cleanup leaves a hidden package for
      // later maintenance; it must not reappear or invalidate the saved choice.
      await rm(trash, { recursive: true, force: true }).catch(() => {});
      return { config };
    });
  }
  try {
    await read();
  } catch (error) {
    // Keep administration reachable to report malformed saved configuration;
    // never reset or overwrite it during startup reconciliation.
    if (!(error instanceof ApiError) || error.statusCode !== 503) throw error;
  }
  return {
    read,
    update,
    listModels,
    catalog,
    updateModel,
    deleteModel,
    validateParameters,
    defaults,
    schema,
  };
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
