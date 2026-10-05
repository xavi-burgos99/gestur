import { constants } from "node:fs";
import { open } from "node:fs/promises";
import path from "node:path";
import { ApiError, atomicJson } from "./store.mjs";

const MAX_PRESETS = 50;
const MAX_FILE_BYTES = 8 * 1024 * 1024;

function presetName(value) {
  if (
    typeof value !== "string" ||
    /[\u0000-\u001f\u007f-\u009f\ud800-\udfff]/u.test(value)
  )
    throw new ApiError(
      400,
      "El nombre debe tener entre 1 y 80 caracteres en una sola línea.",
    );
  const name = value.trim().normalize("NFC");
  if (!name || name.length > 80)
    throw new ApiError(
      400,
      "El nombre debe tener entre 1 y 80 caracteres en una sola línea.",
    );
  return name;
}

function requestName(body, keys) {
  if (
    !body ||
    typeof body !== "object" ||
    Array.isArray(body) ||
    Object.keys(body).length !== keys.length ||
    !keys.every((key) => Object.hasOwn(body, key))
  )
    throw new ApiError(
      400,
      "Indica el preset y los parámetros que requiere la operación.",
    );
  return presetName(body.name);
}

export function createPresets({ configPath, store }) {
  // Names are values inside a list, never filenames or object property keys.
  const filename = path.join(path.dirname(configPath), "presets.json");
  let queue = Promise.resolve();
  function serialized(fn) {
    const result = queue.then(fn);
    queue = result.catch(() => {});
    return result;
  }
  async function readPresets() {
    let handle;
    try {
      handle = await open(filename, constants.O_RDONLY | constants.O_NOFOLLOW);
      const info = await handle.stat();
      if (!info.isFile() || info.size > MAX_FILE_BYTES)
        throw new Error("Invalid preset file");
      const raw = await handle.readFile("utf8");
      if (Buffer.byteLength(raw, "utf8") > MAX_FILE_BYTES)
        throw new Error("Preset file too large");
      const document = JSON.parse(raw);
      if (
        !document ||
        typeof document !== "object" ||
        Array.isArray(document) ||
        document.version !== 1 ||
        !Array.isArray(document.presets) ||
        Object.keys(document).length !== 2 ||
        document.presets.length > MAX_PRESETS
      )
        throw new Error("Invalid preset document");
      const names = new Set();
      return document.presets.map((entry) => {
        if (
          !entry ||
          typeof entry !== "object" ||
          Array.isArray(entry) ||
          Object.keys(entry).length !== 3 ||
          !["name", "parameters", "updated_at"].every((key) =>
            Object.hasOwn(entry, key),
          ) ||
          typeof entry.updated_at !== "string" ||
          !Number.isFinite(Date.parse(entry.updated_at))
        )
          throw new Error("Invalid preset record");
        const name = presetName(entry.name);
        if (names.has(name)) throw new Error("Duplicate preset names");
        names.add(name);
        return {
          name,
          parameters: store.validateParameters(entry.parameters),
          updated_at: entry.updated_at,
        };
      });
    } catch (error) {
      if (error.code === "ENOENT") return [];
      throw new ApiError(
        503,
        "Los presets guardados no son válidos o no se pueden leer. Se ha conservado el archivo para revisarlo.",
      );
    } finally {
      await handle?.close();
    }
  }
  async function list() {
    return serialized(async () => ({ presets: await readPresets() }));
  }
  async function save(body) {
    const name = requestName(body, ["name", "parameters"]);
    const parameters = store.validateParameters(body.parameters);
    return serialized(async () => {
      const presets = await readPresets();
      const index = presets.findIndex((entry) => entry.name === name);
      if (index < 0 && presets.length >= MAX_PRESETS)
        throw new ApiError(
          400,
          "Se pueden guardar hasta 50 presets. Elimina uno antes de añadir otro.",
        );
      const preset = { name, parameters, updated_at: new Date().toISOString() };
      if (index < 0) presets.push(preset);
      else presets[index] = preset;
      await atomicJson(filename, { version: 1, presets });
      return {
        preset: structuredClone(preset),
        presets: structuredClone(presets),
        overwritten: index >= 0,
      };
    });
  }
  async function remove(body) {
    const name = requestName(body, ["name"]);
    return serialized(async () => {
      const saved = await readPresets();
      const presets = saved.filter((entry) => entry.name !== name);
      if (presets.length === saved.length)
        throw new ApiError(404, "El preset ya no está disponible.");
      await atomicJson(filename, { version: 1, presets });
      return { presets };
    });
  }
  async function apply(body) {
    const name = requestName(body, ["name"]);
    return serialized(async () => {
      const preset = (await readPresets()).find((entry) => entry.name === name);
      if (!preset) throw new ApiError(404, "El preset ya no está disponible.");
      // Merge into the latest queued configuration, never a stale selection
      // captured while saving the preset or before a concurrent model change.
      return store.update((current) => ({
        ...current,
        ...structuredClone(preset.parameters),
      }));
    });
  }
  return { list, save, delete: remove, apply };
}
