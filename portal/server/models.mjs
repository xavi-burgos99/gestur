import { preflightModel } from "./model-check.mjs";
import yauzl from "yauzl";
import { mkdir, writeFile, readFile, rename, rm, stat } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { ApiError, validateGlb, atomicJson, resourceUris } from "./store.mjs";
const MAX_EXPANDED = 250 * 1024 * 1024;
const MAX_FILES = 500;
const fail = (message) => new ApiError(400, message);
function safeName(name) {
  if (
    !name ||
    name.length > 200 ||
    name.includes("\\") ||
    name.startsWith("/") ||
    /[\x00-\x1f\x7f:]/.test(name) ||
    name.split("/").some((s) => s === ".." || s === ".")
  )
    throw fail("El ZIP contiene una ruta no permitida.");
  if (!/^[A-Za-z0-9_ .\-/]+$/.test(name))
    throw fail(
      "Usa nombres de archivo con letras sin acentos, números, espacios, guiones y puntos.",
    );
  return name;
}
async function unzip(buffer, destination) {
  const zip = await new Promise((resolve, reject) =>
    yauzl.fromBuffer(
      buffer,
      { lazyEntries: true, strictFileNames: true },
      (error, value) =>
        error ? reject(fail("No se ha podido abrir el ZIP.")) : resolve(value),
    ),
  );
  const files = new Set();
  let total = 0;
  let count = 0;
  await new Promise((resolve, reject) => {
    const stop = (error) => {
      zip.close();
      reject(error instanceof ApiError ? error : fail("El ZIP está dañado."));
    };
    zip.on("error", stop);
    zip.on("end", resolve);
    zip.on("entry", (entry) => {
      (async () => {
        const name = safeName(entry.fileName);
        if (++count > MAX_FILES || files.has(name.toLowerCase()))
          throw fail("El ZIP tiene demasiados archivos o nombres duplicados.");
        const type = (entry.externalFileAttributes >>> 16) & 0xf000;
        if (
          ![0, 0x8000, 0x4000].includes(type) ||
          entry.generalPurposeBitFlag & 1
        )
          throw fail(
            "No se admiten enlaces, archivos especiales ni ZIP cifrados.",
          );
        total += entry.uncompressedSize;
        if (
          total > MAX_EXPANDED ||
          entry.uncompressedSize >
            Math.max(1024 * 1024, entry.compressedSize * 100)
        )
          throw fail("El ZIP supera los límites de descompresión.");
        if (name.endsWith("/")) {
          await mkdir(path.join(destination, name), {
            recursive: true,
            mode: 0o2770,
          });
          zip.readEntry();
          return;
        }
        files.add(name.toLowerCase());
        await mkdir(path.dirname(path.join(destination, name)), {
          recursive: true,
          mode: 0o2770,
        });
        const stream = await new Promise((res, rej) =>
          zip.openReadStream(entry, (err, value) =>
            err ? rej(err) : res(value),
          ),
        );
        const chunks = [];
        let bytes = 0;
        for await (const chunk of stream) {
          bytes += chunk.length;
          if (bytes > entry.uncompressedSize || bytes > MAX_EXPANDED)
            throw fail("El ZIP supera los límites de descompresión.");
          chunks.push(chunk);
        }
        if (bytes !== entry.uncompressedSize)
          throw fail("El ZIP está incompleto.");
        await writeFile(path.join(destination, name), Buffer.concat(chunks), {
          flag: "wx",
          mode: 0o660,
        });
        zip.readEntry();
      })().catch(stop);
    });
    zip.readEntry();
  });
  return files;
}
async function resolveReference(root, source, uri) {
  if (
    typeof uri !== "string" ||
    !uri ||
    /[\x00-\x1f\x7f\\]/.test(uri) ||
    uri.startsWith("/") ||
    /^[a-z][a-z\d+.-]*:/i.test(uri)
  )
    throw fail("El modelo contiene una referencia externa o no permitida.");
  let decoded;
  try {
    decoded = decodeURIComponent(uri);
  } catch {
    throw fail("Referencia no válida.");
  }
  const target = path.resolve(path.dirname(path.join(root, source)), decoded);
  if (!target.startsWith(root + path.sep))
    throw fail("El modelo intenta acceder fuera del paquete.");
  try {
    if (!(await stat(target)).isFile()) throw new Error();
  } catch {
    throw fail(`Falta un archivo del modelo: ${uri}`);
  }
  return path.relative(root, target);
}
async function validateEntrypoint(root, entrypoint) {
  if (entrypoint.endsWith(".glb")) {
    validateGlb(await readFile(path.join(root, entrypoint)));
    return;
  }
  if (entrypoint.endsWith(".gltf")) {
    let doc;
    try {
      doc = JSON.parse(await readFile(path.join(root, entrypoint), "utf8"));
    } catch {
      throw fail("El glTF no contiene JSON válido.");
    }
    if (doc.asset?.version !== "2.0") throw fail("Solo se admite glTF 2.0.");
    for (const uri of resourceUris(doc))
      if (!uri.startsWith("data:"))
        await resolveReference(root, entrypoint, uri);
    return;
  }
  const obj = await readFile(path.join(root, entrypoint), "utf8");
  if (
    !/^[ \t]*v[ \t]+[-+.\d]/m.test(obj) ||
    !/^[ \t]*f[ \t]+[+-]?\d/m.test(obj)
  )
    throw fail("El OBJ debe contener vértices y caras.");
  for (const match of obj.matchAll(/^[ \t]*mtllib[ \t]+([^\r\n]+)$/gm)) {
    // One material filename per directive allows spaces in exported filenames.
    const mtl = await resolveReference(root, entrypoint, match[1].trim());
    const materials = await readFile(path.join(root, mtl), "utf8");
    for (const texture of materials.matchAll(
      /^[ \t]*(?:map_\w+|bump|disp|decal|norm|refl)[ \t]+([^\r\n]+)$/gim,
    )) {
      let value = texture[1].trim();
      // Common Wavefront options; reject unknown options instead of guessing a path.
      while (value.startsWith("-")) {
        const option = value.match(
          /^-(?:blendu|blendv|cc|clamp|imfchan|type|texres|bm|boost)\s+\S+\s+|^-(?:mm)\s+\S+\s+\S+\s+|^-(?:o|s|t)\s+[-+.\d]+(?:\s+[-+.\d]+){0,2}\s+/,
        );
        if (!option)
          throw fail(
            "Una textura MTL usa opciones no admitidas. Exporta el material con rutas simples.",
          );
        value = value.slice(option[0].length);
      }
      await resolveReference(root, mtl, value);
    }
  }
}
export async function importModel(
  buffer,
  filename,
  modelsDir,
  checkModel = preflightModel,
) {
  if (!filename?.toLowerCase().endsWith(".zip"))
    throw fail(
      "Sube un ZIP con un único modelo OBJ, glTF o GLB y todos sus recursos.",
    );
  const uuid = randomUUID();
  const staging = path.join(modelsDir, `.upload-${uuid}`);
  const target = path.join(modelsDir, uuid);
  await mkdir(staging, { mode: 0o2770 });
  try {
    const files = await unzip(buffer, staging);
    // Keep original case by walking actual names, never resolve case-folded names.
    const { readdir } = await import("node:fs/promises");
    const all = await readdir(staging, { recursive: true });
    const candidates = all.filter(
      (n) => !n.startsWith("__MACOSX/") && /\.(obj|gltf|glb)$/.test(n),
    );
    if (candidates.length !== 1)
      throw fail(
        "El ZIP debe contener exactamente un modelo .obj, .gltf o .glb.",
      );
    const entrypoint = candidates[0];
    if (
      !/^[A-Za-z0-9][A-Za-z0-9_. -]*(\/[A-Za-z0-9][A-Za-z0-9_. -]*)*\.(obj|gltf|glb)$/.test(
        entrypoint,
      )
    )
      throw fail(
        "Las carpetas y el modelo deben empezar por una letra o un número.",
      );
    await validateEntrypoint(staging, entrypoint);
    await checkModel(path.join(staging, entrypoint));
    const name =
      filename.replace(/\.zip$/i, "").slice(0, 80) || "Modelo importado";
    await atomicJson(path.join(staging, ".gestur-model.json"), {
      entrypoint,
      name,
      size: buffer.length,
      files: files.size,
    });
    await rename(staging, target);
    return {
      id: `${uuid}/${entrypoint}`,
      name,
      format: path.extname(entrypoint).slice(1).toUpperCase(),
      builtin: false,
      size: buffer.length,
    };
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}
