import path from "node:path";
import { readFile, writeFile } from "node:fs/promises";
import { ApiError } from "./store.mjs";

export const FORMATS = new Set(
  "glb gltf obj fbx dae stl ply 3ds off x lwo ase dxf ac ms3d cob b3d".split(
    " ",
  ),
);
const fail = (message) => new ApiError(422, message);
const fold = (value) => value.normalize("NFC").toLocaleLowerCase("en-US");
const unix = (value) => value.replaceAll("\\", "/");
const stripComment = (value) => value.replace(/\s+#.*$/, "").trim();
const stripQuotes = (value) => value.replace(/^(["'])(.*)\1$/, "$2");

// Only names in this index can be read. Absolute/obsolete references are labels,
// never host filesystem paths. Ambiguous basenames are never silently guessed.
export function resourceResolver(files, warnings = []) {
  const index = new Map(files.map((name) => [fold(name), name]));
  return (source, raw) => {
    if (typeof raw !== "string" || !raw || /[\x00-\x1f\x7f]/.test(raw))
      throw fail("El modelo contiene una ruta de recurso no válida.");
    let reference = stripQuotes(raw.trim());
    try {
      reference = decodeURIComponent(reference);
    } catch {
      /* Literal % is a valid filename. */
    }
    reference = unix(reference).normalize("NFC");
    if (/^(?:https?|ftp|data|javascript):/i.test(reference))
      throw fail(
        `El recurso ${raw.slice(0, 160)} es externo. Inclúyelo dentro del ZIP.`,
      );
    reference = reference.replace(/^file:\/*/i, "/");
    const relative = path.posix.normalize(
      path.posix.join(path.posix.dirname(source), reference),
    );
    const local =
      !reference.startsWith("/") &&
      !/^[a-z]:/i.test(reference) &&
      relative !== ".." &&
      !relative.startsWith("../")
        ? index.get(fold(relative))
        : null;
    const root = index.get(fold(reference.replace(/^\.\//, "")));
    let selected = local || root;
    if (!selected) {
      const parts = reference
        .split("/")
        .filter((p) => p && p !== "." && p !== ".." && !/^[a-z]:$/i.test(p));
      for (let count = parts.length; count >= 1; count--) {
        const suffix = fold(parts.slice(-count).join("/"));
        const matches = files.filter(
          (name) => fold(name) === suffix || fold(name).endsWith(`/${suffix}`),
        );
        if (matches.length > 1)
          throw fail(
            `La ruta «${raw.slice(0, 160)}» coincide con varios archivos. Pon nombres distintos a esos recursos.`,
          );
        if (matches.length === 1) {
          selected = matches[0];
          break;
        }
      }
    }
    if (!selected)
      throw fail(
        `Falta el recurso «${raw.slice(0, 160)}». Incluye el modelo y sus texturas en un ZIP.`,
      );
    const corrected = path.posix.relative(path.posix.dirname(source), selected);
    if (corrected !== raw && warnings.length < 50)
      warnings.push(
        `Ruta reparada: ${raw.slice(0, 100)} → ${corrected.slice(0, 100)}`,
      );
    return selected;
  };
}

function mtlTexture(value) {
  let prefix = "";
  while (value.startsWith("-")) {
    const option = value.match(
      /^-(?:blendu|blendv|cc|clamp|imfchan|type|texres|bm|boost|colorspace)\s+\S+\s+|^-mm\s+\S+\s+\S+\s+|^-(?:o|s|t)\s+[-+.\d]+(?:\s+[-+.\d]+){0,2}\s+/i,
    );
    if (!option)
      throw fail(
        "Una textura MTL usa una opción desconocida. Exporta materiales Wavefront estándar.",
      );
    prefix += option[0];
    value = value.slice(option[0].length);
  }
  return { prefix, value };
}
export async function repairTextResources(root, entrypoint, files, warnings) {
  const resolve = resourceResolver(files, warnings);
  const relative = (source, raw) =>
    path.posix.relative(path.posix.dirname(source), resolve(source, raw));
  const extension = path.extname(entrypoint).toLowerCase();
  if (extension === ".obj") {
    let obj = await readFile(path.join(root, entrypoint), "utf8");
    const materials = new Set();
    obj = obj.replace(
      /^([ \t]*mtllib[ \t]+)([^\r\n]+)$/gm,
      (_, prefix, raw) => {
        const libraries = stripComment(raw);
        // Prefer a whole filename (spaces allowed); fall back to Wavefront's list.
        let names;
        try {
          names = [resolve(entrypoint, libraries)];
        } catch (error) {
          if (!libraries.includes(" ")) throw error;
          names = libraries
            .match(/"[^"]+"|'[^']+'|\S+/g)
            .map((name) => resolve(entrypoint, name));
        }
        for (const name of names) materials.add(name);
        return names
          .map(
            (name) =>
              `${prefix}${path.posix.relative(path.posix.dirname(entrypoint), name)}`,
          )
          .join("\n");
      },
    );
    // OBJ call/csh directives may spawn/include outside resources in other tools.
    if (/^[ \t]*(?:call|csh)\s/im.test(obj))
      throw fail("El OBJ contiene instrucciones externas no admitidas.");
    await writeFile(path.join(root, entrypoint), obj);
    for (const material of materials) {
      const mtl = await readFile(path.join(root, material), "utf8");
      const fixed = mtl.replace(
        /^([ \t]*(?:map_\w+|bump|disp|decal|norm|refl)[ \t]+)([^\r\n]+)$/gim,
        (_, directive, raw) => {
          const { prefix, value } = mtlTexture(stripComment(raw));
          return `${directive}${prefix}${relative(material, value)}`;
        },
      );
      await writeFile(path.join(root, material), fixed);
    }
  } else if (extension === ".dae") {
    let document = await readFile(path.join(root, entrypoint), "utf8");
    if (/<!DOCTYPE|<!ENTITY/i.test(document))
      throw fail("El Collada contiene entidades XML externas no admitidas.");
    // In COLLADA image init_from is the filename; surface init_from is an image ID.
    document = document.replace(
      /(<image\b[^>]*>)([\s\S]*?)(<\/image>)/gi,
      (_, open, body, close) =>
        open +
        body.replace(
          /(<init_from\b[^>]*>)([^<]+)(<\/init_from>)/gi,
          (_, start, raw, end) => {
            const decoded = raw
              .trim()
              .replaceAll("&amp;", "&")
              .replaceAll("&quot;", '"')
              .replaceAll("&apos;", "'")
              .replaceAll("&lt;", "<")
              .replaceAll("&gt;", ">");
            return (
              start +
              relative(entrypoint, decoded)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;") +
              end
            );
          },
        ) +
        close,
    );
    await writeFile(path.join(root, entrypoint), document);
  }
}

export function readGlb(buffer) {
  if (
    buffer.length < 20 ||
    buffer.readUInt32LE(0) !== 0x46546c67 ||
    buffer.readUInt32LE(4) !== 2 ||
    buffer.readUInt32LE(8) !== buffer.length
  )
    throw fail("El archivo no es un GLB 2.0 válido.");
  const chunks = [];
  let offset = 12;
  while (offset < buffer.length) {
    if (offset + 8 > buffer.length) throw fail("El GLB está incompleto.");
    const length = buffer.readUInt32LE(offset),
      type = buffer.readUInt32LE(offset + 4);
    if (length % 4 || offset + 8 + length > buffer.length)
      throw fail("El GLB está incompleto.");
    chunks.push({
      type,
      data: buffer.subarray(offset + 8, offset + 8 + length),
    });
    offset += length + 8;
  }
  if (
    chunks[0]?.type !== 0x4e4f534a ||
    chunks.filter((c) => c.type === 0x4e4f534a).length !== 1 ||
    chunks.filter((c) => c.type === 0x004e4942).length > 1
  )
    throw fail("La estructura GLB no es válida.");
  try {
    return {
      doc: JSON.parse(chunks[0].data.toString("utf8").trim()),
      bin: chunks.find((c) => c.type === 0x004e4942)?.data,
    };
  } catch {
    throw fail("El JSON del GLB no es válido.");
  }
}
function imageMime(data) {
  if (
    data.length > 8 &&
    data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
  )
    return "image/png";
  if (data.length > 3 && data[0] === 255 && data[1] === 216 && data[2] === 255)
    return "image/jpeg";
  if (
    data.length > 12 &&
    data.toString("ascii", 0, 4) === "RIFF" &&
    data.toString("ascii", 8, 12) === "WEBP"
  )
    return "image/webp";
  return null;
}
const MAX_OUTPUT = 512 * 1024 * 1024;
export async function bundleGltf(
  root,
  entrypoint,
  files,
  warnings = [],
  fallbackRoot = root,
  fallbackFiles = files,
  convertImage,
) {
  let doc, bin;
  if (entrypoint.toLowerCase().endsWith(".glb"))
    ({ doc, bin } = readGlb(await readFile(path.join(root, entrypoint))));
  else {
    try {
      doc = JSON.parse(await readFile(path.join(root, entrypoint), "utf8"));
    } catch {
      throw fail("El glTF no contiene JSON válido.");
    }
  }
  if (doc.asset?.version !== "2.0") throw fail("Solo se admite glTF 2.0.");
  const resolve = resourceResolver(files, warnings),
    fallback = resourceResolver(fallbackFiles, warnings);
  const loadUri = async (uri, image = false) => {
    if (typeof uri !== "string") throw fail("URI no válida en glTF.");
    if (uri.startsWith("data:")) {
      const match = uri.match(/^data:[\w.+/-]+;base64,([a-z\d+/=\s]*)$/i);
      if (!match) throw fail("El recurso incrustado no usa base64 válido.");
      const data = Buffer.from(match[1], "base64");
      if (data.length > MAX_OUTPUT)
        throw fail("Un recurso es demasiado grande.");
      return data;
    }
    let filename;
    try {
      filename = path.join(root, resolve(entrypoint, uri));
    } catch (error) {
      // Exporters can retain the input's absolute/Windows texture path.
      if (
        !image ||
        fallbackRoot === root ||
        !error.message.startsWith("Falta el recurso")
      )
        throw error;
      filename = path.join(fallbackRoot, fallback("model.gltf", uri));
    }
    return readFile(filename);
  };
  const buffers = doc.buffers || [],
    chunks = [],
    offsets = [];
  let total = 0;
  const append = (bytes) => {
    const padding = (4 - (bytes.length % 4)) % 4;
    if (total + bytes.length + padding > MAX_OUTPUT)
      throw fail("El modelo convertido supera 512 MB.");
    const start = total;
    chunks.push(bytes, Buffer.alloc(padding));
    total += bytes.length + padding;
    return start;
  };
  for (let i = 0; i < buffers.length; i++) {
    const spec = buffers[i],
      data = spec.uri ? await loadUri(spec.uri) : i === 0 ? bin : null;
    if (
      !data ||
      !Number.isSafeInteger(spec.byteLength) ||
      spec.byteLength < 0 ||
      data.length < spec.byteLength
    )
      throw fail("Faltan datos de geometría del modelo.");
    offsets.push(append(data.subarray(0, spec.byteLength)));
  }
  for (const view of doc.bufferViews || []) {
    const length = buffers[view.buffer]?.byteLength,
      start = view.byteOffset || 0;
    if (
      !Number.isSafeInteger(length) ||
      !Number.isSafeInteger(start) ||
      start < 0 ||
      !Number.isSafeInteger(view.byteLength) ||
      view.byteLength < 0 ||
      start + view.byteLength > length
    )
      throw fail("Una sección de geometría está fuera del archivo.");
    view.byteOffset = offsets[view.buffer] + start;
    view.buffer = 0;
  }
  for (const image of doc.images || []) {
    if (!image.uri) continue;
    let data = await loadUri(image.uri, true),
      mime = imageMime(data);
    // Core glTF/Panda compatibility uses PNG/JPEG. WebP is decoded losslessly
    // into PNG instead of relying on optional viewer codecs/extensions.
    if ((!mime || mime === "image/webp") && convertImage) {
      data = await convertImage(data);
      mime = imageMime(data);
    }
    if (!mime)
      throw fail(
        `La textura «${String(image.uri).slice(0, 100)}» no se puede convertir a PNG o JPEG.`,
      );
    doc.bufferViews ||= [];
    image.bufferView = doc.bufferViews.length;
    doc.bufferViews.push({
      buffer: 0,
      byteOffset: append(data),
      byteLength: data.length,
    });
    image.mimeType = mime;
    delete image.uri;
  }
  // Panda's glTF loader cannot reliably compute normals on strip/fan input.
  // Expand only the index topology; vertex attributes, UVs and materials stay exact.
  const needsTriangles = (doc.meshes || []).some((mesh) =>
    (mesh.primitives || []).some((primitive) =>
      [5, 6].includes(primitive.mode),
    ),
  );
  const geometry = needsTriangles ? Buffer.concat(chunks, total) : null;
  for (const mesh of doc.meshes || [])
    for (const primitive of mesh.primitives || []) {
      if (![5, 6].includes(primitive.mode)) continue;
      const accessor =
        primitive.indices === undefined
          ? doc.accessors?.[primitive.attributes?.POSITION]
          : doc.accessors?.[primitive.indices];
      const count = accessor?.count;
      if (
        !Number.isSafeInteger(count) ||
        count < 3 ||
        count > MAX_OUTPUT / 12 ||
        accessor.sparse
      )
        throw fail(
          "La tira de triángulos no contiene índices compatibles. Exporta el modelo con triángulos.",
        );
      let readIndex = (index) => index;
      if (primitive.indices !== undefined) {
        const view = doc.bufferViews?.[accessor.bufferView];
        const width = { 5121: 1, 5123: 2, 5125: 4 }[accessor.componentType];
        const start = (view?.byteOffset || 0) + (accessor.byteOffset || 0);
        const stride = view?.byteStride || width;
        if (
          !view ||
          !width ||
          accessor.type !== "SCALAR" ||
          !Number.isSafeInteger(start) ||
          start < 0 ||
          start + (count - 1) * stride + width >
            view.byteOffset + view.byteLength
        )
          throw fail("Los índices de la tira de triángulos no son válidos.");
        readIndex = (index) =>
          geometry.readUIntLE(start + index * stride, width);
      }
      const indices = Buffer.alloc((count - 2) * 12);
      let used = 0;
      for (let i = 2; i < count; i++) {
        const a = readIndex(primitive.mode === 6 ? 0 : i - 2),
          b = readIndex(i - 1),
          c = readIndex(i);
        if (a === b || a === c || b === c) continue;
        const triangle = primitive.mode === 5 && i % 2 ? [b, a, c] : [a, b, c];
        for (const index of triangle) {
          indices.writeUInt32LE(index, used);
          used += 4;
        }
      }
      doc.bufferViews ||= [];
      doc.accessors ||= [];
      const viewIndex = doc.bufferViews.length;
      doc.bufferViews.push({
        buffer: 0,
        byteOffset: append(indices.subarray(0, used)),
        byteLength: used,
        target: 34963,
      });
      primitive.indices = doc.accessors.length;
      doc.accessors.push({
        bufferView: viewIndex,
        componentType: 5125,
        count: used / 4,
        type: "SCALAR",
      });
      primitive.mode = 4;
    }
  // URI-bearing extensions can fetch resources too. Only core buffers/images are
  // repackaged; reject unknown references rather than publish a partial model.
  const pending = [doc];
  while (pending.length) {
    const value = pending.pop();
    for (const [key, child] of Object.entries(value)) {
      if (
        key === "uri" &&
        typeof child === "string" &&
        !child.startsWith("data:")
      ) {
        if (buffers.some((b) => b === value)) continue;
        throw fail(
          "Una extensión glTF necesita recursos externos no admitidos. Exporta como GLB estándar.",
        );
      }
      if (child && typeof child === "object") pending.push(child);
    }
  }
  doc.buffers = [{ byteLength: total }];
  const jsonBytes = Buffer.from(JSON.stringify(doc));
  const json = Buffer.concat([
    jsonBytes,
    Buffer.alloc((4 - (jsonBytes.length % 4)) % 4, 32),
  ]);
  const binary = Buffer.concat(chunks, total);
  const result = Buffer.alloc(28 + json.length + binary.length);
  result.writeUInt32LE(0x46546c67);
  result.writeUInt32LE(2, 4);
  result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(json.length, 12);
  result.writeUInt32LE(0x4e4f534a, 16);
  json.copy(result, 20);
  const offset = 20 + json.length;
  result.writeUInt32LE(binary.length, offset);
  result.writeUInt32LE(0x004e4942, offset + 4);
  binary.copy(result, offset + 8);
  return result;
}
