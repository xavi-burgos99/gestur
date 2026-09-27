import yauzl from "yauzl";
import {
  mkdir,
  writeFile,
  readFile,
  rename,
  rm,
  readdir,
  stat,
} from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { ApiError, atomicJson, validateGlb } from "./store.mjs";
import { FORMATS, repairTextResources } from "./package-model.mjs";
import { convertModel, simplifyModel } from "./native-model.mjs";
import { preflightModel } from "./model-check.mjs";

export const HIGH_TRIANGLES = 1000000;
export const TARGET_TRIANGLES = 500000;
const MAX_EXPANDED = 250 * 1024 * 1024;
const MAX_FILES = 500;
const fail = (message) => new ApiError(400, message);
function safeName(raw) {
  const name = raw.replaceAll("\\", "/").normalize("NFC");
  if (
    !name ||
    name.length > 240 ||
    name.startsWith("/") ||
    /[\x00-\x1f\x7f:]/.test(name) ||
    name
      .split("/")
      .some(
        (part) =>
          part === ".." || part === "." || (!part && !name.endsWith("/")),
      )
  )
    throw fail("El paquete contiene una ruta no permitida.");
  return name;
}
export async function extractUpload(buffer, filename, destination) {
  await mkdir(destination, { recursive: true, mode: 0o2770 });
  if (!filename?.toLowerCase().endsWith(".zip")) {
    const name = safeName(
      path.posix.basename(filename?.replaceAll("\\", "/") || ""),
    );
    if (!FORMATS.has(path.extname(name).slice(1).toLowerCase()))
      throw fail(
        "Formato no admitido. Sube GLB, glTF, OBJ, FBX, STL, PLY, DAE, 3DS u otro formato compatible, o un ZIP con sus texturas.",
      );
    await writeFile(path.join(destination, name), buffer, {
      flag: "wx",
      mode: 0o660,
    });
    return { entrypoint: name, files: [name] };
  }
  const zip = await new Promise((resolve, reject) =>
    yauzl.fromBuffer(
      buffer,
      { lazyEntries: true, strictFileNames: false },
      (error, value) =>
        error ? reject(fail("No se ha podido abrir el ZIP.")) : resolve(value),
    ),
  );
  const names = new Set(),
    files = [];
  let total = 0,
    count = 0;
  await new Promise((resolve, reject) => {
    let stopped = false;
    const stop = (error) => {
      if (stopped) return;
      stopped = true;
      zip.close();
      reject(error instanceof ApiError ? error : fail("El ZIP está dañado."));
    };
    zip.on("error", stop);
    zip.on("end", resolve);
    zip.on("entry", (entry) => {
      (async () => {
        const name = safeName(entry.fileName),
          key = name.toLocaleLowerCase("en-US");
        if (++count > MAX_FILES || names.has(key))
          throw fail("El ZIP tiene demasiados archivos o nombres duplicados.");
        names.add(key);
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
        await mkdir(path.dirname(path.join(destination, name)), {
          recursive: true,
          mode: 0o2770,
        });
        const stream = await new Promise((res, rej) =>
          zip.openReadStream(entry, (error, value) =>
            error ? rej(error) : res(value),
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
        files.push(name);
        zip.readEntry();
      })().catch(stop);
    });
    zip.readEntry();
  });
  const candidates = files.filter(
    (name) =>
      !name
        .split("/")
        .some((part) => part.startsWith(".") || part === "__MACOSX") &&
      FORMATS.has(path.extname(name).slice(1).toLowerCase()),
  );
  if (candidates.length !== 1)
    throw fail(
      "El ZIP debe contener un único modelo principal, junto con sus materiales y texturas.",
    );
  return { entrypoint: candidates[0], files };
}

export async function createImporter({
  modelsDir,
  modelChecker = preflightModel,
  modelConverter = convertModel,
  modelSimplifier = simplifyModel,
  onPublished = async () => {},
  persistJob = atomicJson,
}) {
  await mkdir(modelsDir, { recursive: true, mode: 0o2770 });
  const stateFile = path.join(modelsDir, ".import-job.json");
  let job = null,
    running = null,
    closed = false;
  const abort = new AbortController();
  const workspaceFor = (id) => path.join(modelsDir, `.import-${id}`);
  const snapshot = () => {
    if (!job) return null;
    const visible = structuredClone(job);
    // Persisting the final result (or cleaning up a failure) is still work.
    // Exposing a terminal/decision state earlier lets clients immediately ask
    // for the next action while start/decide and model mutations remain locked.
    if (
      running &&
      ["completed", "failed", "awaiting_decision"].includes(visible.state)
    ) {
      visible.state = "processing";
      visible.stage = "finishing";
      visible.message = "Finalizando la importación…";
      visible.model = null;
      delete visible.error;
    }
    return visible;
  };
  const persist = async () => persistJob(stateFile, job);
  try {
    job = JSON.parse(await readFile(stateFile, "utf8"));
  } catch {}
  if (job && !/^[a-f0-9-]{36}$/.test(job.id)) job = null;
  if (job?.state === "processing") {
    // A final rename may have completed just before the service stopped.
    try {
      const metadata = JSON.parse(
        await readFile(
          path.join(modelsDir, job.id, ".gestur-model.json"),
          "utf8",
        ),
      );
      job = {
        ...job,
        state: "completed",
        stage: "completed",
        message: "Modelo importado.",
        model: publicModel(job.id, metadata),
      };
    } catch {
      job = {
        ...job,
        state: "failed",
        stage: "failed",
        message:
          "La importación se interrumpió al reiniciar. Vuelve a subir el archivo.",
        error:
          "La importación se interrumpió al reiniciar. Vuelve a subir el archivo.",
      };
      await rm(workspaceFor(job.id), { recursive: true, force: true });
    }
    await persist();
  }
  if (job?.state === "awaiting_decision") {
    try {
      validateGlb(await readFile(path.join(workspaceFor(job.id), "model.glb")));
    } catch {
      job = {
        ...job,
        state: "failed",
        stage: "failed",
        error:
          "La importación pendiente ya no está disponible. Vuelve a subir el archivo.",
        message: "Vuelve a subir el archivo.",
      };
      await rm(workspaceFor(job.id), { recursive: true, force: true });
      await persist();
    }
  }
  for (const name of await readdir(modelsDir))
    if (
      /^\.(?:upload|import)-[a-f0-9-]{36}$/.test(name) &&
      name !== (job?.state === "awaiting_decision" ? `.import-${job.id}` : null)
    )
      await rm(path.join(modelsDir, name), { recursive: true, force: true });
  function publicModel(id, metadata) {
    return {
      id: `${id}/${metadata.entrypoint}`,
      name: metadata.name,
      orientation: metadata.orientation || { x: 0, y: 0, z: 0 },
      format: "GLB",
      builtin: false,
      size: metadata.size,
      triangles: metadata.triangles,
      originalTriangles: metadata.originalTriangles,
      sourceFormat: metadata.sourceFormat,
      simplified: metadata.simplified,
      warnings: metadata.warnings || [],
    };
  }
  async function publish(simplify) {
    const workspace = workspaceFor(job.id);
    let entrypoint = "model.glb",
      triangles = job.proposal?.originalTriangles || job.triangles;
    if (simplify) {
      job.stage = "simplifying";
      job.message = "La Raspberry Pi está reduciendo los polígonos…";
      await persist();
      ({ entrypoint } = await modelSimplifier({
        workspace,
        ...job.proposal,
        signal: abort.signal,
      }));
      if (entrypoint !== "simplified.glb")
        throw new Error("Invalid simplifier output");
      const result = await modelChecker(path.join(workspace, entrypoint));
      triangles = result.triangles ?? result.primitives;
      if (
        !Number.isSafeInteger(triangles) ||
        triangles <= 0 ||
        triangles >= job.proposal.originalTriangles
      )
        throw new ApiError(
          422,
          "No se ha podido reducir este modelo conservando una geometría válida. Vuelve a subirlo y continúa sin simplificar.",
        );
    }
    if (simplify && triangles > job.proposal.targetTriangles * 1.05)
      job.warnings.push(
        `Se han conservado ${triangles.toLocaleString("es-ES")} triángulos para proteger la forma y las uniones de las texturas; el objetivo era ${job.proposal.targetTriangles.toLocaleString("es-ES")}.`,
      );
    validateGlb(await readFile(path.join(workspace, entrypoint)));
    const metadata = {
      entrypoint,
      name: job.name,
      size: (await stat(path.join(workspace, entrypoint))).size,
      uploadSize: job.size,
      files: job.files,
      triangles,
      originalTriangles: job.proposal?.originalTriangles || job.triangles,
      sourceFormat: job.sourceFormat,
      simplified: simplify,
      warnings: job.warnings,
    };
    await atomicJson(path.join(workspace, ".gestur-model.json"), metadata);
    // Keep the upload and the canonical original for later reprocessing. Only
    // finished imports are renamed into the catalog's UUID namespace.
    await rm(path.join(workspace, "converted"), {
      recursive: true,
      force: true,
    });
    await rename(workspace, path.join(modelsDir, job.id));
    await onPublished();
    job = {
      ...job,
      state: "completed",
      stage: "completed",
      message: "Modelo importado. Ya puedes mostrarlo.",
      model: publicModel(job.id, metadata),
    };
    await persist().catch(() => {});
  }
  function work(fn) {
    running = (async () => {
      try {
        await fn();
      } catch (error) {
        job = {
          ...job,
          state: "failed",
          stage: "failed",
          error:
            error instanceof ApiError
              ? error.message
              : "No se ha podido importar el modelo. Revisa sus archivos y vuelve a intentarlo.",
          message: "La importación no se ha completado.",
        };
        await rm(workspaceFor(job.id), { recursive: true, force: true }).catch(
          () => {},
        );
        await persist().catch(() => {});
      }
    })().finally(() => {
      running = null;
    });
  }
  return {
    current: snapshot,
    get(id) {
      if (!job || job.id !== id)
        throw new ApiError(404, "No se encuentra esta importación.");
      return snapshot();
    },
    async start(buffer, filename) {
      if (
        closed ||
        running ||
        ["processing", "awaiting_decision"].includes(job?.state)
      )
        throw new ApiError(
          409,
          "Ya hay una importación en curso. Termínala antes de subir otro modelo.",
        );
      const id = randomUUID();
      // Reserve synchronously before the first await to serialize uploads.
      job = {
        id,
        state: "processing",
        stage: "preparing",
        message: "Preparando el modelo…",
        proposal: null,
        model: null,
        warnings: [],
        name: path.posix
          .basename(filename?.replaceAll("\\", "/") || "Modelo")
          .replace(/\.[^.]+$/, "")
          .slice(0, 80),
        size: buffer.length,
      };
      const accepted = snapshot();
      try {
        if (!buffer.length || buffer.length > 100 * 1024 * 1024)
          throw new ApiError(413, "El archivo está vacío o supera los 100 MB.");
        await mkdir(workspaceFor(id), { mode: 0o2770 });
        await persist();
      } catch (error) {
        job = null;
        await rm(workspaceFor(id), { recursive: true, force: true });
        throw error;
      }
      work(async () => {
        const workspace = workspaceFor(id),
          sourceRoot = path.join(workspace, "source");
        const { entrypoint, files } = await extractUpload(
          buffer,
          filename,
          sourceRoot,
        );
        await writeFile(
          path.join(
            workspace,
            "original-upload" +
              (filename.toLowerCase().endsWith(".zip")
                ? ".zip"
                : path.extname(entrypoint)),
          ),
          buffer,
        );
        job.sourceFormat = path.extname(entrypoint).slice(1).toUpperCase();
        job.files = files.length;
        job.stage = "converting";
        job.message = "Convirtiendo el modelo y reparando sus texturas…";
        await persist();
        await repairTextResources(sourceRoot, entrypoint, files, job.warnings);
        await modelConverter({
          workspace,
          source: entrypoint,
          files,
          warnings: job.warnings,
          signal: abort.signal,
        });
        validateGlb(await readFile(path.join(workspace, "model.glb")));
        job.stage = "checking";
        job.message = "Comprobando geometría y texturas…";
        await persist();
        const result = await modelChecker(path.join(workspace, "model.glb"));
        job.triangles = result.triangles ?? result.primitives;
        if (!Number.isSafeInteger(job.triangles) || job.triangles <= 0)
          throw new ApiError(
            422,
            "El modelo no contiene triángulos visibles válidos.",
          );
        if (job.triangles > HIGH_TRIANGLES) {
          job.proposal = {
            originalTriangles: job.triangles,
            targetTriangles: TARGET_TRIANGLES,
            reductionPercent:
              Math.round((1 - TARGET_TRIANGLES / job.triangles) * 1000) / 10,
          };
          job.state = "awaiting_decision";
          job.stage = "awaiting_decision";
          job.message =
            "Este modelo tiene muchos triángulos. Puedes reducirlos o conservar el original.";
          await persist();
        } else await publish(false);
      });
      return accepted;
    },
    async decide(id, simplify) {
      if (typeof simplify !== "boolean")
        throw fail("Indica si deseas simplificar el modelo.");
      if (
        closed ||
        !job ||
        job.id !== id ||
        job.state !== "awaiting_decision" ||
        running
      )
        throw new ApiError(
          409,
          "Esta importación no está esperando una decisión.",
        );
      const pending = snapshot();
      job.state = "processing";
      job.stage = simplify ? "simplifying" : "finishing";
      job.message = simplify
        ? "Preparando la reducción…"
        : "Conservando la geometría original…";
      try {
        await persist();
      } catch (error) {
        job = pending;
        throw error;
      }
      const accepted = snapshot();
      work(() => publish(simplify));
      return accepted;
    },
    async close() {
      closed = true;
      abort.abort();
      await running;
    },
    async idle() {
      await running;
    },
  };
}
