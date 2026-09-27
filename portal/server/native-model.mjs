import { spawn } from "node:child_process";
import {
  access,
  readdir,
  readFile,
  writeFile,
  mkdir,
  realpath,
  lstat,
} from "node:fs/promises";
import path from "node:path";
import { ApiError, repository, validateGlb } from "./store.mjs";
import { bundleGltf } from "./package-model.mjs";

const exists = async (filename) => {
  try {
    await access(filename);
    return true;
  } catch {
    return false;
  }
};
export async function sandboxCommand(
  command,
  args,
  workspace,
  platform = process.platform,
) {
  if (platform === "linux") {
    const mounts = [];
    for (const directory of ["/usr", "/lib", "/lib64", "/bin", "/sbin"])
      if (await exists(directory))
        mounts.push("--ro-bind", directory, directory);
    return {
      // The HTTP service inherits CAP_NET_BIND_SERVICE. Non-setuid bwrap
      // rejects any effective capabilities before parsing --cap-drop, so
      // clear inherited/ambient caps before exec. Node and sudo stay intact.
      command: "/usr/bin/setpriv",
      args: [
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--",
        "/usr/bin/bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        ...mounts,
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/etc",
        "--ro-bind-try",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--ro-bind-try",
        "/etc/fonts",
        "/etc/fonts",
        "--bind",
        workspace,
        "/work",
        "--ro-bind",
        path.join(repository, "portal/server/model-worker.mjs"),
        "/worker/model-worker.mjs",
        "--ro-bind",
        path.join(repository, "portal/node_modules"),
        "/worker/node_modules",
        "--ro-bind",
        process.execPath,
        "/node",
        "--chdir",
        "/work",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--setenv",
        "OMP_NUM_THREADS",
        "2",
        "--",
        "/usr/bin/nice",
        "-n",
        "10",
        "/usr/bin/prlimit",
        ...(command === process.execPath ? [] : ["--as=3221225472"]),
        "--cpu=540",
        "--fsize=536870912",
        "--",
        command === process.execPath ? "/node" : command,
        ...args.map((arg) =>
          arg.startsWith(workspace + path.sep)
            ? "/work/" + path.relative(workspace, arg)
            : arg === path.join(repository, "portal/server/model-worker.mjs")
              ? "/worker/model-worker.mjs"
              : arg,
        ),
      ],
    };
  }
  if (platform === "darwin") {
    const resolvedWorkspace = await realpath(workspace);
    const quoted = (value) => JSON.stringify(value);
    const profile = `(version 1)(deny default)(allow process*)(allow sysctl-read)(allow mach-lookup)(allow file-read-metadata)(allow file-read* (literal "/") (subpath "/System") (subpath "/usr") (subpath "/private/var/db/dyld") (subpath "/opt/homebrew") (literal ${quoted(process.execPath)}) (literal ${quoted(path.join(repository, "portal/server/model-worker.mjs"))}) (subpath ${quoted(path.join(repository, "portal/node_modules"))}) (subpath ${quoted(resolvedWorkspace)}))(allow file-write* (subpath ${quoted(resolvedWorkspace)}))(allow file-read* file-write* (literal "/dev/null") (literal "/dev/urandom"))`;
    return {
      command: "/usr/bin/sandbox-exec",
      args: ["-p", profile, command, ...args],
    };
  }
  throw new ApiError(
    503,
    "La importación de modelos no está disponible en este sistema.",
  );
}
export async function runNative(
  command,
  args,
  workspace,
  { signal, timeout = 600000 } = {},
) {
  if (signal?.aborted)
    throw new ApiError(503, "La importación se ha interrumpido.");
  const invocation = await sandboxCommand(command, args, workspace);
  return new Promise((resolve, reject) => {
    const child = spawn(invocation.command, invocation.args, {
      cwd: workspace,
      env: {
        PATH: process.env.PATH || "/usr/bin:/bin",
        HOME: workspace,
        TMPDIR: workspace,
        OMP_NUM_THREADS: "2",
        OPENBLAS_NUM_THREADS: "2",
        LANG: "C.UTF-8",
      },
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
    });
    let output = "",
      timedOut = false;
    const kill = () => {
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {
        child.kill("SIGKILL");
      }
    };
    const timer = setTimeout(() => {
      timedOut = true;
      kill();
    }, timeout);
    signal?.addEventListener("abort", kill, { once: true });
    const consume = (chunk) => {
      if (output.length < 65536)
        output += chunk.toString().slice(0, 65536 - output.length);
    };
    child.stdout.on("data", consume);
    child.stderr.on("data", consume);
    child.on("error", (error) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", kill);
      reject(
        new ApiError(
          503,
          "No está instalado el conversor 3D o su aislamiento. Ejecuta el instalador actualizado.",
        ),
      );
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", kill);
      if (signal?.aborted)
        return reject(
          new ApiError(
            503,
            "La importación se interrumpió al detener el servicio.",
          ),
        );
      if (timedOut)
        return reject(
          new ApiError(
            422,
            "El modelo tardó más de 10 minutos en procesarse. Exporta una versión más sencilla.",
          ),
        );
      if (code !== 0)
        return reject(
          new ApiError(
            422,
            "El conversor no pudo abrir el modelo. Revisa el formato y que el ZIP incluya todos sus recursos.",
          ),
        );
      resolve(output);
    });
  });
}
const assimpBinary = () =>
  process.env.GESTUR_ASSIMP ||
  (process.platform === "darwin"
    ? "/opt/homebrew/bin/assimp"
    : "/usr/bin/assimp");
const workerArgs = [
  "--max-old-space-size=1024",
  "--disable-wasm-trap-handler",
  "--wasm-max-mem-pages=16384",
  path.join(repository, "portal/server/model-worker.mjs"),
];
export async function convertModel({
  workspace,
  source,
  files,
  warnings,
  signal,
}) {
  const sourceRoot = path.join(workspace, "source");
  const output = path.join(workspace, "converted");
  await mkdir(output, { recursive: true });
  let root = sourceRoot,
    entrypoint = source,
    available = files;
  if (!/\.(gltf|glb)$/i.test(source)) {
    await runNative(
      assimpBinary(),
      [
        "export",
        path.join(sourceRoot, source),
        path.join(output, "model.gltf"),
        "-fgltf2",
        "-tri",
      ],
      workspace,
      { signal },
    );
    root = output;
    entrypoint = "model.gltf";
    available = [];
    for (const name of await readdir(output, { recursive: true })) {
      const info = await lstat(path.join(output, name));
      if (info.isSymbolicLink() || (!info.isFile() && !info.isDirectory()))
        throw new ApiError(422, "El conversor generó un recurso no permitido.");
      if (info.isFile()) available.push(name);
    }
  }
  let imageIndex = 0;
  const convertImage = async (data) => {
    const input = path.join(output, `texture-${imageIndex++}.image`),
      result = `${input}.png`;
    await writeFile(input, data);
    await runNative(
      process.execPath,
      [...workerArgs, "image", input, result],
      workspace,
      { signal },
    );
    if (!(await lstat(result)).isFile())
      throw new ApiError(422, "No se pudo convertir la textura.");
    return readFile(result);
  };
  const buffer = await bundleGltf(
    root,
    entrypoint,
    available,
    warnings,
    sourceRoot,
    files,
    convertImage,
  );
  validateGlb(buffer);
  await writeFile(path.join(workspace, "model.glb"), buffer);
  return { entrypoint: "model.glb" };
}
export async function simplifyModel({
  workspace,
  originalTriangles,
  targetTriangles,
  signal,
}) {
  const original = path.join(workspace, "model.glb"),
    simplified = path.join(workspace, "simplified.glb");
  await runNative(
    process.execPath,
    [
      ...workerArgs,
      "simplify",
      original,
      simplified,
      String(targetTriangles / originalTriangles),
    ],
    workspace,
    { signal },
  );
  if (!(await lstat(simplified)).isFile())
    throw new ApiError(422, "No se pudo guardar el modelo reducido.");
  validateGlb(await readFile(simplified));
  return { entrypoint: "simplified.glb" };
}

export async function verifyImporter(workspace) {
  return runNative(process.execPath, [...workerArgs, "check"], workspace);
}
