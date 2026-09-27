import { spawn } from "node:child_process";
import { ApiError } from "./store.mjs";

export function validateHostname(value, allowEmpty = false) {
  if (allowEmpty && (value === undefined || value === "")) return "";
  if (
    typeof value !== "string" ||
    value.length > 63 ||
    !/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(value)
  )
    throw new ApiError(
      400,
      "Usa de 1 a 63 letras minúsculas, números o guiones, sin guiones al principio ni al final.",
    );
  return value;
}

export function validateDeviceRequest(kind, body) {
  const fields = {
    setup: ["hostname", "password", "confirmation"],
    hostname: ["hostname"],
    reset: ["confirmation"],
  }[kind];
  if (
    !body ||
    typeof body !== "object" ||
    Array.isArray(body) ||
    Object.keys(body).some((key) => !fields.includes(key))
  )
    throw new ApiError(400, "Solicitud de configuración no válida.");
  if (kind === "reset") {
    if (body.confirmation !== "BORRAR")
      throw new ApiError(400, "Escribe BORRAR para confirmar.");
    return { confirmation: "BORRAR" };
  }
  const hostname = validateHostname(body.hostname, kind === "setup");
  if (kind === "hostname") return { hostname };
  if (
    typeof body.password !== "string" ||
    !/^[\x20-\x7e]{8,63}$/.test(body.password)
  )
    throw new ApiError(
      400,
      "La contraseña debe tener entre 8 y 63 caracteres ASCII, compatibles con Wi-Fi.",
    );
  if (body.password !== body.confirmation)
    throw new ApiError(400, "Las contraseñas no coinciden.");
  return { hostname, password: body.password, confirmation: body.confirmation };
}

export function systemDevice() {
  return async (action, settings) =>
    new Promise((resolve, reject) => {
      const child = spawn(
        "/usr/bin/sudo",
        ["-n", "/usr/local/libexec/gestur-device", action],
        {
          stdio: ["pipe", "pipe", "pipe"],
          env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin" },
        },
      );
      let output = "";
      let forcedStop;
      const timer = setTimeout(
        () => {
          if (action === "status") child.kill("SIGKILL");
          else {
            // The helper handles TERM by rolling back. Allow bounded network and
            // hostname restorations to finish before a last-resort forced stop.
            child.kill("SIGTERM");
            forcedStop = setTimeout(() => child.kill("SIGKILL"), 120000);
          }
        },
        action === "status" ? 30000 : 240000,
      );
      child.stdout.on("data", (chunk) => {
        output += chunk;
        if (output.length > 32768) child.kill("SIGKILL");
      });
      child.stderr.resume();
      const failed = () =>
        reject(
          new ApiError(
            503,
            "No se pudo completar la configuración del dispositivo. Comprueba su estado antes de volver a intentarlo.",
          ),
        );
      child.on("error", () => {
        clearTimeout(timer);
        clearTimeout(forcedStop);
        failed();
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        clearTimeout(forcedStop);
        if (code !== 0) return failed();
        try {
          resolve(JSON.parse(output));
        } catch {
          failed();
        }
      });
      child.stdin.on("error", () => {});
      child.stdin.end(settings ? JSON.stringify(settings) : "");
    });
}
