import { spawn } from "node:child_process";
import { ApiError } from "./store.mjs";
export function validateWifi(value) {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).some((k) => !["ssid", "password"].includes(k))
  )
    throw new ApiError(400, "Solicitud Wi-Fi no válida.");
  if (
    typeof value.ssid !== "string" ||
    Buffer.byteLength(value.ssid) < 1 ||
    Buffer.byteLength(value.ssid) > 32 ||
    /[\x00-\x1f\x7f]/.test(value.ssid)
  )
    throw new ApiError(
      400,
      "El nombre Wi-Fi debe ocupar entre 1 y 32 bytes, sin caracteres de control.",
    );
  if (
    value.password !== undefined &&
    value.password !== null &&
    (typeof value.password !== "string" ||
      !/^[\x20-\x7e]{8,63}$/.test(value.password))
  )
    throw new ApiError(
      400,
      "La contraseña debe tener entre 8 y 63 caracteres ASCII.",
    );
  return value;
}
export function systemWifi() {
  return async (action, settings) =>
    new Promise((resolve, reject) => {
      const child = spawn(
        "/usr/bin/sudo",
        ["-n", "/usr/local/libexec/gestur-wifi", action],
        {
          stdio: ["pipe", "pipe", "pipe"],
          env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin" },
        },
      );
      let output = "";
      const timeout = setTimeout(() => child.kill("SIGKILL"), 90000);
      child.stdout.on("data", (chunk) => {
        output += chunk;
        if (output.length > 32768) child.kill();
      });
      // Helper deliberately emits only redacted errors; never copy subprocess stderr to clients/logs.
      child.stderr.resume();
      child.on("error", () => {
        clearTimeout(timeout);
        reject(new ApiError(503, "El servicio Wi-Fi no está disponible."));
      });
      child.on("close", (code) => {
        clearTimeout(timeout);
        if (code !== 0)
          return reject(
            new ApiError(
              503,
              "No se ha podido aplicar la configuración Wi-Fi. Revisa NetworkManager en el dispositivo.",
            ),
          );
        try {
          resolve(JSON.parse(output));
        } catch {
          reject(new ApiError(503, "Respuesta Wi-Fi no válida."));
        }
      });
      child.stdin.on("error", () => {});
      child.stdin.end(settings ? JSON.stringify(settings) : "");
    });
}
