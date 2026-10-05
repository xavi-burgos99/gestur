import Fastify from "fastify";
import { createPreviews } from "./previews.mjs";
import cookie from "@fastify/cookie";
import multipart from "@fastify/multipart";
import staticFiles from "@fastify/static";
import { readFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { randomBytes, randomUUID } from "node:crypto";
import { createStore, ApiError, atomicJson } from "./store.mjs";
import { createImporter } from "./models.mjs";
import { systemWifi, validateWifi } from "./wifi.mjs";
import { createCredentials } from "./credentials.mjs";
import { systemDevice, validateDeviceRequest } from "./device.mjs";
import { createPresets } from "./presets.mjs";

export async function createApp(options) {
  const {
    configPath,
    modelsDir,
    token,
    staticRoot = path.resolve(import.meta.dirname, "../dist"),
    wifi = systemWifi(),
    wifiDelayMs = 5000,
    device = systemDevice(),
    deviceDelayMs = 5000,
    deviceStatePath = process.env.GESTUR_DEVICE_STATE ||
      "/etc/gestur/device.json",
  } = options;
  const credentials = createCredentials({ token, deviceStatePath });
  await credentials.read();
  const store = await createStore(options);
  const presets = createPresets({ configPath, store });
  const app = Fastify({ logger: false, bodyLimit: 128 * 1024 });
  const sessions = new Map();
  const attempts = new Map();
  const timers = new Set();
  const deviceOperations = new Set();
  const jobFile = path.join(path.dirname(configPath), "wifi-job.json");
  const deviceJobFile = path.join(path.dirname(configPath), "device-job.json");
  let deviceJob = null,
    schedulingDevice = false,
    verifyingPasswords = 0,
    activeMutations = 0;
  const deviceBusy = () =>
    schedulingDevice ||
    ["pending", "applying", "rebooting"].includes(deviceJob?.state);
  let job = null;
  const importer = await createImporter({
    ...options,
    onPublished: () => store.read(),
  });
  let receivingUpload = false;
  let mutatingModels = false;
  const modelBusy = () =>
    receivingUpload ||
    mutatingModels ||
    ["processing", "awaiting_decision"].includes(importer.current()?.state);
  try {
    deviceJob = JSON.parse(await readFile(deviceJobFile, "utf8"));
  } catch {}
  if (["pending", "applying", "rebooting"].includes(deviceJob?.state)) {
    let completed = false;
    if (deviceJob.state === "rebooting") {
      try {
        const state = await device("status");
        completed =
          state.hostname === deviceJob.hostname &&
          (deviceJob.kind === "reset"
            ? !state.setup_complete
            : state.setup_complete);
      } catch {}
    }
    deviceJob = {
      ...deviceJob,
      state: completed ? "completed" : "failed",
      updated: Date.now(),
      message: completed
        ? "Configuración aplicada."
        : "El servicio se reinició antes de confirmar el cambio. Comprueba la configuración del dispositivo.",
    };
    await atomicJson(deviceJobFile, deviceJob);
  }
  try {
    job = JSON.parse(await readFile(jobFile, "utf8"));
  } catch {}
  if (["pending", "applying"].includes(job?.state)) {
    job = {
      state: "failed",
      message:
        "El servicio se reinició durante el cambio. Comprueba el estado Wi-Fi.",
      updated: Date.now(),
    };
    await atomicJson(jobFile, job);
  }
  await app.register(cookie);
  await app.register(multipart, {
    limits: { fileSize: 100 * 1024 * 1024, files: 1, fields: 0, parts: 1 },
  });
  await app.register(staticFiles, { root: staticRoot, wildcard: false });
  app.addHook("onRequest", async (request, reply) => {
    reply
      .header("X-Content-Type-Options", "nosniff")
      .header("Referrer-Policy", "same-origin")
      .header("X-Frame-Options", "DENY");
    reply.header(
      "Content-Security-Policy",
      "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'",
    );
    if (!request.url.startsWith("/api/")) return;
    reply.header("Cache-Control", "no-store");
    if (!["GET", "HEAD"].includes(request.method)) {
      const origin = request.headers.origin;
      if (
        request.headers["x-gestur-request"] !== "1" ||
        (origin && origin !== `${request.protocol}://${request.headers.host}`)
      )
        throw new ApiError(403, "La solicitud debe hacerse desde este panel.");
    }
    const endpoint = request.url.split("?")[0];
    const credential = await credentials.read();
    request.gesturCredential = credential;
    if (endpoint === "/api/session" || endpoint === "/api/setup") return;
    const session = sessions.get(request.cookies.gestur_session);
    if (
      credential.required ||
      !session ||
      session.expires < Date.now() ||
      session.revision !== credential.revision
    )
      throw new ApiError(401, "Introduce la contraseña para continuar.");
    if (!["GET", "HEAD"].includes(request.method) && deviceBusy())
      throw new ApiError(
        409,
        "Espera a que termine la configuración del dispositivo.",
      );
  });
  app.setErrorHandler((error, request, reply) => {
    const status = error.statusCode || 500;
    reply.code(status).send({
      error:
        status < 500 || error instanceof ApiError
          ? error.message
          : "No se ha podido completar la operación.",
    });
  });
  app.addHook("preHandler", async (request) => {
    const endpoint = request.url.split("?")[0];
    if (
      !endpoint.startsWith("/api/") ||
      ["GET", "HEAD"].includes(request.method) ||
      [
        "/api/session",
        "/api/setup",
        "/api/device/hostname",
        "/api/device/reset",
      ].includes(endpoint)
    )
      return;
    if (deviceBusy())
      throw new ApiError(
        409,
        "Espera a que termine la configuración del dispositivo.",
      );
    request.gesturMutation = true;
    activeMutations++;
  });
  app.addHook("onResponse", async (request) => {
    if (request.gesturMutation) {
      request.gesturMutation = false;
      activeMutations--;
    }
  });
  app.get("/api/session", async (request) => ({
    authenticated:
      !request.gesturCredential.required &&
      (sessions.get(request.cookies.gestur_session)?.expires || 0) >
        Date.now() &&
      sessions.get(request.cookies.gestur_session)?.revision ===
        request.gesturCredential.revision,
    setup_required: request.gesturCredential.required,
  }));
  app.post("/api/session", async (request, reply) => {
    if (request.gesturCredential.required)
      throw new ApiError(409, "Completa la configuración inicial.");
    const now = Date.now();
    for (const [key, value] of attempts)
      if (value.until < now) attempts.delete(key);
    const attempt = attempts.get(request.ip) || {
      count: 0,
      until: now + 60000,
    };
    if (attempt.count >= 5)
      throw new ApiError(429, "Demasiados intentos. Espera un minuto.");
    if (verifyingPasswords >= 4)
      throw new ApiError(
        429,
        "Espera unos segundos antes de volver a intentarlo.",
      );
    attempt.count++;
    if (attempts.size > 1000) attempts.clear();
    attempts.set(request.ip, attempt);
    verifyingPasswords++;
    let valid;
    try {
      valid = await request.gesturCredential.verify(request.body?.token);
    } finally {
      verifyingPasswords--;
    }
    if (!valid) {
      throw new ApiError(401, "La contraseña no es correcta.");
    }
    attempts.delete(request.ip);
    for (const [key, value] of sessions)
      if (value.expires < now) sessions.delete(key);
    if (sessions.size >= 64) sessions.delete(sessions.keys().next().value);
    const session = randomBytes(32).toString("hex");
    sessions.set(session, {
      expires: now + 8 * 60 * 60 * 1000,
      revision: request.gesturCredential.revision,
    });
    reply.setCookie("gestur_session", session, {
      httpOnly: true,
      sameSite: "strict",
      secure: request.protocol === "https",
      path: "/",
      maxAge: 28800,
    });
    return { authenticated: true };
  });
  app.delete("/api/session", async (request, reply) => {
    sessions.delete(request.cookies.gestur_session);
    reply.clearCookie("gestur_session", { path: "/" });
    return { authenticated: false };
  });
  function publicDevice(state = {}) {
    return {
      hostname: state.hostname || os.hostname(),
      default_hostname: state.default_hostname ?? null,
      ssid: state.ssid ?? state.wifi?.ssid ?? null,
      portal_url: state.portal_url ?? null,
      setup_complete: state.setup_complete,
    };
  }
  app.get("/api/setup", async (request) => {
    let state = {},
      available = true;
    try {
      state = await device("status");
    } catch {
      available = false;
    }
    return {
      ...publicDevice(state),
      required: request.gesturCredential.required,
      device_available: available,
      job: deviceJob,
    };
  });
  app.get("/api/device", async () => ({
    ...publicDevice(await device("status")),
    job: deviceJob,
  }));
  app.get("/api/device/job", async () => ({ job: deviceJob }));

  async function scheduleDevice(kind, body, reply) {
    const settings = validateDeviceRequest(kind, body);
    const blocked = () =>
      activeMutations > 0 ||
      ["pending", "applying"].includes(job?.state) ||
      modelBusy();
    if (deviceBusy() || blocked())
      throw new ApiError(
        409,
        "Espera a que terminen los cambios o la importación en curso.",
      );
    // Reserve before the asynchronous preflight; otherwise two requests could
    // both pass the check and race password, hostname or reset transactions.
    schedulingDevice = true;
    try {
      const current = publicDevice(await device("status"));
      if (blocked()) throw new ApiError(409, "Hay otro cambio en curso.");
      const hostname =
        kind === "reset"
          ? current.default_hostname
          : settings.hostname || current.hostname;
      if (!hostname)
        throw new ApiError(
          503,
          "No se pudo obtener el nombre del dispositivo.",
        );
      deviceJob = {
        id: randomUUID(),
        kind,
        state: "pending",
        created_at: Date.now(),
        updated: Date.now(),
        hostname,
        ssid:
          kind === "reset"
            ? `GESTUR-${hostname.slice(-4).toUpperCase()}`
            : current.ssid,
        portal_url: current.portal_url,
        message: "Guardando la configuración. El dispositivo se reiniciará.",
      };
      await atomicJson(deviceJobFile, deviceJob);
      const timer = setTimeout(() => {
        timers.delete(timer);
        const operation = (async () => {
          try {
            deviceJob = {
              ...deviceJob,
              state: "applying",
              updated: Date.now(),
            };
            await atomicJson(deviceJobFile, deviceJob);
            const state = await device(
              kind === "setup" ? "onboarding" : kind,
              settings,
            );
            deviceJob = {
              ...deviceJob,
              ...publicDevice(state),
              state: "rebooting",
              updated: Date.now(),
              message: "Configuración guardada. Reiniciando el dispositivo.",
            };
            sessions.clear();
            attempts.clear();
          } catch {
            deviceJob = {
              ...deviceJob,
              state: "failed",
              updated: Date.now(),
              message:
                "No se pudo completar el cambio. Comprueba la configuración del dispositivo antes de volver a intentarlo.",
            };
          } finally {
            delete settings.password;
            delete settings.confirmation;
            await atomicJson(deviceJobFile, deviceJob).catch(() => {});
          }
        })().finally(() => deviceOperations.delete(operation));
        deviceOperations.add(operation);
      }, deviceDelayMs);
      timers.add(timer);
      return reply.code(202).send({
        job: deviceJob,
        hostname: deviceJob.hostname,
        ssid: deviceJob.ssid,
        portal_url: deviceJob.portal_url,
        reboot: true,
      });
    } catch (error) {
      // A failed preflight/persist must not leave the portal locked forever.
      if (deviceJob?.state === "pending") deviceJob = null;
      throw error;
    } finally {
      schedulingDevice = false;
    }
  }
  app.post("/api/setup", async (request, reply) => {
    if (!request.gesturCredential.required)
      throw new ApiError(409, "La configuración inicial ya está completada.");
    return scheduleDevice("setup", request.body, reply);
  });
  app.put("/api/device/hostname", async (request, reply) =>
    scheduleDevice("hostname", request.body, reply),
  );
  app.post("/api/device/reset", async (request, reply) =>
    scheduleDevice("reset", request.body, reply),
  );
  app.get("/api/config", async () => ({
    config: await store.read(),
    defaults: store.defaults,
  }));
  app.put("/api/config", async (request) => ({
    config: await store.update((current) => {
      if (request.body?.active_model !== current.active_model)
        throw new ApiError(
          409,
          "El modelo ha cambiado. Vuelve a Modelos 3D antes de guardar parámetros.",
        );
      return { ...request.body, screen: current.screen };
    }),
  }));
  app.put("/api/screen", async (request) => ({
    config: await store.update((current) => ({
      ...current,
      screen: request.body,
    })),
  }));
  app.get("/api/presets", async () => presets.list());
  app.put("/api/presets", async (request) => presets.save(request.body));
  app.delete("/api/presets", async (request) => presets.delete(request.body));
  app.post("/api/presets/apply", async (request) => ({
    config: await presets.apply(request.body),
  }));
  app.get("/api/runtime", async () => {
    try {
      const state = JSON.parse(
        await readFile(
          options.statusPath ||
            path.join(path.dirname(configPath), "runtime-status.json"),
          "utf8",
        ),
      );
      return {
        online:
          Number.isFinite(state.updated_at) &&
          Math.abs(Date.now() / 1000 - state.updated_at) < 10,
        selected_model: state.selected_model,
        rendered_model: state.rendered_model,
        rendered_orientation: state.rendered_orientation ?? null,
        idle: state.idle ?? null,
        error: state.error || null,
        render_fps: state.render?.render_fps ?? null,
      };
    } catch {
      return { online: false, error: null };
    }
  });
  app.get("/api/models", async () => store.catalog());
  const preview = createPreviews({ store, modelsDir });
  app.get("/api/models/preview", async (request, reply) => {
    const image = await preview(request.query.id);
    return reply
      .type("image/png")
      .header("cache-control", "private, no-store")
      .send(image);
  });
  async function mutateModel(action, body) {
    if (modelBusy())
      throw new ApiError(
        409,
        "Espera a que termine la importación o el cambio del modelo.",
      );
    mutatingModels = true;
    try {
      return await action(body);
    } finally {
      mutatingModels = false;
    }
  }
  app.patch("/api/models", async (request) =>
    mutateModel(store.updateModel, request.body),
  );
  app.delete("/api/models", async (request) =>
    mutateModel(store.deleteModel, request.body),
  );
  app.post("/api/models", async (request, reply) => {
    if (modelBusy())
      throw new ApiError(409, "Ya se está recibiendo otro archivo.");
    receivingUpload = true;
    try {
      const upload = await request.file();
      if (!upload) throw new ApiError(400, "Selecciona un modelo o un ZIP.");
      const buffer = await upload.toBuffer();
      if (upload.file.truncated)
        throw new ApiError(413, "El archivo supera los 100 MB.");
      const job = await importer.start(buffer, upload.filename);
      return reply.code(202).send({ job });
    } finally {
      receivingUpload = false;
    }
  });
  app.get("/api/imports/current", async () => ({ job: importer.current() }));
  app.get("/api/imports/:id", async (request) => ({
    job: importer.get(request.params.id),
  }));
  app.post("/api/imports/:id/decision", async (request, reply) => {
    if (
      !request.body ||
      Object.keys(request.body).length !== 1 ||
      typeof request.body.simplify !== "boolean"
    )
      throw new ApiError(
        400,
        "Elige reducir los polígonos o continuar sin simplificar.",
      );
    return reply.code(202).send({
      job: await importer.decide(request.params.id, request.body.simplify),
    });
  });
  app.put("/api/models/active", async (request) => {
    if (
      !request.body ||
      (request.body.id !== null && typeof request.body.id !== "string")
    )
      throw new ApiError(400, "Selecciona un modelo.");
    return {
      config: await store.update((current) => ({
        ...current,
        active_model: request.body.id,
      })),
    };
  });
  app.get("/api/wifi", async () => ({ ...(await wifi("status")), job }));
  app.put("/api/wifi", async (request, reply) => {
    const settings = validateWifi(request.body);
    if (["pending", "applying"].includes(job?.state))
      throw new ApiError(409, "Ya hay un cambio Wi-Fi en curso.");
    await wifi("status"); // Fail before accepting a mutation if the privileged service is unavailable.
    if (["pending", "applying"].includes(job?.state))
      throw new ApiError(409, "Ya hay un cambio Wi-Fi en curso.");
    job = {
      state: "pending",
      ssid: settings.ssid,
      updated: Date.now(),
      message:
        "El cambio se aplicará en 5 segundos. Vuelve a conectarte a la red si se interrumpe la conexión.",
    };
    await atomicJson(jobFile, job);
    const timer = setTimeout(async () => {
      timers.delete(timer);
      try {
        job = { ...job, state: "applying", updated: Date.now() };
        await atomicJson(jobFile, job);
        await wifi("apply", settings);
        job = {
          ...job,
          state: "completed",
          updated: Date.now(),
          message: "Punto de acceso actualizado.",
        };
      } catch {
        job = {
          ...job,
          state: "failed",
          updated: Date.now(),
          message:
            "No se pudo completar el cambio. Se intentó restaurar la configuración anterior; comprueba la red en el dispositivo.",
        };
      } finally {
        delete settings.password;
        await atomicJson(jobFile, job).catch(() => {});
      }
    }, wifiDelayMs);
    timers.add(timer);
    return reply.code(202).send({ job });
  });
  app.addHook("onClose", async () => {
    for (const timer of timers) clearTimeout(timer);
    await Promise.allSettled([...deviceOperations]);
    await importer.close();
  });
  return app;
}
