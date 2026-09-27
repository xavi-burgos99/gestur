import Fastify from "fastify";
import cookie from "@fastify/cookie";
import multipart from "@fastify/multipart";
import staticFiles from "@fastify/static";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { createStore, ApiError, atomicJson } from "./store.mjs";
import { createImporter } from "./models.mjs";
import { systemWifi, validateWifi } from "./wifi.mjs";

export async function createApp(options) {
  const {
    configPath,
    modelsDir,
    token,
    staticRoot = path.resolve(import.meta.dirname, "../dist"),
    wifi = systemWifi(),
    wifiDelayMs = 5000,
  } = options;
  if (typeof token !== "string" || token.length < 24)
    throw new Error(
      "GESTUR requires an administrator token of at least 24 characters.",
    );
  const store = await createStore(options);
  const app = Fastify({ logger: false, bodyLimit: 128 * 1024 });
  const sessions = new Map();
  const attempts = new Map();
  const timers = new Set();
  const jobFile = path.join(path.dirname(configPath), "wifi-job.json");
  let job = null;
  const importer = await createImporter({
    ...options,
    onPublished: () => store.read(),
  });
  let receivingUpload = false;
  let mutatingModels = false;
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
    if (request.url.split("?")[0] === "/api/session") return;
    const session = sessions.get(request.cookies.gestur_session);
    if (!session || session < Date.now())
      throw new ApiError(
        401,
        "Introduce la clave de administración para continuar.",
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
  app.get("/api/session", async (request) => ({
    authenticated:
      (sessions.get(request.cookies.gestur_session) || 0) > Date.now(),
  }));
  app.post("/api/session", async (request, reply) => {
    const now = Date.now();
    for (const [key, value] of attempts)
      if (value.until < now) attempts.delete(key);
    const attempt = attempts.get(request.ip) || {
      count: 0,
      until: now + 60000,
    };
    if (attempt.count >= 5)
      throw new ApiError(429, "Demasiados intentos. Espera un minuto.");
    const supplied = Buffer.from(
      typeof request.body?.token === "string" ? request.body.token : "",
    );
    const expected = Buffer.from(token);
    if (
      supplied.length !== expected.length ||
      !timingSafeEqual(supplied, expected)
    ) {
      attempt.count++;
      if (attempts.size > 1000) attempts.clear();
      attempts.set(request.ip, attempt);
      throw new ApiError(401, "La clave de administración no es correcta.");
    }
    attempts.delete(request.ip);
    for (const [key, expiry] of sessions)
      if (expiry < now) sessions.delete(key);
    if (sessions.size >= 64) sessions.delete(sessions.keys().next().value);
    const session = randomBytes(32).toString("hex");
    sessions.set(session, now + 8 * 60 * 60 * 1000);
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
      return request.body;
    }),
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
  async function mutateModel(action, body) {
    if (
      receivingUpload ||
      mutatingModels ||
      ["processing", "awaiting_decision"].includes(importer.current()?.state)
    )
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
    if (
      receivingUpload ||
      mutatingModels ||
      ["processing", "awaiting_decision"].includes(importer.current()?.state)
    )
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
    await importer.close();
  });
  return app;
}
