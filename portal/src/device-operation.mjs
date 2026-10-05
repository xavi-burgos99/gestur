const OPERATION_KEY = "gestur-device-operation";
const ACTIVE_STATES = ["pending", "applying", "rebooting"];
const HOSTNAME = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
export const validSetupPassword = (value) =>
  typeof value === "string" && /^[\x20-\x7e]{8,63}$/.test(value);
export const deviceJobActive = (job) => ACTIVE_STATES.includes(job?.state);
export function hostnameError(value, optional = false) {
  if (optional && value === "") return "";
  return typeof value === "string" && HOSTNAME.test(value)
    ? ""
    : "Usa de 1 a 63 letras minúsculas, números o guiones. Sin espacios ni .local; empieza y termina con una letra o un número.";
}
export function localUrl(hostname) {
  return typeof hostname === "string" && HOSTNAME.test(hostname)
    ? `http://${hostname}.local/`
    : null;
}
export function ipUrl(value) {
  try {
    const url = new URL(value);
    if (
      ["http:", "https:"].includes(url.protocol) &&
      (/^(?:\d{1,3}\.){3}\d{1,3}$/.test(url.hostname) ||
        /^\[[\da-f:]+\]$/i.test(url.hostname))
    ) {
      return url.origin + "/";
    }
  } catch {
    /* A hostname is not an IP fallback. */
  }
  return null;
}
export function operationData(
  value,
  locationHref = globalThis.location?.href,
  now = Date.now(),
) {
  const job = value.job || {};
  return {
    kind: value.kind || job.kind,
    hostname: job.hostname || value.hostname,
    ssid: job.ssid || value.ssid,
    portal_url:
      ipUrl(job.portal_url || value.portal_url) || ipUrl(locationHref),
    startedAt: value.startedAt || now,
    job: {
      id: job.id,
      created_at: Number.isFinite(job.created_at) ? job.created_at : undefined,
      kind: job.kind || value.kind,
      state: job.state || "unknown",
      message: typeof job.message === "string" ? job.message : undefined,
      error: typeof job.error === "string" ? job.error : undefined,
    },
  };
}
export function matchesDeviceJob(job, operation, expectedId) {
  if (!job || job.kind !== operation.kind) return false;
  if (expectedId) return job.id === expectedId;
  if (job.hostname !== operation.hostname) return false;
  if (deviceJobActive(job)) return true;
  // A request may complete/fail before its dropped response can be recovered.
  // Accept terminal state only when its fixed creation time matches this attempt.
  return (
    ["completed", "failed"].includes(job.state) &&
    Number.isFinite(job.created_at) &&
    Number.isFinite(operation.startedAt) &&
    job.created_at >= operation.startedAt - 5000 &&
    job.created_at <= operation.startedAt + 10 * 60 * 1000
  );
}
export function rememberDeviceOperation(value, storage) {
  // Never persist form values, passwords or arbitrary server job fields.
  const operation = operationData(value);
  try {
    (storage ?? globalThis.sessionStorage)?.setItem(
      OPERATION_KEY,
      JSON.stringify(operation),
    );
  } catch {
    /* Storage may be disabled. */
  }
  return operation;
}
export function forgetDeviceOperation(storage) {
  try {
    (storage ?? globalThis.sessionStorage)?.removeItem(OPERATION_KEY);
  } catch {
    /* Storage may be disabled. */
  }
}
export function restoreDeviceOperation(storage, now = Date.now()) {
  try {
    const value = JSON.parse(
      (storage ?? globalThis.sessionStorage)?.getItem(OPERATION_KEY) || "null",
    );
    if (
      value &&
      ["setup", "hostname", "reset"].includes(value.kind) &&
      Number.isFinite(value.startedAt) &&
      now >= value.startedAt &&
      now - value.startedAt < 10 * 60 * 1000
    ) {
      return operationData(value, undefined, now);
    }
  } catch {
    /* A missing or invalid recovery record is harmless. */
  }
  forgetDeviceOperation(storage);
  return null;
}
