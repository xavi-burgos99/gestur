export async function api(
  url,
  { method = "GET", body, silentAuth = false, signal } = {},
) {
  const form = body instanceof FormData;
  const response = await fetch(`/api/${url}`, {
    method,
    signal,
    headers: {
      "X-Gestur-Request": "1",
      ...(!form && body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { body: form ? body : JSON.stringify(body) } : {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && url !== "session" && !silentAuth)
      window.dispatchEvent(new Event("gestur-expired"));
    const error = new Error(
      data.error || "No hay conexión con el dispositivo.",
    );
    error.status = response.status;
    throw error;
  }
  return data;
}
