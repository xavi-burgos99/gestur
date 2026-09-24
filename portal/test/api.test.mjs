import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { createApp } from "../server/app.mjs";
const token = "a-test-admin-token-with-32-characters";
async function fixture(
  t,
  wifi = async () => ({ ssid: "GESTUR-ABCD", secured: false, active: true }),
) {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-api-"));
  const app = await createApp({
    configPath: path.join(folder, "config.json"),
    modelsDir: path.join(folder, "models"),
    token,
    wifi,
    wifiDelayMs: 5,
  });
  t.after(async () => {
    await app.close();
    await rm(folder, { recursive: true, force: true });
  });
  const login = await app.inject({
    method: "POST",
    url: "/api/session",
    headers: { "x-gestur-request": "1" },
    payload: { token },
  });
  assert.equal(login.statusCode, 200);
  const cookie = login.headers["set-cookie"].split(";")[0];
  const request = (method, url, payload, extra = {}) =>
    app.inject({
      method,
      url: `/api/${url}`,
      headers: { cookie, "x-gestur-request": "1", ...extra },
      ...(payload === undefined ? {} : { payload }),
    });
  return { app, folder, request, cookie };
}
test("authentication, strict cookie, origin protection, token is never exposed", async (t) => {
  const { app, request } = await fixture(t);
  assert.equal((await app.inject("/api/config")).statusCode, 401);
  assert.equal(
    (
      await request(
        "PUT",
        "wifi",
        { ssid: "test" },
        { origin: "http://evil.test" },
      )
    ).statusCode,
    403,
  );
  assert.equal(
    (
      await app.inject({
        method: "POST",
        url: "/api/session",
        payload: { token },
      })
    ).statusCode,
    403,
  );
  for (let i = 0; i < 5; i++)
    assert.equal(
      (
        await app.inject({
          method: "POST",
          url: "/api/session",
          headers: { "x-gestur-request": "1" },
          payload: { token: "wrong" },
        })
      ).statusCode,
      401,
    );
  assert.equal(
    (
      await app.inject({
        method: "POST",
        url: "/api/session",
        headers: { "x-gestur-request": "1" },
        payload: { token },
      })
    ).statusCode,
    429,
  );
  assert.ok(!(await request("GET", "config")).body.includes(token));
});
test("persisted controls, schema and cross-field validation, immutable builtin model", async (t) => {
  const { request, folder } = await fixture(t);
  const config = (await request("GET", "config")).json().config;
  assert.equal(
    (await request("GET", "models")).json().models[0].name,
    "Capitel",
  );
  config.tracking.use_hands = true;
  assert.equal((await request("PUT", "config", config)).statusCode, 200);
  assert.equal(
    JSON.parse(await readFile(path.join(folder, "config.json"), "utf8"))
      .tracking.use_hands,
    true,
  );
  config.controls.mappings[0].left_threshold = 0.8;
  assert.equal((await request("PUT", "config", config)).statusCode, 400);
  config.controls.mappings[0].left_threshold = 0.25;
  config.tracking.inference_fps = 1000;
  assert.equal((await request("PUT", "config", config)).statusCode, 400);
  assert.equal(
    (await request("PUT", "models/active", { id: "../../etc/passwd" }))
      .statusCode,
    400,
  );
  assert.equal(
    (await request("PUT", "models/active", { id: "missing.obj" })).statusCode,
    400,
  );
});
test("bad saved configuration is preserved and reported, not silently reset", async (t) => {
  const { request, folder } = await fixture(t);
  await writeFile(path.join(folder, "config.json"), "{broken");
  assert.equal((await request("GET", "config")).statusCode, 503);
  assert.equal(
    await readFile(path.join(folder, "config.json"), "utf8"),
    "{broken",
  );
});
test("Wi-Fi changes acknowledged before apply, persist status but never passwords", async (t) => {
  const calls = [];
  const { request, folder } = await fixture(t, async (action, value) => {
    calls.push({ action, value: structuredClone(value) });
    return {
      ssid: value?.ssid || "GESTUR-ABCD",
      secured: !!value?.password,
      active: true,
    };
  });
  const response = await request("PUT", "wifi", {
    ssid: "GESTUR-MUSEO",
    password: "secret123",
  });
  assert.equal(response.statusCode, 202);
  assert.equal(response.json().job.state, "pending");
  assert.equal(calls.filter((c) => c.action === "apply").length, 0);
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal(calls.filter((c) => c.action === "apply").length, 1);
  assert.equal((await request("GET", "wifi")).json().job.state, "completed");
  assert.ok(
    !(await readFile(path.join(folder, "wifi-job.json"), "utf8")).includes(
      "secret123",
    ),
  );
});
test("Wi-Fi invalid settings and unavailable helper do not simulate success", async (t) => {
  const { request } = await fixture(t, async () => {
    throw new Error("unavailable");
  });
  assert.equal(
    (await request("PUT", "wifi", { ssid: "test", password: "short" }))
      .statusCode,
    400,
  );
  assert.equal(
    (
      await request("PUT", "wifi", {
        ssid: "test",
        password: "12345678",
        command: "rm",
      })
    ).statusCode,
    400,
  );
  assert.equal((await request("GET", "wifi")).statusCode, 500);
  assert.equal(
    (await request("PUT", "wifi", { ssid: "test", password: null })).statusCode,
    500,
  );
});

test("concurrent Wi-Fi requests schedule only one mutation", async (t) => {
  const { request } = await fixture(t, async (action) => {
    if (action === "status")
      await new Promise((resolve) => setTimeout(resolve, 5));
    return { ssid: "GESTUR-ABCD", secured: false, active: true };
  });
  const replies = await Promise.all([
    request("PUT", "wifi", { ssid: "one" }),
    request("PUT", "wifi", { ssid: "two" }),
  ]);
  assert.deepEqual(replies.map((r) => r.statusCode).sort(), [202, 409]);
});

test("runtime status reports stale/absent viewer without claiming active rendering", async (t) => {
  const { request, folder } = await fixture(t);
  assert.equal((await request("GET", "runtime")).json().online, false);
  await writeFile(
    path.join(folder, "runtime-status.json"),
    JSON.stringify({
      updated_at: Date.now() / 1000,
      selected_model: "capitell.obj",
      rendered_model: "capitell.obj",
      error: null,
    }),
  );
  assert.equal((await request("GET", "runtime")).json().online, true);
  await writeFile(
    path.join(folder, "runtime-status.json"),
    JSON.stringify({
      updated_at: Date.now() / 1000 - 60,
      rendered_model: "capitell.obj",
    }),
  );
  assert.equal((await request("GET", "runtime")).json().online, false);
});
