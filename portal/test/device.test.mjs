import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { scryptSync } from "node:crypto";
import { createApp } from "../server/app.mjs";
import { createCredentials } from "../server/credentials.mjs";
import { validateHostname } from "../server/device.mjs";

const legacy = "legacy-administrator-token-at-least-24-chars";
const password = "Mi clave 123";
const headers = { "x-gestur-request": "1" };
const pending = () => ({
  version: 1,
  setup_complete: false,
  password_hash: null,
});
function configured(value = password) {
  const salt = Buffer.alloc(16, 7);
  return {
    version: 1,
    setup_complete: true,
    password_hash: {
      algorithm: "scrypt",
      salt: salt.toString("hex"),
      key: scryptSync(value, salt, 32, { N: 16384, r: 8, p: 1 }).toString(
        "hex",
      ),
      n: 16384,
      r: 8,
      p: 1,
    },
  };
}
async function fixture(t, initial = pending(), extra = {}) {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-device-"));
  const deviceStatePath = path.join(folder, "device.json");
  await writeFile(deviceStatePath, JSON.stringify(initial));
  let hostname = "gestur-ab12",
    fail = false;
  const calls = [];
  const device = async (action, settings) => {
    if (fail) throw new Error("unavailable");
    if (action !== "status") calls.push(action);
    if (action === "onboarding") {
      hostname = settings.hostname || hostname;
      await writeFile(
        deviceStatePath,
        JSON.stringify(configured(settings.password)),
      );
    } else if (action === "hostname") hostname = settings.hostname;
    else if (action === "reset") {
      hostname = "gestur-ab12";
      await writeFile(deviceStatePath, JSON.stringify(pending()));
    }
    return {
      hostname,
      default_hostname: "gestur-ab12",
      ssid: "GESTUR-AB12",
      portal_url: "http://10.42.0.1",
      setup_complete: JSON.parse(await readFile(deviceStatePath, "utf8"))
        .setup_complete,
    };
  };
  const options = {
    configPath: path.join(folder, "config.json"),
    modelsDir: path.join(folder, "models"),
    token: legacy,
    device,
    deviceStatePath,
    deviceDelayMs: 20,
    wifi: async () => ({ ssid: "GESTUR-AB12", active: true, secured: false }),
    ...extra,
  };
  const app = await createApp(options);
  t.after(async () => {
    await app.close();
    await rm(folder, { recursive: true, force: true });
  });
  const request = (method, url, payload, cookie) =>
    app.inject({
      method,
      url: `/api/${url}`,
      headers: { ...headers, ...(cookie ? { cookie } : {}) },
      ...(payload === undefined ? {} : { payload }),
    });
  const login = async (value = password) => {
    const reply = await request("POST", "session", { token: value });
    assert.equal(reply.statusCode, 200, reply.body);
    return reply.headers["set-cookie"].split(";")[0];
  };
  const settle = async () => {
    for (let i = 0; i < 100; i++) {
      const job = (await request("GET", "setup")).json().job;
      if (!["pending", "applying"].includes(job?.state)) return job;
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    throw new Error("Device operation did not settle");
  };
  return {
    app,
    request,
    login,
    settle,
    folder,
    deviceStatePath,
    options,
    calls,
    unavailable: () => {
      fail = true;
    },
  };
}

test("hostname accepts only lowercase DNS labels, without suffix or edge hyphens", () => {
  for (const value of ["a", "0", "gestur-1234", "x".repeat(63)])
    assert.equal(validateHostname(value), value);
  for (const value of [
    "",
    "Gestur",
    "a.local",
    "-a",
    "a-",
    "a_b",
    "á",
    "a b",
    "x".repeat(64),
    null,
    [],
    "a\n",
  ])
    assert.throws(() => validateHostname(value), /minúsculas/);
});

test("fresh setup is public, protected against cross-origin posts, and never accepts legacy login", async (t) => {
  const { app, request, calls } = await fixture(t);
  const setup = (await request("GET", "setup")).json();
  assert.equal(setup.required, true);
  assert.equal(setup.hostname, "gestur-ab12");
  assert.equal((await request("GET", "config")).statusCode, 401);
  assert.equal(
    (await request("POST", "session", { token: legacy })).statusCode,
    409,
  );
  assert.equal(
    (
      await app.inject({
        method: "POST",
        url: "/api/setup",
        headers: { ...headers, origin: "http://evil.invalid" },
        payload: { password, confirmation: password },
      })
    ).statusCode,
    403,
  );
  assert.equal(
    (await app.inject({ method: "POST", url: "/api/setup", payload: {} }))
      .statusCode,
    403,
  );
  assert.deepEqual(calls, []);
});

test("setup validates confirmation and hostname before doing any privileged mutation", async (t) => {
  const { request, calls } = await fixture(t);
  for (const body of [
    null,
    {},
    [],
    { password, confirmation: "wrong" },
    { password: "123", confirmation: "123" },
    { password: "contraseña", confirmation: "contraseña" },
    { hostname: "A", password, confirmation: password },
    { hostname: "a.local", password, confirmation: password },
    { password, confirmation: password, path: "/etc" },
  ])
    assert.equal((await request("POST", "setup", body)).statusCode, 400);
  assert.deepEqual(calls, []);
});

test("setup retains a blank hostname, acknowledges before apply, hashes credentials and does not persist secrets", async (t) => {
  const { request, settle, folder, login, calls } = await fixture(t);
  const reply = await request("POST", "setup", {
    hostname: "",
    password,
    confirmation: password,
  });
  assert.equal(reply.statusCode, 202, reply.body);
  assert.equal(reply.json().hostname, "gestur-ab12");
  assert.equal(reply.json().job.state, "pending");
  assert.deepEqual(calls, []);
  assert.equal((await settle()).state, "rebooting");
  assert.deepEqual(calls, ["onboarding"]);
  assert.equal((await request("GET", "setup")).json().required, false);
  const cookie = await login();
  assert.equal(
    (await request("GET", "config", undefined, cookie)).statusCode,
    200,
  );
  assert.equal(
    (await request("POST", "session", { token: legacy })).statusCode,
    401,
  );
  assert.equal(
    (await request("POST", "setup", { password, confirmation: password }))
      .statusCode,
    409,
  );
  for (const name of ["device.json", "device-job.json"]) {
    const raw = await readFile(path.join(folder, name), "utf8");
    assert.ok(!raw.includes(password));
  }
  assert.ok(!(await request("GET", "setup")).body.includes("password_hash"));
});

test("simultaneous setup requests apply exactly once", async (t) => {
  const { request, settle, calls } = await fixture(t);
  const replies = await Promise.all(
    [1, 2].map(() =>
      request("POST", "setup", { password, confirmation: password }),
    ),
  );
  assert.deepEqual(replies.map((r) => r.statusCode).sort(), [202, 409]);
  await settle();
  assert.deepEqual(calls, ["onboarding"]);
});

test("hostname and reset need authentication; reset requires exact confirmation", async (t) => {
  const { request, login, settle, calls } = await fixture(t, configured());
  assert.equal(
    (await request("POST", "device/reset", { confirmation: "BORRAR" }))
      .statusCode,
    401,
  );
  assert.equal(
    (await request("PUT", "device/hostname", { hostname: "museo" })).statusCode,
    401,
  );
  const cookie = await login();
  for (const body of [
    {},
    { confirmation: true },
    { confirmation: "BORRAR", path: "/etc" },
  ])
    assert.equal(
      (await request("POST", "device/reset", body, cookie)).statusCode,
      400,
    );
  assert.equal(
    (await request("POST", "device/reset", { confirmation: "BORRAR" }, cookie))
      .statusCode,
    202,
  );
  assert.equal(
    (await request("PUT", "wifi", { ssid: "other" }, cookie)).statusCode,
    409,
  );
  await settle();
  assert.deepEqual(calls, ["reset"]);
  assert.equal(
    (await request("GET", "config", undefined, cookie)).statusCode,
    401,
  );
  assert.equal((await request("GET", "setup")).json().required, true);
  assert.equal(
    (await request("POST", "session", { token: password })).statusCode,
    409,
  );
});

test("hostname change preserves password and excludes other mutations until reboot", async (t) => {
  const { request, login, settle, calls } = await fixture(t, configured());
  const cookie = await login();
  const config = (await request("GET", "config", undefined, cookie)).json()
    .config;
  const response = await request(
    "PUT",
    "device/hostname",
    { hostname: "sala-1" },
    cookie,
  );
  assert.equal(response.statusCode, 202);
  assert.equal(response.json().hostname, "sala-1");
  assert.equal(
    (await request("PUT", "config", config, cookie)).statusCode,
    409,
  );
  assert.equal(
    (await request("POST", "device/reset", { confirmation: "BORRAR" }, cookie))
      .statusCode,
    409,
  );
  await settle();
  assert.equal((await request("GET", "setup")).json().hostname, "sala-1");
  assert.deepEqual(calls, ["hostname"]);
  await login();
});

test("helper failures do not announce success or leave preflight locked", async (t) => {
  const { request, unavailable, calls } = await fixture(t);
  unavailable();
  for (let i = 0; i < 2; i++)
    assert.equal(
      (await request("POST", "setup", { password, confirmation: password }))
        .statusCode,
      500,
    );
  const state = (await request("GET", "setup")).json();
  assert.equal(state.required, true);
  assert.equal(state.device_available, false);
  assert.equal(state.job, null);
  assert.deepEqual(calls, []);
});

test("credential changes invalidate existing sessions; malformed state never falls back to legacy", async (t) => {
  const { request, login, deviceStatePath } = await fixture(t, configured());
  const cookie = await login();
  await writeFile(
    deviceStatePath,
    JSON.stringify(configured("Otra clave 456")),
  );
  assert.equal(
    (await request("GET", "config", undefined, cookie)).statusCode,
    401,
  );
  await login("Otra clave 456");
  await writeFile(deviceStatePath, "{broken");
  assert.equal(
    (await request("POST", "session", { token: legacy })).statusCode,
    500,
  );
});

test("legacy updates retain old login until explicitly reset", async (t) => {
  const { request, login } = await fixture(t, {
    version: 1,
    setup_complete: true,
    legacy_auth: true,
    password_hash: null,
  });
  const cookie = await login(legacy);
  assert.equal((await request("GET", "setup")).json().required, false);
  assert.equal(
    (await request("GET", "config", undefined, cookie)).statusCode,
    200,
  );
});

test("restarted portal confirms rebooting jobs and fails interrupted pending jobs without replaying secrets", async (t) => {
  const { app, options, folder, calls } = await fixture(t, configured());
  await app.close();
  const jobFile = path.join(folder, "device-job.json");
  for (const [state, expected] of [
    ["rebooting", "completed"],
    ["pending", "failed"],
  ]) {
    await writeFile(
      jobFile,
      JSON.stringify({
        id: "test",
        kind: "setup",
        hostname: "gestur-ab12",
        state,
      }),
    );
    const restarted = await createApp(options);
    assert.equal(
      (await restarted.inject("/api/setup")).json().job.state,
      expected,
    );
    await restarted.close();
  }
  assert.deepEqual(calls, []);
});

test("scrypt state rejects unbounded parameters and wrong key encodings", async (t) => {
  const { deviceStatePath } = await fixture(t);
  const credentials = createCredentials({ token: legacy, deviceStatePath });
  for (const values of [
    { n: 2 ** 25 },
    { key: "invalid" },
    { salt: "0" },
    { algorithm: "plain" },
  ]) {
    const state = configured();
    Object.assign(state.password_hash, values);
    await writeFile(deviceStatePath, JSON.stringify(state));
    await assert.rejects(credentials.read());
  }
});
