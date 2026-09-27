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
  options = {},
) {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-api-"));
  const app = await createApp({
    configPath: path.join(folder, "config.json"),
    modelsDir: path.join(folder, "models"),
    token,
    wifi,
    wifiDelayMs: 5,
    ...options,
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
test("persisted controls, schema and cross-field validation, empty model catalog", async (t) => {
  const { request, folder } = await fixture(t);
  const config = (await request("GET", "config")).json().config;
  assert.deepEqual((await request("GET", "models")).json().models, []);
  assert.equal((await request("GET", "models")).json().active, null);
  assert.equal(
    (await request("PUT", "models/active", { id: null })).statusCode,
    200,
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
      rendered_orientation: { x: 90, y: 0, z: 270 },
      idle: { mode: "float", active: true },
      error: null,
    }),
  );
  const runtime = (await request("GET", "runtime")).json();
  assert.equal(runtime.online, true);
  assert.deepEqual(runtime.rendered_orientation, { x: 90, y: 0, z: 270 });
  assert.deepEqual(runtime.idle, { mode: "float", active: true });
  await writeFile(
    path.join(folder, "runtime-status.json"),
    JSON.stringify({
      updated_at: Date.now() / 1000 - 60,
      rendered_model: "capitell.obj",
    }),
  );
  assert.equal((await request("GET", "runtime")).json().online, false);
});

function testGlb() {
  const text = JSON.stringify({ asset: { version: "2.0" } });
  const json = Buffer.from(text.padEnd(Math.ceil(text.length / 4) * 4, " "));
  const result = Buffer.alloc(20 + json.length);
  result.writeUInt32LE(0x46546c67);
  result.writeUInt32LE(2, 4);
  result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(json.length, 12);
  result.writeUInt32LE(0x4e4f534a, 16);
  json.copy(result, 20);
  return result;
}
function uploadBody(filename = "model.ply") {
  return Buffer.from(
    `--upload\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\nContent-Type: application/octet-stream\r\n\r\nply\r\n--upload--\r\n`,
  );
}
async function settledJob(request) {
  for (let i = 0; i < 100; i++) {
    const job = (await request("GET", "imports/current")).json().job;
    if (job?.state !== "processing") return job;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("Import did not settle");
}
test("multipart import returns202, polls durablejob, prompts onlyhighpoly and serializes consent", async (t) => {
  const { request, app } = await fixture(t, undefined, {
    modelConverter: async ({ workspace }) =>
      writeFile(path.join(workspace, "model.glb"), testGlb()),
    modelChecker: async () => ({ triangles: 1200000 }),
  });
  assert.equal((await request("GET", "imports/current")).json().job, null);
  assert.equal((await app.inject("/api/imports/current")).statusCode, 401);
  const response = await request("POST", "models", uploadBody(), {
    "content-type": "multipart/form-data; boundary=upload",
  });
  assert.equal(response.statusCode, 202);
  assert.equal(response.json().job.state, "processing");
  const pending = await settledJob(request);
  assert.equal(pending.state, "awaiting_decision");
  assert.equal(pending.proposal.targetTriangles, 500000);
  assert.deepEqual((await request("GET", "models")).json().models, []);
  assert.equal(
    (await request("GET", `imports/${pending.id}`)).json().job.id,
    pending.id,
  );
  assert.equal((await request("GET", "imports/missing")).statusCode, 404);
  assert.equal(
    (
      await request("POST", `imports/${pending.id}/decision`, {
        simplify: false,
        target: 1,
      })
    ).statusCode,
    400,
  );
  assert.equal(
    (
      await request("POST", `imports/${pending.id}/decision`, {
        simplify: false,
      })
    ).statusCode,
    202,
  );
  assert.equal(
    (
      await request("POST", `imports/${pending.id}/decision`, {
        simplify: false,
      })
    ).statusCode,
    409,
  );
  const completed = await settledJob(request);
  assert.equal(completed.state, "completed");
  const catalog = (await request("GET", "models")).json();
  assert.equal(catalog.models.length, 1);
  assert.equal(catalog.active, completed.model.id);
  assert.equal(
    (await request("PUT", "models/active", { id: completed.model.id }))
      .statusCode,
    200,
  );
});

test("first import is selected without polling config, later imports retain it and API rejects null", async (t) => {
  const { request, folder } = await fixture(t, undefined, {
    modelConverter: async ({ workspace }) =>
      writeFile(path.join(workspace, "model.glb"), testGlb()),
    modelChecker: async () => ({ triangles: 100 }),
  });
  async function upload() {
    assert.equal(
      (
        await request("POST", "models", uploadBody(), {
          "content-type": "multipart/form-data; boundary=upload",
        })
      ).statusCode,
      202,
    );
    const done = await settledJob(request);
    assert.equal(done.state, "completed", done.error);
    return done.model.id;
  }
  const first = await upload();
  const persisted = () =>
    readFile(path.join(folder, "config.json"), "utf8").then(JSON.parse);
  assert.equal((await persisted()).active_model, first);
  const second = await upload();
  assert.notEqual(second, first);
  assert.equal((await persisted()).active_model, first);
  assert.equal(
    (await request("PUT", "models/active", { id: null })).statusCode,
    400,
  );
  const invalid = await persisted();
  invalid.active_model = null;
  assert.equal((await request("PUT", "config", invalid)).statusCode, 409);
  assert.equal((await persisted()).active_model, first);
});

async function importedModelFixture(
  t,
  checker = async () => ({ triangles: 100 }),
) {
  const fixtureData = await fixture(t, undefined, {
    modelConverter: async ({ workspace }) =>
      writeFile(path.join(workspace, "model.glb"), testGlb()),
    modelChecker: checker,
  });
  const upload = async () => {
    const response = await fixtureData.request("POST", "models", uploadBody(), {
      "content-type": "multipart/form-data; boundary=upload",
    });
    assert.equal(response.statusCode, 202);
    return settledJob(fixtureData.request);
  };
  return { ...fixtureData, upload };
}

test("authenticated model management persists edits, preserves selection and deletes with fallback", async (t) => {
  const { request, upload } = await importedModelFixture(t);
  const first = (await upload()).model.id;
  const second = (await upload()).model.id;
  const changed = await request("PATCH", "models", {
    id: second,
    name: "  Capitel  ",
    orientation: { x: 270, y: 0, z: 90 },
  });
  assert.equal(changed.statusCode, 200, changed.body);
  assert.equal(changed.json().model.name, "Capitel");
  assert.deepEqual(changed.json().model.orientation, { x: 270, y: 0, z: 90 });
  assert.equal(changed.json().config.active_model, first);
  assert.deepEqual(
    (await request("GET", "models"))
      .json()
      .models.find((model) => model.id === second).orientation,
    { x: 270, y: 0, z: 90 },
  );
  const deleted = await request("DELETE", "models", { id: first });
  assert.equal(deleted.statusCode, 200);
  assert.equal(deleted.json().config.active_model, second);
  const last = await request("DELETE", "models", { id: second });
  assert.equal(last.statusCode, 200);
  assert.equal(last.json().config.active_model, null);
  assert.deepEqual((await request("GET", "models")).json().models, []);
});

test("model mutation validation rejects malformed ids, nulls, extra fields and invalid rotations", async (t) => {
  const { app, request, upload } = await importedModelFixture(t);
  const id = (await upload()).model.id;
  assert.equal(
    (
      await app.inject({
        method: "DELETE",
        url: "/api/models",
        headers: { "x-gestur-request": "1" },
        payload: { id },
      })
    ).statusCode,
    401,
  );
  for (const body of [
    null,
    [],
    {},
    { id: null },
    { id },
    { id, name: null },
    { id, name: " " },
    { id, name: "a".repeat(101) },
    { id, name: "line\nbreak" },
    { id, name: "Valid", entrypoint: "../../outside.glb" },
    { id, orientation: null },
    { id, orientation: { x: 0, y: 0 } },
    { id, orientation: { x: 0, y: 0, z: 90, extra: 1 } },
    { id, orientation: { x: -90, y: 0, z: 0 } },
    { id, orientation: { x: 360, y: 0, z: 0 } },
    { id, orientation: { x: 45, y: 0, z: 0 } },
    { id, orientation: { x: "90", y: 0, z: 0 } },
  ]) {
    assert.equal(
      (await request("PATCH", "models", body)).statusCode,
      400,
      JSON.stringify(body),
    );
  }
  for (const badId of [
    "../config.json",
    "../../etc/passwd",
    "/tmp/model.glb",
    "missing/model.glb",
  ]) {
    assert.equal(
      (await request("PATCH", "models", { id: badId, name: "No" })).statusCode,
      404,
    );
    assert.equal(
      (await request("DELETE", "models", { id: badId })).statusCode,
      404,
    );
  }
  for (const body of [null, {}, { id: null }, { id, recursive: true }])
    assert.equal((await request("DELETE", "models", body)).statusCode, 400);
  assert.equal((await request("GET", "models")).json().active, id);
});

test("pending imports prevent destructive model management until the import finishes", async (t) => {
  let triangles = 100;
  const { request, upload } = await importedModelFixture(t, async () => ({
    triangles,
  }));
  const id = (await upload()).model.id;
  triangles = 1200000;
  const pending = await upload();
  assert.equal(pending.state, "awaiting_decision");
  assert.equal(
    (await request("PATCH", "models", { id, name: "Wait" })).statusCode,
    409,
  );
  assert.equal((await request("DELETE", "models", { id })).statusCode, 409);
  assert.equal(
    (
      await request("POST", `imports/${pending.id}/decision`, {
        simplify: false,
      })
    ).statusCode,
    202,
  );
  assert.equal((await settledJob(request)).state, "completed");
  assert.equal(
    (await request("PATCH", "models", { id, name: "Ready" })).statusCode,
    200,
  );
});

test("lighting and standby settings round-trip through API and legacy requests keep defaults", async (t) => {
  const { request, folder } = await fixture(t);
  const initial = (await request("GET", "config")).json();
  assert.equal(initial.defaults.render.ambient_light, "none");
  assert.equal(initial.defaults.render.exposure, 50);
  assert.equal(initial.defaults.controls.idle_mode, "return");
  const config = initial.config;
  config.controls.smoothing_ms = 123;
  for (const light of ["none", "studio", "gallery", "sunset", "rim"]) {
    for (const mode of ["hold", "return", "float"]) {
      config.render.ambient_light = light;
      config.controls.idle_mode = mode;
      const saved = await request("PUT", "config", config);
      assert.equal(saved.statusCode, 200, saved.body);
      assert.deepEqual(saved.json().config, config);
      assert.deepEqual((await request("GET", "config")).json().config, config);
    }
  }
  for (const [section, field] of [
    ["render", "ambient_light"],
    ["controls", "idle_mode"],
  ]) {
    for (const value of [null, "automatic", 1]) {
      const invalid = structuredClone(config);
      invalid[section][field] = value;
      assert.equal((await request("PUT", "config", invalid)).statusCode, 400);
      assert.deepEqual((await request("GET", "config")).json().config, config);
    }
  }
  delete config.render.ambient_light;
  delete config.render.exposure;
  delete config.controls.idle_mode;
  const migrated = await request("PUT", "config", config);
  assert.equal(migrated.statusCode, 200);
  const expected = structuredClone(config);
  expected.render.ambient_light = "none";
  expected.render.exposure = 50;
  expected.controls.idle_mode = "return";
  assert.deepEqual(migrated.json().config, expected);
  assert.deepEqual(
    JSON.parse(await readFile(path.join(folder, "config.json"), "utf8")),
    expected,
  );
});

test("exposure round-trips independently of lighting and legacy names resolve to scenarios", async (t) => {
  const { request, folder } = await fixture(t);
  const config = (await request("GET", "config")).json().config;
  config.render.ambient_light = "gallery";
  for (const exposure of [10, 37, 50, 100]) {
    config.render.exposure = exposure;
    const saved = await request("PUT", "config", config);
    assert.equal(saved.statusCode, 200, saved.body);
    assert.deepEqual(saved.json().config, config);
    assert.deepEqual((await request("GET", "config")).json().config, config);
  }
  for (const exposure of [null, 9, 101, 49.5, "50", true, [], {}]) {
    const invalid = structuredClone(config);
    invalid.render.exposure = exposure;
    assert.equal((await request("PUT", "config", invalid)).statusCode, 400);
    assert.deepEqual((await request("GET", "config")).json().config, config);
  }
  for (const [oldPreset, preset] of [
    ["soft", "studio"],
    ["warm", "sunset"],
    ["cool", "gallery"],
    ["contrast", "rim"],
  ]) {
    const legacy = structuredClone(config);
    legacy.render.ambient_light = oldPreset;
    delete legacy.render.exposure;
    const response = await request("PUT", "config", legacy);
    assert.equal(response.statusCode, 200, response.body);
    const expected = structuredClone(legacy);
    expected.render.ambient_light = preset;
    expected.render.exposure = 50;
    assert.deepEqual(response.json().config, expected);
    assert.deepEqual(
      JSON.parse(await readFile(path.join(folder, "config.json"), "utf8")),
      expected,
    );
  }
});
