import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createApp } from "../server/app.mjs";

test("authenticated presets store the draft, overwrite by name, apply all parameters and delete", async (t) => {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-presets-api-"));
  const token = "test-presets-administrator-token-long-enough";
  const app = await createApp({
    token,
    configPath: path.join(folder, "config.json"),
    deviceStatePath: path.join(folder, "device.json"),
    modelsDir: path.join(folder, "models"),
  });
  t.after(async () => {
    await app.close();
    await rm(folder, { recursive: true, force: true });
  });
  assert.equal((await app.inject("/api/presets")).statusCode, 401);
  const headers = { "x-gestur-request": "1" };
  const login = await app.inject({
    method: "POST",
    url: "/api/session",
    headers,
    payload: { token },
  });
  const cookie = login.headers["set-cookie"].split(";")[0];
  const request = (method, endpoint, payload, extra = {}) =>
    app.inject({
      method,
      url: `/api/${endpoint}`,
      headers: { ...headers, cookie, ...extra },
      ...(payload === undefined ? {} : { payload }),
    });
  const original = (await request("GET", "config")).json().config;
  const parameters = Object.fromEntries(
    ["tracking", "render", "controls"].map((key) => [
      key,
      structuredClone(original[key]),
    ]),
  );
  parameters.tracking.mirror = !parameters.tracking.mirror;
  parameters.render.exposure = 5;
  parameters.controls.idle_mode = "float";
  parameters.controls.mappings[0].enabled = false;
  assert.equal(
    (
      await request(
        "PUT",
        "presets",
        { name: "Sala", parameters },
        { origin: "http://evil.invalid" },
      )
    ).statusCode,
    403,
  );
  const saved = await request("PUT", "presets", { name: "Sala", parameters });
  assert.equal(saved.statusCode, 200, saved.body);
  assert.equal(saved.json().overwritten, false);
  assert.deepEqual((await request("GET", "config")).json().config, original);
  parameters.render.exposure = 75;
  const replacement = await request("PUT", "presets", {
    name: "Sala",
    parameters,
  });
  assert.equal(replacement.json().overwritten, true);
  assert.equal((await request("GET", "presets")).json().presets.length, 1);
  assert.ok(
    !(await readFile(path.join(folder, "presets.json"), "utf8")).includes(
      "active_model",
    ),
  );
  const applied = await request("POST", "presets/apply", { name: "Sala" });
  assert.equal(applied.statusCode, 200, applied.body);
  assert.deepEqual(applied.json().config, { ...original, ...parameters });
  assert.equal(
    (await request("DELETE", "presets", { name: "Sala" })).statusCode,
    200,
  );
  assert.deepEqual((await request("GET", "presets")).json().presets, []);
  assert.equal(
    (await request("POST", "presets/apply", { name: "Sala" })).statusCode,
    404,
  );
  assert.deepEqual((await request("GET", "config")).json().config, {
    ...original,
    ...parameters,
  });
});
