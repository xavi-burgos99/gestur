import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createApp } from "../server/app.mjs";

test("screen settings require authentication and survive stale parameter saves", async (t) => {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-screen-api-"));
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
  assert.equal(
    (
      await app.inject({
        method: "PUT",
        url: "/api/screen",
        headers,
        payload: {},
      })
    ).statusCode,
    401,
  );
  const original = (await request("GET", "config")).json().config;
  const screen = {
    orientation: 90,
    content_size: "large",
    model_size: "small",
  };
  const saved = await request("PUT", "screen", screen);
  assert.equal(saved.statusCode, 200, saved.body);
  assert.deepEqual(saved.json().config.screen, screen);
  original.render.exposure = 75;
  const stale = await request("PUT", "config", original);
  assert.equal(stale.statusCode, 200, stale.body);
  assert.deepEqual(stale.json().config.screen, screen);
  assert.equal(stale.json().config.render.exposure, 75);
  const invalid = await request("PUT", "screen", {
    ...screen,
    orientation: 45,
  });
  assert.equal(invalid.statusCode, 400, invalid.body);
  assert.deepEqual(
    (await request("GET", "config")).json().config.screen,
    screen,
  );
});
