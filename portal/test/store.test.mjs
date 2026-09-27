import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { createStore } from "../server/store.mjs";

async function fixture(t) {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-store-"));
  t.after(() => rm(folder, { recursive: true, force: true }));
  const configPath = path.join(folder, "config.json");
  const modelsDir = path.join(folder, "models");
  const store = await createStore({ configPath, modelsDir });
  return { folder, configPath, modelsDir, store };
}

test("fresh installation starts empty and parameters can be saved without a model", async (t) => {
  const { store } = await fixture(t);
  assert.equal((await store.read()).active_model, null);
  assert.deepEqual(await store.listModels(), []);
  const saved = await store.update((config) => ({
    ...config,
    tracking: { ...config.tracking, use_hands: true },
  }));
  assert.equal(saved.active_model, null);
  assert.equal(saved.tracking.use_hands, true);
});

test("legacy builtin selection migrates to welcome while preserving controls", async (t) => {
  const { configPath, store } = await fixture(t);
  const saved = structuredClone(store.defaults);
  saved.active_model = "capitell.obj";
  saved.controls.smoothing_ms = 234;
  await writeFile(configPath, JSON.stringify(saved));
  assert.equal((await store.read()).active_model, null);
  const migrated = JSON.parse(await readFile(configPath, "utf8"));
  assert.equal(migrated.active_model, null);
  assert.equal(migrated.controls.smoothing_ms, 234);
});

test("existing user imports survive migration and can return to the welcome screen", async (t) => {
  const { store, modelsDir } = await fixture(t);
  const uuid = randomUUID();
  await mkdir(path.join(modelsDir, uuid));
  await writeFile(path.join(modelsDir, uuid, "model.glb"), "fixture");
  await writeFile(
    path.join(modelsDir, uuid, ".gestur-model.json"),
    JSON.stringify({
      name: "Mi pieza",
      entrypoint: "model.glb",
      size: 321,
      triangles: 499998,
      originalTriangles: 1500000,
      sourceFormat: "FBX",
      simplified: true,
      warnings: ["Ruta de textura reparada."],
    }),
  );
  const id = `${uuid}/model.glb`;
  await store.update((config) => ({ ...config, active_model: id }));
  assert.equal((await store.read()).active_model, id);
  const [model] = await store.listModels();
  assert.equal(model.triangles, 499998);
  assert.equal(model.sourceFormat, "FBX");
  assert.equal(model.simplified, true);
  await store.update((config) => ({ ...config, active_model: null }));
  assert.equal((await store.listModels()).length, 1);
});

test("catalog metadata cannot reference files outside an imported package", async (t) => {
  const { store, folder, modelsDir } = await fixture(t);
  const uuid = randomUUID();
  await mkdir(path.join(modelsDir, uuid));
  await writeFile(path.join(folder, "outside.glb"), "fixture");
  await writeFile(
    path.join(modelsDir, uuid, ".gestur-model.json"),
    JSON.stringify({ entrypoint: "../../outside.glb" }),
  );
  assert.deepEqual(await store.listModels(), []);
});
