import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtemp,
  mkdir,
  readFile,
  writeFile,
  rm,
  stat,
  chmod,
  readdir,
  symlink,
} from "node:fs/promises";
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

test("existing user imports keep their selection and cannot return to welcome", async (t) => {
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
  await assert.rejects(
    store.update((config) => ({ ...config, active_model: null })),
    /Debe haber un modelo seleccionado/,
  );
  assert.equal((await store.read()).active_model, id);
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

async function addModel(modelsDir, uuid = randomUUID()) {
  const folder = path.join(modelsDir, uuid);
  await mkdir(folder);
  await writeFile(path.join(folder, "model.glb"), "fixture");
  await writeFile(
    path.join(folder, ".gestur-model.json"),
    JSON.stringify({ entrypoint: "model.glb", name: uuid }),
  );
  return `${uuid}/model.glb`;
}

test("startup persists a fallback and restart retains the last selected model", async (t) => {
  const { configPath, modelsDir, store } = await fixture(t);
  const second = await addModel(
    modelsDir,
    "22222222-2222-2222-2222-222222222222",
  );
  const first = await addModel(
    modelsDir,
    "11111111-1111-1111-1111-111111111111",
  );
  const restarted = await createStore({ configPath, modelsDir });
  assert.equal((await restarted.read()).active_model, first);
  assert.equal(
    JSON.parse(await readFile(configPath, "utf8")).active_model,
    first,
  );
  await store.update((current) => ({ ...current, active_model: second }));
  const again = await createStore({ configPath, modelsDir });
  assert.equal((await again.read()).active_model, second);
  // New imports do not replace the user's chosen exhibition.
  await addModel(modelsDir, "00000000-0000-0000-0000-000000000000");
  assert.equal((await again.read()).active_model, second);
});

test("missing selection falls back to another model, and only an empty catalog selects null", async (t) => {
  const { configPath, modelsDir, store } = await fixture(t);
  const first = await addModel(
    modelsDir,
    "11111111-1111-1111-1111-111111111111",
  );
  const second = await addModel(
    modelsDir,
    "22222222-2222-2222-2222-222222222222",
  );
  await store.update((current) => ({ ...current, active_model: second }));
  await rm(path.join(modelsDir, second.split("/")[0]), { recursive: true });
  assert.equal((await store.read()).active_model, first);
  await rm(path.join(modelsDir, first.split("/")[0]), { recursive: true });
  assert.equal((await store.read()).active_model, null);
  assert.equal(
    JSON.parse(await readFile(configPath, "utf8")).active_model,
    null,
  );
});

test("reconciliation waits for a concurrent explicit selection and preserves it", async (t) => {
  const { configPath, modelsDir, store } = await fixture(t);
  await addModel(modelsDir, "11111111-1111-1111-1111-111111111111");
  const second = await addModel(
    modelsDir,
    "22222222-2222-2222-2222-222222222222",
  );
  let enter;
  let resume;
  const entered = new Promise((resolve) => {
    enter = resolve;
  });
  const paused = new Promise((resolve) => {
    resume = resolve;
  });
  const update = store.update(async (current) => {
    enter();
    await paused;
    return { ...current, active_model: second };
  });
  await entered;
  const read = store.read();
  const snapshot = store.catalog();
  resume();
  await update;
  assert.equal((await read).active_model, second);
  const catalog = await snapshot;
  assert.equal(catalog.active, second);
  assert.ok(catalog.models.some((model) => model.id === catalog.active));
  assert.equal(
    JSON.parse(await readFile(configPath, "utf8")).active_model,
    second,
  );
});

test("startup keeps malformed configuration intact and reports it through the store", async (t) => {
  const { configPath, modelsDir } = await fixture(t);
  await writeFile(configPath, "{broken");
  const restarted = await createStore({ configPath, modelsDir });
  await assert.rejects(
    restarted.read(),
    /La configuración guardada no es válida/,
  );
  assert.equal(await readFile(configPath, "utf8"), "{broken");
});

test("model name and fixed orientation persist independently from selection and geometry", async (t) => {
  const { store, configPath, modelsDir } = await fixture(t);
  const id = await addModel(modelsDir);
  const other = await addModel(modelsDir);
  await store.update((current) => ({ ...current, active_model: other }));
  const configBefore = await readFile(configPath);
  const geometryBefore = await readFile(path.join(modelsDir, id));
  const stampBefore = await stat(modelsDir, { bigint: true });
  assert.deepEqual(
    (await store.catalog()).models.find((model) => model.id === id).orientation,
    { x: 0, y: 0, z: 0 },
  );
  const changed = await store.updateModel({
    id,
    name: "  Capitel de prueba  ",
    orientation: { x: 90, y: 180, z: 270 },
  });
  assert.equal(changed.model.name, "Capitel de prueba");
  assert.deepEqual(changed.model.orientation, { x: 90, y: 180, z: 270 });
  assert.equal(changed.config.active_model, other);
  assert.deepEqual(await readFile(configPath), configBefore);
  assert.deepEqual(await readFile(path.join(modelsDir, id)), geometryBefore);
  const stampAfter = await stat(modelsDir, { bigint: true });
  assert.notEqual(stampAfter.mtimeNs, stampBefore.mtimeNs);
  const restarted = await createStore({ configPath, modelsDir });
  const renamed = (await restarted.catalog()).models.find(
    (model) => model.id === id,
  );
  assert.equal(renamed.name, "Capitel de prueba");
  assert.deepEqual(renamed.orientation, changed.model.orientation);
  await restarted.updateModel({ id, name: "Segundo nombre" });
  assert.deepEqual(
    (await restarted.catalog()).models.find((model) => model.id === id)
      .orientation,
    changed.model.orientation,
  );
});

test("delete keeps another selection, falls back deterministically and shows empty only after last model", async (t) => {
  const { store, configPath, modelsDir } = await fixture(t);
  const first = await addModel(
    modelsDir,
    "11111111-1111-1111-1111-111111111111",
  );
  const second = await addModel(
    modelsDir,
    "22222222-2222-2222-2222-222222222222",
  );
  const third = await addModel(
    modelsDir,
    "33333333-3333-3333-3333-333333333333",
  );
  await store.update((current) => ({ ...current, active_model: second }));
  assert.equal(
    (await store.deleteModel({ id: first })).config.active_model,
    second,
  );
  assert.equal(
    (await store.deleteModel({ id: second })).config.active_model,
    third,
  );
  const restarted = await createStore({ configPath, modelsDir });
  assert.equal((await restarted.read()).active_model, third);
  assert.equal(
    (await restarted.deleteModel({ id: third })).config.active_model,
    null,
  );
  assert.deepEqual((await restarted.catalog()).models, []);
  assert.equal(
    JSON.parse(await readFile(configPath, "utf8")).active_model,
    null,
  );
  assert.equal(
    (await readdir(modelsDir)).some((entry) => entry.startsWith(".trash-")),
    false,
  );
});

test(
  "failed selection save rolls deletion back without losing the model",
  { skip: process.getuid?.() === 0 },
  async (t) => {
    const { store, folder, configPath, modelsDir } = await fixture(t);
    const id = await addModel(modelsDir);
    await store.read();
    const before = await readFile(configPath);
    await chmod(folder, 0o500);
    try {
      await assert.rejects(store.deleteModel({ id }), /EACCES|EPERM/);
      assert.equal(await readFile(path.join(modelsDir, id), "utf8"), "fixture");
      assert.deepEqual(await readFile(configPath), before);
      assert.equal(
        (await readdir(modelsDir)).some((entry) => entry.startsWith(".trash-")),
        false,
      );
    } finally {
      await chmod(folder, 0o700);
    }
    assert.equal((await store.catalog()).active, id);
  },
);

test("mutations reject symlink packages without modifying their target", async (t) => {
  const { store, modelsDir } = await fixture(t);
  const realId = await addModel(
    modelsDir,
    "11111111-1111-1111-1111-111111111111",
  );
  const alias = "22222222-2222-2222-2222-222222222222";
  await symlink(
    path.join(modelsDir, realId.split("/")[0]),
    path.join(modelsDir, alias),
  );
  await assert.rejects(
    store.updateModel({ id: `${alias}/model.glb`, name: "No" }),
    /carpeta del modelo/,
  );
  await assert.rejects(
    store.deleteModel({ id: `${alias}/model.glb` }),
    /carpeta del modelo/,
  );
  assert.equal(await readFile(path.join(modelsDir, realId), "utf8"), "fixture");
});

test("legacy v1 defaults migrate atomically once without changing user parameters", async (t) => {
  const { store, configPath } = await fixture(t);
  const legacy = structuredClone(store.defaults);
  delete legacy.render.ambient_light;
  delete legacy.controls.idle_mode;
  legacy.render.target_fps = 30;
  legacy.controls.smoothing_ms = 237;
  legacy.controls.mappings[0].scale = 44;
  legacy.tracking.use_hands = true;
  await writeFile(configPath, JSON.stringify(legacy));
  const originalInode = (await stat(configPath)).ino;
  const expected = structuredClone(legacy);
  expected.render.ambient_light = "none";
  expected.controls.idle_mode = "return";
  assert.deepEqual(await store.read(), expected);
  assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), expected);
  const migratedInode = (await stat(configPath)).ino;
  assert.notEqual(migratedInode, originalInode);
  assert.deepEqual(await store.read(), expected);
  assert.equal((await stat(configPath)).ino, migratedInode);
  // An explicit setting survives even if only the other field needs migration.
  legacy.render.ambient_light = "cool";
  await writeFile(configPath, JSON.stringify(legacy));
  assert.equal((await store.read()).render.ambient_light, "cool");
  delete legacy.render.ambient_light;
  legacy.controls.idle_mode = "hold";
  await writeFile(configPath, JSON.stringify(legacy));
  assert.equal((await store.read()).controls.idle_mode, "hold");
});

test("explicit invalid lighting or idle setting is never replaced with a default", async (t) => {
  const { store, configPath } = await fixture(t);
  for (const [section, field] of [
    ["render", "ambient_light"],
    ["controls", "idle_mode"],
  ]) {
    for (const value of [null, "", "automatic", 1, false, [], {}]) {
      const invalid = structuredClone(store.defaults);
      invalid[section][field] = value;
      const bytes = JSON.stringify(invalid);
      await writeFile(configPath, bytes);
      await assert.rejects(store.read(), /configuración guardada no es válida/);
      assert.equal(await readFile(configPath, "utf8"), bytes);
    }
  }
});
