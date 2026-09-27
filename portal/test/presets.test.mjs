import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtemp,
  mkdir,
  readFile,
  writeFile,
  rm,
  readdir,
  chmod,
  symlink,
} from "node:fs/promises";
import { randomUUID } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { createStore } from "../server/store.mjs";
import { createPresets } from "../server/presets.mjs";

const sections = (config) =>
  Object.fromEntries(
    ["tracking", "render", "controls"].map((key) => [
      key,
      structuredClone(config[key]),
    ]),
  );
const status = (code) => (error) => error.statusCode === code;

async function fixture(t) {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-presets-"));
  t.after(() => rm(folder, { recursive: true, force: true }));
  const configPath = path.join(folder, "config.json");
  const modelsDir = path.join(folder, "models");
  const store = await createStore({ configPath, modelsDir });
  const presets = createPresets({ configPath, store });
  return {
    folder,
    configPath,
    modelsDir,
    filename: path.join(folder, "presets.json"),
    store,
    presets,
    parameters: sections(store.defaults),
  };
}

async function addModel(modelsDir) {
  const directory = randomUUID();
  await mkdir(path.join(modelsDir, directory));
  await writeFile(path.join(modelsDir, directory, "model.glb"), "fixture");
  await writeFile(
    path.join(modelsDir, directory, ".gestur-model.json"),
    JSON.stringify({ entrypoint: "model.glb", name: directory }),
  );
  return `${directory}/model.glb`;
}

test("presets save the complete parameter draft without applying it, persist across restart and return detached copies", async (t) => {
  const { configPath, store, presets, parameters } = await fixture(t);
  assert.deepEqual(await presets.list(), { presets: [] });
  const before = await readFile(configPath);
  parameters.tracking = {
    ...parameters.tracking,
    camera_index: 2,
    use_hands: true,
    width: 1280,
    height: 720,
  };
  parameters.render = {
    ...parameters.render,
    exposure: 73,
    ambient_light: "gallery",
    target_fps: 30,
  };
  parameters.controls.idle_mode = "float";
  parameters.controls.mappings[0].continuous_speed = 42;
  const expected = structuredClone(parameters);
  const saved = await presets.save({ name: "  Sala de piedra  ", parameters });
  assert.equal(saved.overwritten, false);
  assert.equal(saved.preset.name, "Sala de piedra");
  assert.deepEqual(saved.preset.parameters, expected);
  assert.ok(Number.isFinite(Date.parse(saved.preset.updated_at)));
  assert.deepEqual(saved.presets, [saved.preset]);
  assert.deepEqual(await readFile(configPath), before);
  parameters.controls.mappings[0].continuous_speed = 250;
  saved.preset.parameters.render.exposure = 0;
  saved.presets[0].parameters.tracking.camera_index = 5;
  const restarted = createPresets({ configPath, store });
  assert.deepEqual((await restarted.list()).presets[0].parameters, expected);
});

test("trim and NFC names overwrite exactly while case differences stay distinct", async (t) => {
  const { presets, parameters } = await fixture(t);
  const first = await presets.save({ name: "Cafe\u0301", parameters });
  assert.equal(first.preset.name, "Café");
  parameters.render.exposure = 20;
  const updated = await presets.save({ name: "  Café ", parameters });
  assert.equal(updated.overwritten, true);
  assert.equal(updated.presets.length, 1);
  assert.equal(updated.preset.parameters.render.exposure, 20);
  assert.equal(
    (await presets.save({ name: "café", parameters })).overwritten,
    false,
  );
  const removed = await presets.delete({ name: " Cafe\u0301 " });
  assert.deepEqual(
    removed.presets.map((entry) => entry.name),
    ["café"],
  );
  await assert.rejects(presets.delete({ name: "Café" }), status(404));
  await assert.rejects(presets.apply({ name: "missing" }), status(404));
});

test("apply replaces only parameters and preserves the model selected after saving and during a queued update", async (t) => {
  const { store, presets, modelsDir, parameters } = await fixture(t);
  const first = await addModel(modelsDir),
    second = await addModel(modelsDir);
  await store.update((current) => ({ ...current, active_model: first }));
  parameters.controls.smoothing_ms = 327;
  parameters.render.exposure = 0;
  await presets.save({ name: "Escultura", parameters });
  let entered, release;
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const changing = store.update(async (current) => {
    entered();
    await gate;
    return {
      ...current,
      active_model: second,
      render: { ...current.render, exposure: 100 },
    };
  });
  await started;
  const applying = presets.apply({ name: "Escultura" });
  release();
  await changing;
  const applied = await applying;
  assert.equal(applied.active_model, second);
  assert.equal(applied.schema_version, 1);
  assert.deepEqual(sections(applied), parameters);
  assert.deepEqual(await store.read(), applied);
  assert.equal((await store.catalog()).models.length, 2);
});

test("concurrent saves are serialized without lost presets or duplicate names", async (t) => {
  const { presets, parameters } = await fixture(t);
  const results = await Promise.all(
    Array.from({ length: 20 }, (_, index) =>
      presets.save({ name: `Preset ${index}`, parameters }),
    ),
  );
  assert.equal(results.filter((result) => result.overwritten).length, 0);
  const duplicate = await Promise.all(
    [10, 20, 30].map((exposure) =>
      presets.save({
        name: "Preset 0",
        parameters: {
          ...parameters,
          render: { ...parameters.render, exposure },
        },
      }),
    ),
  );
  assert.ok(duplicate.every((result) => result.overwritten));
  const list = (await presets.list()).presets;
  assert.equal(list.length, 20);
  assert.equal(
    list.find((entry) => entry.name === "Preset 0").parameters.render.exposure,
    30,
  );
});

test("50-preset limit permits overwrite and frees a slot after deletion", async (t) => {
  const { presets, parameters } = await fixture(t);
  await Promise.all(
    Array.from({ length: 50 }, (_, index) =>
      presets.save({ name: `Preset ${index}`, parameters }),
    ),
  );
  await assert.rejects(
    presets.save({ name: "Preset 50", parameters }),
    status(400),
  );
  assert.equal(
    (await presets.save({ name: "Preset 0", parameters })).overwritten,
    true,
  );
  await presets.delete({ name: "Preset 1" });
  assert.equal(
    (await presets.save({ name: "Preset 50", parameters })).presets.length,
    50,
  );
});

test("names are data even when they look like paths or object prototype keys", async (t) => {
  const { folder, configPath, presets, parameters } = await fixture(t);
  const outside = path.join(folder, "outside.json");
  await writeFile(outside, "Keep this file");
  const before = await readFile(configPath);
  for (const name of [
    "../outside.json",
    "__proto__",
    "constructor",
    "/etc/passwd",
  ]) {
    const saved = await presets.save({ name, parameters });
    assert.equal(saved.preset.name, name);
    assert.deepEqual(sections(await presets.apply({ name })), parameters);
    await presets.delete({ name });
  }
  assert.equal(await readFile(outside, "utf8"), "Keep this file");
  assert.deepEqual(await readFile(configPath), before);
  assert.deepEqual((await readdir(folder)).sort(), [
    "config.json",
    "models",
    "outside.json",
    "presets.json",
  ]);
  assert.equal({}.polluted, undefined);
});

test("invalid names and parameter relationships cannot replace a saved preset or add model/network settings", async (t) => {
  const { presets, parameters, filename, configPath } = await fixture(t);
  await presets.save({ name: "Keep", parameters });
  const saved = await readFile(filename),
    config = await readFile(configPath);
  for (const name of [
    null,
    true,
    [],
    {},
    "",
    "  ",
    "a".repeat(81),
    "line\nbreak",
    "name\x00",
    "name\x85",
    "bad\ud800",
  ])
    await assert.rejects(presets.save({ name, parameters }), status(400));
  const invalid = [
    null,
    [],
    {},
    { ...parameters, active_model: null },
    { ...parameters, wifi: {} },
    { tracking: parameters.tracking, controls: parameters.controls },
    { ...parameters, render: { ...parameters.render, exposure: 101 } },
  ];
  const reversedThresholds = structuredClone(parameters);
  reversedThresholds.controls.mappings[0].left_threshold = 0.8;
  invalid.push(reversedThresholds);
  const duplicateOutput = structuredClone(parameters);
  duplicateOutput.controls.mappings[1].output =
    duplicateOutput.controls.mappings[0].output;
  invalid.push(duplicateOutput);
  for (const candidate of invalid)
    await assert.rejects(
      presets.save({ name: "Keep", parameters: candidate }),
      status(400),
    );
  for (const body of [
    null,
    {},
    { name: "Keep", parameters, active_model: null },
  ])
    await assert.rejects(presets.save(body), status(400));
  await assert.rejects(
    presets.apply({ name: "Keep", active_model: "other.glb" }),
    status(400),
  );
  await assert.rejects(
    presets.delete({ name: "Keep", all: true }),
    status(400),
  );
  assert.deepEqual(await readFile(filename), saved);
  assert.deepEqual(await readFile(configPath), config);
});

test("corrupt or ambiguous files report 503 and remain untouched by all operations", async (t) => {
  const { presets, parameters, filename, configPath } = await fixture(t);
  const record = {
    name: "Café",
    parameters,
    updated_at: new Date().toISOString(),
  };
  const cases = [
    "{broken",
    "[]",
    "null",
    JSON.stringify({ version: 2, presets: [] }),
    JSON.stringify({ version: 1, presets: [{ ...record, parameters: {} }] }),
    JSON.stringify({
      version: 1,
      presets: [record, { ...record, name: " Cafe\u0301 " }],
    }),
    JSON.stringify({
      version: 1,
      presets: [{ ...record, updated_at: "not a date" }],
    }),
  ];
  const config = await readFile(configPath);
  for (const raw of cases) {
    await writeFile(filename, raw);
    for (const action of [
      () => presets.list(),
      () => presets.save({ name: "New", parameters }),
      () => presets.delete({ name: "Café" }),
      () => presets.apply({ name: "Café" }),
    ])
      await assert.rejects(action(), status(503));
    assert.equal(await readFile(filename, "utf8"), raw);
    assert.deepEqual(await readFile(configPath), config);
  }
});

test("preset symlinks cannot read or overwrite files outside the storage location", async (t) => {
  const { folder, presets, parameters, filename } = await fixture(t);
  const target = path.join(folder, "unrelated.json");
  const content = JSON.stringify({ version: 1, presets: [] });
  await writeFile(target, content);
  await symlink(target, filename);
  await assert.rejects(presets.list(), status(503));
  await assert.rejects(presets.save({ name: "No", parameters }), status(503));
  assert.equal(await readFile(target, "utf8"), content);
});

test(
  "failed atomic writes preserve presets and do not poison the operation queue",
  { skip: process.getuid?.() === 0 },
  async (t) => {
    const { folder, presets, parameters, filename } = await fixture(t);
    await presets.save({ name: "Keep", parameters });
    const before = await readFile(filename);
    await chmod(folder, 0o500);
    try {
      await assert.rejects(
        presets.save({ name: "No", parameters }),
        /EACCES|EPERM/,
      );
      await assert.rejects(presets.delete({ name: "Keep" }), /EACCES|EPERM/);
      assert.deepEqual(await readFile(filename), before);
    } finally {
      await chmod(folder, 0o700);
    }
    assert.equal(
      (await presets.save({ name: "After", parameters })).presets.length,
      2,
    );
    assert.equal(
      (await readdir(folder)).some((name) => name.endsWith(".tmp")),
      false,
    );
    await rm(filename);
    assert.deepEqual(await presets.list(), { presets: [] });
  },
);

test("shared parameter validation migrates legacy fields without mutating the caller or live config", async (t) => {
  const { store, parameters, configPath } = await fixture(t);
  parameters.render.ambient_light = "warm";
  delete parameters.render.exposure;
  delete parameters.controls.idle_mode;
  const before = structuredClone(parameters),
    file = await readFile(configPath);
  const checked = store.validateParameters(parameters);
  assert.equal(checked.render.ambient_light, "sunset");
  assert.equal(checked.render.exposure, 50);
  assert.equal(checked.controls.idle_mode, "return");
  assert.deepEqual(parameters, before);
  assert.deepEqual(await readFile(configPath), file);
});
