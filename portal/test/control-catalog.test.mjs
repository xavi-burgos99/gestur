import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  CONTROL_OPTIONS,
  GESTURE_OPTIONS,
  createControlMapping,
  modeChanges,
  responseOptions,
} from "../src/control-catalog.mjs";
import { createStore } from "../server/store.mjs";

test("the gesture and movement selectors cover the runtime schema without dropping legacy gestures", async () => {
  const schema = JSON.parse(
    await readFile(
      new URL("../../config/schema.json", import.meta.url),
      "utf8",
    ),
  );
  const properties =
    schema.properties.controls.properties.mappings.items.properties;
  assert.deepEqual(
    CONTROL_OPTIONS.map((option) => option.id).sort(),
    properties.output.enum.toSorted(),
  );
  assert.deepEqual(
    GESTURE_OPTIONS.map((option) => option.id).sort(),
    properties.input.enum.toSorted(),
  );
  for (const side of ["left", "right"]) {
    const legacy = GESTURE_OPTIONS.find(
      (option) => option.id === `${side}_hand_rotation`,
    );
    const roll = GESTURE_OPTIONS.find(
      (option) => option.id === `${side}_hand_roll`,
    );
    assert.notEqual(legacy.label, roll.label);
    assert.equal(legacy.icon.side, side);
    assert.equal(roll.icon.side, side);
  }
  assert.match(
    CONTROL_OPTIONS.find((option) => option.id === "position_z").label,
    /vertical/,
  );
  assert.match(
    CONTROL_OPTIONS.find((option) => option.id === "position_y").label,
    /profundidad/,
  );
});

test("every selectable gesture and response saves through the real configuration validator", async (t) => {
  const folder = await mkdtemp(path.join(os.tmpdir(), "gestur-controls-"));
  t.after(() => rm(folder, { recursive: true, force: true }));
  const store = await createStore({
    configPath: path.join(folder, "config.json"),
    modelsDir: path.join(folder, "models"),
  });
  for (const output of CONTROL_OPTIONS) {
    for (const gesture of GESTURE_OPTIONS) {
      for (const mode of responseOptions(output.id)) {
        const control = createControlMapping(
          "test_control",
          output.id,
          gesture.id,
        );
        const mapping = { ...control, ...modeChanges(control, mode.value) };
        await store.update((config) => ({
          ...config,
          controls: { ...config.controls, mappings: [mapping] },
        }));
      }
    }
  }
});

test("response changes keep the selected movement and previous tuning", () => {
  const control = {
    ...createControlMapping("existing", "rotation_roll", "left_hand_rotation"),
    center: 0.3,
    left_threshold: 0.1,
    right_threshold: 0.9,
    continuous_speed: 74,
  };
  const changed = { ...control, ...modeChanges(control, "hybrid") };
  assert.equal(changed.output, control.output);
  assert.equal(changed.input, control.input);
  assert.equal(changed.continuous_speed, 74);
  assert.equal(changed.left_threshold, 0.1);
  assert.deepEqual(modeChanges(control, "stepped"), {});
  const recentered = modeChanges({ ...changed, center: 0.05 }, "hybrid");
  assert.ok(recentered.left_threshold < recentered.center);
  assert.ok(recentered.center < recentered.right_threshold);
  assert.deepEqual(
    modeChanges(createControlMapping("move", "position_x", "head_x"), "hybrid"),
    {},
  );
});
