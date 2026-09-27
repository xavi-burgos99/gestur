import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  CONTROL_OPTIONS,
  GESTURE_OPTIONS,
  LEGACY_GESTURE_OPTIONS,
  emptyGestureSelection,
  selectionForGesture,
  gestureFromSelection,
  changeGestureSelection,
  gestureSteps,
  createControlMapping,
  modeChanges,
  responseOptions,
} from "../src/control-catalog.mjs";
import { createStore } from "../server/store.mjs";

test("progressive choices expose the next step only after its parent, with an icon on every option", () => {
  let selection = emptyGestureSelection();
  const ids = () => gestureSteps(selection).map((step) => step.id);
  assert.deepEqual(ids(), ["body"]);
  selection = changeGestureSelection(selection, "body", "hands");
  assert.deepEqual(ids(), ["body", "side"]);
  selection = changeGestureSelection(selection, "side", "left");
  assert.deepEqual(ids(), ["body", "side", "movement"]);
  selection = changeGestureSelection(selection, "movement", "translation");
  assert.deepEqual(ids(), ["body", "side", "movement", "axis"]);
  assert.equal(gestureFromSelection(selection), null);
  selection = changeGestureSelection(selection, "axis", "scale");
  assert.equal(gestureFromSelection(selection), "left_hand_scale");

  const reachable = new Set();
  function visit(current) {
    const input = gestureFromSelection(current);
    if (input) {
      assert.ok(!reachable.has(input), `Duplicate path to ${input}`);
      reachable.add(input);
      return;
    }
    const step = gestureSteps(current).find((item) => !current[item.id]);
    assert.ok(step, "Every incomplete selection must have a next step");
    for (const option of step.options) {
      assert.ok(option.label);
      assert.ok(
        ["head", "body", "hand", "hands"].includes(option.icon.subject),
      );
      visit(changeGestureSelection(current, step.id, option.id));
    }
  }
  visit(emptyGestureSelection());
  assert.deepEqual(
    [...reachable].sort(),
    GESTURE_OPTIONS.filter((item) => item.selection)
      .map((item) => item.id)
      .sort(),
  );
});

test("changing a parent clears incompatible descendants and cannot retain a previous completed gesture", () => {
  const hand = selectionForGesture("left_hand_scale");
  const otherSide = changeGestureSelection(hand, "side", "right");
  assert.equal(otherSide.movement, null);
  assert.equal(otherSide.axis, null);
  assert.equal(gestureFromSelection(otherSide), null);
  const rotation = changeGestureSelection(hand, "movement", "rotation");
  assert.equal(rotation.axis, null);
  assert.equal(gestureFromSelection(rotation), null);
  const body = changeGestureSelection(hand, "body", "body");
  assert.equal(body.side, null);
  assert.equal(body.movement, null);
  assert.equal(body.axis, null);
  assert.equal(gestureFromSelection(body), null);
  const cleared = changeGestureSelection(hand, "body", null);
  assert.deepEqual(cleared, emptyGestureSelection());
});

test("opening hands and combined distance finish without asking for an axis or an unrelated type", () => {
  const opening = selectionForGesture("right_hand_openness");
  assert.deepEqual(
    gestureSteps(opening).map((step) => step.id),
    ["body", "side", "movement"],
  );
  const combined = changeGestureSelection(
    emptyGestureSelection(),
    "body",
    "combined",
  );
  assert.deepEqual(
    gestureSteps(combined).map((step) => step.id),
    ["body", "combined"],
  );
  assert.equal(gestureFromSelection(combined), null);
  assert.equal(
    gestureFromSelection(
      changeGestureSelection(combined, "combined", "distance"),
    ),
    "hands_distance",
  );
});

test("editing reconstructs every progressive and legacy gesture without silently replacing its input", () => {
  for (const gesture of GESTURE_OPTIONS) {
    const selection = selectionForGesture(gesture.id);
    assert.equal(gestureFromSelection(selection), gesture.id);
    assert.equal(!!selection.legacy, !gesture.selection);
  }
  assert.deepEqual(LEGACY_GESTURE_OPTIONS.map((gesture) => gesture.id).sort(), [
    "hands_center_x",
    "hands_center_y",
    "hands_separation_x",
    "left_hand_pinch",
    "left_hand_rotation",
    "right_hand_pinch",
    "right_hand_rotation",
  ]);
});

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
