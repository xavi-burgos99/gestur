import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  CONTROL_OPTIONS,
  GESTURE_OPTIONS,
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

const displayOnlyInputs = [
  "hands_center_x",
  "hands_center_y",
  "hands_separation_x",
  "left_hand_rotation",
  "right_hand_rotation",
];

test("object movement labels follow the exhibition camera without changing saved axes or physical gestures", () => {
  const expected = [
    ["rotation_roll", "Giro horizontal", "yaw"],
    ["rotation_pitch", "Giro vertical", "pitch"],
    ["rotation_yaw", "Inclinación lateral", "roll"],
    ["position_x", "Desplazamiento horizontal", "translate-x"],
    ["position_y", "Desplazamiento vertical", "translate-y"],
    ["position_z", "Desplazamiento en profundidad", "depth"],
  ];
  for (const [id, label, motion] of expected) {
    const option = CONTROL_OPTIONS.find((item) => item.id === id);
    assert.equal(option.label, label);
    assert.deepEqual(option.icon, { subject: "model", motion });
    assert.equal(createControlMapping("existing", id, "head_yaw").output, id);
  }
  // Body tracking angles keep their physical meaning; only object output
  // presentation compensates for the viewer's camera orientation.
  for (const body of ["head", "torso", "left_hand", "right_hand"]) {
    const yaw = GESTURE_OPTIONS.find((item) => item.id === `${body}_yaw`);
    const roll = GESTURE_OPTIONS.find((item) => item.id === `${body}_roll`);
    assert.equal(yaw.label, "Giro horizontal");
    assert.equal(yaw.icon.motion, "yaw");
    assert.ok(roll.label.startsWith("Inclinación lateral"));
    assert.equal(roll.icon.motion, "roll");
    assert.equal(gestureFromSelection(selectionForGesture(yaw.id)), yaw.id);
    assert.equal(gestureFromSelection(selectionForGesture(roll.id)), roll.id);
  }
});

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
  assert.equal(reachable.size, 29);
  for (const input of displayOnlyInputs) assert.ok(!reachable.has(input));
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

test("opening, pinching and combined distance finish without asking for an axis or an unrelated type", () => {
  for (const side of ["left", "right"]) {
    for (const movement of ["openness", "pinch"]) {
      const input = `${side}_hand_${movement}`;
      const selection = selectionForGesture(input);
      const steps = gestureSteps(selection);
      assert.deepEqual(
        steps.map((step) => step.id),
        ["body", "side", "movement"],
      );
      assert.deepEqual(
        steps.at(-1).options.map((option) => option.id),
        ["translation", "rotation", "openness", "pinch"],
      );
      assert.equal(selection.axis, null);
      assert.equal(gestureFromSelection(selection), input);
    }
  }
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

test("editing reconstructs selectable gestures and starts unselected for display-only gestures", () => {
  for (const gesture of GESTURE_OPTIONS) {
    const selection = selectionForGesture(gesture.id);
    if (gesture.selection) {
      assert.equal(gestureFromSelection(selection), gesture.id);
    } else {
      assert.deepEqual(selection, emptyGestureSelection());
      assert.deepEqual(
        gestureSteps(selection).map((step) => step.id),
        ["body"],
      );
      assert.equal(gestureFromSelection(selection), null);
      assert.equal(
        createControlMapping("new", "rotation_roll", gesture.id),
        null,
      );
    }
  }
  assert.deepEqual(
    GESTURE_OPTIONS.filter((gesture) => !gesture.selection)
      .map((gesture) => gesture.id)
      .sort(),
    displayOnlyInputs,
  );
  assert.deepEqual(selectionForGesture("unknown"), emptyGestureSelection());
  assert.equal(
    gestureFromSelection({
      ...emptyGestureSelection(),
      legacy: "left_hand_rotation",
    }),
    null,
  );
});

test("the full display catalog covers the runtime schema including existing legacy gestures", async () => {
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
  assert.equal(GESTURE_OPTIONS.length, 34);
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
    CONTROL_OPTIONS.find((option) => option.id === "position_y").label,
    /vertical/,
  );
  assert.match(
    CONTROL_OPTIONS.find((option) => option.id === "position_z").label,
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
    for (const gesture of GESTURE_OPTIONS.filter(
      (option) => option.selection,
    )) {
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

test("saved display-only gestures and their tuning remain valid without choosing a replacement", async (t) => {
  const folder = await mkdtemp(
    path.join(os.tmpdir(), "gestur-existing-controls-"),
  );
  t.after(() => rm(folder, { recursive: true, force: true }));
  const store = await createStore({
    configPath: path.join(folder, "config.json"),
    modelsDir: path.join(folder, "models"),
  });
  for (const input of displayOnlyInputs) {
    const existing = {
      ...createControlMapping("existing", "rotation_roll", "left_hand_roll"),
      input,
      scale: 74,
      center: 0.3,
      invert: true,
    };
    await store.update((config) => ({
      ...config,
      controls: { ...config.controls, mappings: [existing] },
    }));
    const draft = await store.read();
    const original = structuredClone(draft.controls.mappings[0]);
    const selection = selectionForGesture(draft.controls.mappings[0].input);
    assert.equal(gestureFromSelection(selection), null);
    assert.deepEqual(draft.controls.mappings[0], original);
    const updated = await store.update((config) => ({
      ...config,
      controls: {
        ...config.controls,
        mappings: config.controls.mappings.map((mapping) => ({
          ...mapping,
          enabled: false,
        })),
      },
    }));
    assert.deepEqual(updated.controls.mappings[0], {
      ...existing,
      enabled: false,
    });
    assert.deepEqual((await store.read()).controls.mappings[0], {
      ...existing,
      enabled: false,
    });
  }
});

test("response changes keep the selected movement and previous tuning", () => {
  const control = {
    ...createControlMapping("existing", "rotation_roll", "left_hand_roll"),
    input: "left_hand_rotation",
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
