import test from "node:test";
import assert from "node:assert/strict";
import {
  presetName,
  presetNameError,
  presetSaveBody,
  parametersEqual,
} from "../src/presets.mjs";

test("preset payload stores the entire visible draft without model or device data", () => {
  const draft = {
    active_model: "keep-current-model",
    network: { password: "never-save-this" },
    schema_version: 3,
    tracking: { use_hands: true, camera_index: 2 },
    render: { exposure: 75, ambient_light: "studio" },
    controls: {
      idle_mode: "float",
      mappings: [{ gesture: "head_x", enabled: true }],
    },
  };
  const payload = presetSaveBody("  Sala 1  ", draft);
  assert.equal(payload.name, "Sala 1");
  assert.deepEqual(Object.keys(payload.parameters), [
    "tracking",
    "render",
    "controls",
  ]);
  assert.deepEqual(payload.parameters.controls, draft.controls);
  assert.equal(payload.parameters.render.exposure, 75);
  assert.ok(!JSON.stringify(payload).includes("never-save-this"));
  assert.ok(!JSON.stringify(payload).includes("keep-current-model"));
  draft.controls.mappings[0].enabled = false;
  draft.render.exposure = 10;
  assert.equal(payload.parameters.controls.mappings[0].enabled, true);
  assert.equal(payload.parameters.render.exposure, 75);
});

test("preset names share backend trimming, Unicode normalization and limits", () => {
  assert.equal(presetName("  Cafe\u0301  "), "Café");
  assert.equal(presetNameError("a".repeat(80)), "");
  for (const invalid of [" ", "a".repeat(81), "a\nb", "a\x7fb", "a\x85b"]) {
    assert.ok(presetNameError(invalid));
    assert.throws(() => presetSaveBody(invalid, {}));
  }
  assert.notEqual(presetName("Sala"), presetName("sala"));
});

test("unsaved parameter detection ignores a model selection and includes all parameter groups", () => {
  const first = {
    tracking: { use_hands: false },
    render: { exposure: 50 },
    controls: { mappings: [] },
    active_model: "first",
  };
  assert.equal(
    parametersEqual(first, { ...first, active_model: "second" }),
    true,
  );
  for (const group of ["tracking", "render", "controls"]) {
    assert.equal(
      parametersEqual(first, { ...first, [group]: { changed: true } }),
      false,
    );
  }
});
