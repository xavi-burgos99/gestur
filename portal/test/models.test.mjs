import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import yazl from "yazl";
import { importModel as importModelWithChecker } from "../server/models.mjs";
import { validateGlb } from "../server/store.mjs";
async function zip(entries) {
  const archive = new yazl.ZipFile();
  for (const [name, data, options] of entries)
    archive.addBuffer(Buffer.from(data), name, options);
  archive.end();
  const chunks = [];
  for await (const chunk of archive.outputStream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
const importModel = (buffer, filename, dir) =>
  importModelWithChecker(buffer, filename, dir, async () => ({ ok: true }));
const obj = "mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
const entries = [
  ["scene/model.obj", obj],
  ["scene/model.mtl", "newmtl stone\nmap_Kd textures/stone.jpg\n"],
  ["scene/textures/stone.jpg", "jpeg"],
];
async function fixture(t) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "gestur-models-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}
function glb(doc) {
  const json = Buffer.from(
    JSON.stringify(doc).padEnd(
      Math.ceil(JSON.stringify(doc).length / 4) * 4,
      " ",
    ),
  );
  const data = Buffer.alloc(20 + json.length);
  data.writeUInt32LE(0x46546c67);
  data.writeUInt32LE(2, 4);
  data.writeUInt32LE(data.length, 8);
  data.writeUInt32LE(json.length, 12);
  data.writeUInt32LE(0x4e4f534a, 16);
  json.copy(data, 20);
  return data;
}
test("imports OBJ archive with nested material/texture and stable catalog metadata", async (t) => {
  const dir = await fixture(t);
  const result = await importModel(
    await zip(entries),
    "Capitel nuevo.zip",
    dir,
  );
  assert.equal(result.name, "Capitel nuevo");
  assert.match(result.id, /^[a-f0-9-]+\/scene\/model.obj$/);
  assert.equal(await readFile(path.join(dir, result.id), "utf8"), obj);
});
test("imports glTF+BIN+texture and GLB bundles", async (t) => {
  const dir = await fixture(t);
  const doc = {
    asset: { version: "2.0" },
    buffers: [{ uri: "data.bin", byteLength: 4 }],
    images: [{ uri: "texture.png" }],
  };
  assert.equal(
    (
      await importModel(
        await zip([
          ["model.gltf", JSON.stringify(doc)],
          ["data.bin", "abcd"],
          ["texture.png", "png"],
        ]),
        "gltf.zip",
        dir,
      )
    ).format,
    "GLTF",
  );
  assert.equal(
    (
      await importModel(
        await zip([["model.glb", glb({ asset: { version: "2.0" } })]]),
        "glb.zip",
        dir,
      )
    ).format,
    "GLB",
  );
});
test("rejects missing material, missing texture, remote resources and multiple models", async (t) => {
  const dir = await fixture(t);
  for (const bad of [
    [entries[0]],
    entries.slice(0, 2),
    [["model.obj", "v 0 0 0\nf 1 1 1\nmtllib http://evil.test/texture"]],
    [...entries, ["other.obj", obj]],
    [
      [
        "model.gltf",
        JSON.stringify({
          asset: { version: "2.0" },
          buffers: [{ uri: "https://evil.test/a.bin" }],
        }),
      ],
    ],
  ]) {
    await assert.rejects(importModel(await zip(bad), "bad.zip", dir));
    assert.deepEqual(await readdir(dir), []);
  }
});
test("rejects symlinks and zip traversal, leaving no staging files", async (t) => {
  const dir = await fixture(t);
  await assert.rejects(
    importModel(
      await zip([["link", "target", { mode: 0o120777 }]]),
      "link.zip",
      dir,
    ),
  );
  const safe = await zip([["safe/model.obj", "v 0 0 0\nf 1 1 1"]]);
  const bad = Buffer.from(safe);
  let offset = 0;
  while ((offset = bad.indexOf("safe/model.obj", offset)) !== -1) {
    bad.write("../x/model.obj", offset);
    offset += 14;
  }
  await assert.rejects(importModel(bad, "traversal.zip", dir));
  assert.deepEqual(await readdir(dir), []);
});
test("rejects expansion bomb and case-insensitive duplicate names", async (t) => {
  const dir = await fixture(t);
  await assert.rejects(
    importModel(
      await zip([["large.bin", Buffer.alloc(2 * 1024 * 1024)], ...entries]),
      "bomb.zip",
      dir,
    ),
  );
  await assert.rejects(
    importModel(
      await zip([
        ["A.obj", obj],
        ["a.obj", obj],
      ]),
      "duplicate.zip",
      dir,
    ),
  );
  assert.deepEqual(await readdir(dir), []);
});
test("GLB validates declared length and refuses external resource URIs", () => {
  assert.throws(() => validateGlb(Buffer.from("bad")));
  assert.throws(() =>
    validateGlb(
      glb({ asset: { version: "2.0" }, images: [{ uri: "../../private" }] }),
    ),
  );
});

test("renderer preflight failure prevents import and removes staging", async (t) => {
  const dir = await fixture(t);
  let checked = false;
  await assert.rejects(
    importModelWithChecker(await zip(entries), "bad.zip", dir, async () => {
      checked = true;
      throw new Error("invalid geometry");
    }),
  );
  assert.equal(checked, true);
  assert.deepEqual(await readdir(dir), []);
});

test(
  "real renderer accepts triangle ZIP and rejects malformed glTF before publication",
  { skip: !process.env.GESTUR_PYTHON },
  async (t) => {
    const dir = await fixture(t);
    const good = await importModelWithChecker(
      await zip([["triangle.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"]]),
      "triangle.zip",
      dir,
    );
    assert.equal(good.format, "OBJ");
    await assert.rejects(
      importModelWithChecker(
        await zip([
          [
            "broken.gltf",
            JSON.stringify({
              asset: { version: "2.0" },
              scenes: [{ nodes: [99] }],
            }),
          ],
        ]),
        "broken.zip",
        dir,
      ),
      (error) => error.statusCode === 422,
    );
    assert.equal((await readdir(dir)).length, 1);
  },
);

test("leading whitespace in MTL cannot bypass resource confinement", async (t) => {
  const dir = await fixture(t);
  await assert.rejects(
    importModel(
      await zip([
        [
          "model.obj",
          "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        ],
        ["material.mtl", "\tmap_Kd ../../outside.jpg\n"],
      ]),
      "unsafe.zip",
      dir,
    ),
  );
  assert.deepEqual(await readdir(dir), []);
});
