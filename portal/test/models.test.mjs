import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, readdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import yazl from "yazl";
import {
  createImporter,
  extractUpload,
  HIGH_TRIANGLES,
  TARGET_TRIANGLES,
} from "../server/models.mjs";
import {
  resourceResolver,
  repairTextResources,
  bundleGltf,
  readGlb,
} from "../server/package-model.mjs";
import {
  sandboxCommand,
  runNative,
  simplifyModel,
  verifyImporter,
} from "../server/native-model.mjs";
import { validateGlb } from "../server/store.mjs";
import { preflightModel } from "../server/model-check.mjs";
async function zip(entries) {
  const archive = new yazl.ZipFile();
  for (const [name, data, options] of entries)
    archive.addBuffer(Buffer.from(data), name, options);
  archive.end();
  const chunks = [];
  for await (const chunk of archive.outputStream) chunks.push(chunk);
  return Buffer.concat(chunks);
}
const obj =
  "mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl stone\nf 1/1 2/2 3/3\n";
const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAEUlEQVQImWNY1ZH2H4QZYAwAW5QKXe5oBBUAAAAASUVORK5CYII=",
  "base64",
);
const entries = [
  ["scene/model.obj", obj],
  ["scene/model.mtl", "newmtl stone\nmap_Kd C:\\Old\\TEXTURES\\Piédra.PNG\n"],
  ["scene/textures/piédra.png", png],
];
async function fixture(t) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "gestur-models-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}
function glb(doc = { asset: { version: "2.0" } }) {
  const raw = Buffer.from(JSON.stringify(doc));
  const json = Buffer.concat([
    raw,
    Buffer.alloc((4 - (raw.length % 4)) % 4, 32),
  ]);
  const data = Buffer.alloc(20 + json.length);
  data.writeUInt32LE(0x46546c67);
  data.writeUInt32LE(2, 4);
  data.writeUInt32LE(data.length, 8);
  data.writeUInt32LE(json.length, 12);
  data.writeUInt32LE(0x4e4f534a, 16);
  json.copy(data, 20);
  return data;
}
const fakeConverter = async ({ workspace }) =>
  writeFile(path.join(workspace, "model.glb"), glb());
const fakeSimplifier = async ({ workspace }) => {
  await writeFile(path.join(workspace, "simplified.glb"), glb());
  return { entrypoint: "simplified.glb" };
};
async function importer(t, options = {}) {
  const modelsDir = await fixture(t);
  const worker = await createImporter({
    modelsDir,
    modelChecker: async () => ({ triangles: 100 }),
    modelConverter: fakeConverter,
    modelSimplifier: fakeSimplifier,
    ...options,
  });
  t.after(() => worker.close());
  return { worker, modelsDir };
}
test("repairs Windows, absolute, case, Unicode and encoded paths only inside upload; ambiguity fails", () => {
  const warnings = [],
    resolve = resourceResolver(
      ["Textures/Piedra Á.png", "Material.mtl"],
      warnings,
    );
  assert.equal(
    resolve("model.obj", "C:\\old\\textures\\PIEDRA A\u0301.png"),
    "Textures/Piedra Á.png",
  );
  assert.equal(
    resolve("model.gltf", "../gone/Piedra%20%C3%81.png"),
    "Textures/Piedra Á.png",
  );
  assert.equal(
    resolve("model.obj", "/home/export/Material.mtl"),
    "Material.mtl",
  );
  assert.ok(warnings.length >= 3);
  assert.throws(
    () =>
      resourceResolver(["a/stone.jpg", "b/stone.jpg"])(
        "x.obj",
        "C:\\old\\stone.jpg",
      ),
    /varios/,
  );
  assert.throws(
    () => resolve("model.gltf", "https://evil/Material.mtl"),
    /externo/,
  );
  assert.throws(() => resolve("model.obj", "/etc/passwd"), /Falta/);
});
test("repairs OBJ material textures while preserving geometry and supporting comments", async (t) => {
  const dir = await fixture(t),
    { entrypoint, files } = await extractUpload(
      await zip(entries),
      "modelo.zip",
      dir,
    ),
    warnings = [];
  await repairTextResources(dir, entrypoint, files, warnings);
  assert.match(
    await readFile(path.join(dir, "scene/model.mtl"), "utf8"),
    /map_Kd textures\/piédra.png/,
  );
  assert.equal(await readFile(path.join(dir, entrypoint), "utf8"), obj);
  assert.ok(warnings.length > 0);
  await writeFile(
    path.join(dir, entrypoint),
    obj.replace("model.mtl", "model.mtl # generated"),
  );
  await repairTextResources(dir, entrypoint, files, []);
  await writeFile(
    path.join(dir, "scene/second.mtl"),
    "newmtl other\nKd 1 1 1\n",
  );
  await writeFile(
    path.join(dir, entrypoint),
    obj.replace("model.mtl", "model.mtl second.mtl # exported libraries"),
  );
  await repairTextResources(
    dir,
    entrypoint,
    [...files, "scene/second.mtl"],
    [],
  );
  assert.match(
    await readFile(path.join(dir, entrypoint), "utf8"),
    /mtllib model.mtl\nmtllib second.mtl/,
  );
});
test("bundles glTF buffers/images as self-contained GLB and retains exact geometry bytes", async (t) => {
  const dir = await fixture(t),
    doc = {
      asset: { version: "2.0" },
      buffers: [{ uri: "C:\\old\\DATA.BIN", byteLength: 4 }],
      bufferViews: [{ buffer: 0, byteLength: 4 }],
      images: [{ uri: "../lost/PIEDRA.PNG" }],
    };
  await writeFile(path.join(dir, "model.gltf"), JSON.stringify(doc));
  await writeFile(path.join(dir, "data.bin"), "1234");
  await writeFile(path.join(dir, "piedra.png"), png);
  const result = await bundleGltf(dir, "model.gltf", [
      "model.gltf",
      "data.bin",
      "piedra.png",
    ]),
    output = validateGlb(result);
  assert.equal(output.buffers.length, 1);
  assert.equal(output.images[0].mimeType, "image/png");
  assert.equal(output.images[0].uri, undefined);
  assert.equal(readGlb(result).bin.subarray(0, 4).toString(), "1234");
});
test("missing/remote resources and URI extensions fail rather than silently losing textures", async (t) => {
  const dir = await fixture(t);
  for (const uri of [
    "gone.png",
    "https://example.com/t.png",
    "file:///etc/passwd",
  ]) {
    await writeFile(
      path.join(dir, "m.gltf"),
      JSON.stringify({ asset: { version: "2.0" }, images: [{ uri }] }),
    );
    await assert.rejects(bundleGltf(dir, "m.gltf", ["m.gltf"]));
  }
  await writeFile(
    path.join(dir, "m.gltf"),
    JSON.stringify({
      asset: { version: "2.0" },
      extensions: { TEST: { uri: "secret" } },
    }),
  );
  await assert.rejects(bundleGltf(dir, "m.gltf", ["m.gltf"]), /extensión/);
});
test("rejects ZIP symlinks, traversal, case/Unicode duplicates, bombs and multiple models", async (t) => {
  const dir = await fixture(t),
    archives = [
      await zip([["link", "target", { mode: 0o120777 }]]),
      await zip([
        ["A.obj", obj],
        ["a.obj", obj],
      ]),
      await zip([
        ["e\u0301.obj", obj],
        ["é.obj", obj],
      ]),
      await zip([["large.bin", Buffer.alloc(2 * 1024 * 1024)], ...entries]),
      await zip([...entries, ["another.stl", "x"]]),
    ];
  const bad = await zip([["safe/model.obj", obj]]);
  let offset = 0;
  while ((offset = bad.indexOf("safe/model.obj", offset)) !== -1) {
    bad.write("../x/model.obj", offset);
    offset += 14;
  }
  archives.push(bad);
  for (let i = 0; i < archives.length; i++)
    await assert.rejects(
      extractUpload(archives[i], "bad.zip", path.join(dir, String(i))),
    );
});
test("low-poly direct uploads finish asynchronously with GLB metadata and retained original", async (t) => {
  const { worker, modelsDir } = await importer(t),
    accepted = await worker.start(Buffer.from("ply"), "Nuevo.PLY");
  assert.equal(accepted.state, "processing");
  await worker.idle();
  const job = worker.current();
  assert.equal(job.state, "completed");
  assert.equal(job.model.sourceFormat, "PLY");
  assert.equal(job.model.triangles, 100);
  assert.equal(
    await readFile(path.join(modelsDir, job.id, "original-upload.PLY"), "utf8"),
    "ply",
  );
  assert.ok((await readdir(modelsDir)).includes(job.id));
});
test("only >1M triangles prompts with fixed500k target; publish waits for explicit choice", async (t) => {
  assert.equal(HIGH_TRIANGLES, 1000000);
  assert.equal(TARGET_TRIANGLES, 500000);
  for (const count of [HIGH_TRIANGLES, HIGH_TRIANGLES + 1]) {
    const { worker, modelsDir } = await importer(t, {
      modelChecker: async () => ({ triangles: count }),
    });
    const accepted = await worker.start(Buffer.from("obj"), "m.obj");
    await worker.idle();
    if (count === HIGH_TRIANGLES) {
      assert.equal(worker.current().state, "completed");
      continue;
    }
    const job = worker.current();
    assert.equal(job.state, "awaiting_decision");
    assert.equal(job.proposal.targetTriangles, 500000);
    assert.equal(job.proposal.reductionPercent, 50);
    assert.equal(
      (await readdir(modelsDir)).filter((f) => /^[a-f0-9-]{36}$/.test(f))
        .length,
      0,
    );
    await assert.rejects(
      worker.start(Buffer.from("obj"), "other.obj"),
      /curso/,
    );
    await worker.decide(accepted.id, false);
    await worker.idle();
    assert.equal(worker.current().model.triangles, count);
    assert.equal(worker.current().model.simplified, false);
    await assert.rejects(worker.decide(accepted.id, true), /decisión/);
  }
});
test("accepting reduction invokes worker once and records actual triangles", async (t) => {
  let checks = 0,
    simplified = 0;
  const { worker } = await importer(t, {
    modelChecker: async () => ({
      triangles: ++checks === 1 ? 2000000 : 499998,
    }),
    modelSimplifier: async (args) => {
      simplified++;
      assert.equal(args.targetTriangles, 500000);
      return fakeSimplifier(args);
    },
  });
  const job = await worker.start(Buffer.from("stl"), "m.stl");
  await worker.idle();
  assert.equal(worker.current().proposal.reductionPercent, 75);
  const results = await Promise.allSettled([
    worker.decide(job.id, true),
    worker.decide(job.id, true),
  ]);
  assert.equal(results.filter((r) => r.status === "fulfilled").length, 1);
  await worker.idle();
  assert.equal(simplified, 1);
  assert.equal(worker.current().model.triangles, 499998);
  assert.equal(worker.current().model.originalTriangles, 2000000);
});
test("pending consent survives restart; interrupted processing fails and cleans staging", async (t) => {
  const { worker, modelsDir } = await importer(t, {
      modelChecker: async () => ({ triangles: 1500000 }),
    }),
    job = await worker.start(Buffer.from("ply"), "m.ply");
  await worker.idle();
  await worker.close();
  const restored = await createImporter({ modelsDir });
  assert.equal(restored.current().state, "awaiting_decision");
  await restored.close();
  await writeFile(
    path.join(modelsDir, ".import-job.json"),
    JSON.stringify({ id: job.id, state: "processing" }),
  );
  const stale = await createImporter({ modelsDir });
  assert.equal(stale.current().state, "failed");
  assert.equal(
    (await readdir(modelsDir)).some((f) => f.startsWith(".import-" + job.id)),
    false,
  );
  await stale.close();
});
test("preflight failure prevents catalog publication and cleans staging", async (t) => {
  const { worker, modelsDir } = await importer(t, {
    modelChecker: async () => {
      throw new Error("invalid");
    },
  });
  await worker.start(Buffer.from("ply"), "m.ply");
  await worker.idle();
  assert.equal(worker.current().state, "failed");
  assert.deepEqual(
    (await readdir(modelsDir)).filter((f) => !f.startsWith(".")),
    [],
  );
});
test("Linux sandbox denies host/network access and bounds native converters", async (t) => {
  const dir = await fixture(t),
    cmd = await sandboxCommand(
      "/usr/bin/assimp",
      ["export", dir + "/source/a.obj", dir + "/converted/a.gltf"],
      dir,
      "linux",
    );
  assert.equal(cmd.command, "/usr/bin/bwrap");
  assert.ok(cmd.args.includes("--unshare-all"));
  assert.ok(cmd.args.includes("--cap-drop"));
  assert.ok(cmd.args.includes("--as=3221225472"));
  assert.ok(cmd.args.includes("/work/source/a.obj"));
  assert.equal(
    cmd.args.some((a, i) => a === "--ro-bind" && cmd.args[i + 1] === "/"),
    false,
  );
  const node = await sandboxCommand(process.execPath, [], dir, "linux");
  assert.ok(node.args.includes("/node"));
  assert.equal(node.args.includes("--as=3221225472"), false);
});
test(
  "real Assimp→GLB→Panda imports repaired texturedOBJ, STL, PLY; malformed geometry fails",
  { skip: !process.env.GESTUR_PYTHON || !process.env.GESTUR_NATIVE_TESTS },
  async (t) => {
    const modelsDir = await fixture(t),
      worker = await createImporter({ modelsDir });
    t.after(() => worker.close());
    for (const [filename, buffer] of [
      ["textured.zip", await zip(entries)],
      [
        "triangle.stl",
        Buffer.from(
          "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n",
        ),
      ],
      [
        "triangle.ply",
        Buffer.from(
          "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
        ),
      ],
    ]) {
      await worker.start(buffer, filename);
      await worker.idle();
      const job = worker.current();
      assert.equal(job.state, "completed", job.error);
      assert.equal(job.model.triangles, 1);
      if (filename.endsWith("zip")) {
        const result = await preflightModel(path.join(modelsDir, job.model.id));
        assert.equal(result.textures, 1);
        assert.ok(job.warnings.length > 0);
      }
    }
    await worker.start(Buffer.from("bad geometry"), "bad.obj");
    await worker.idle();
    assert.equal(worker.current().state, "failed");
  },
);

test(
  "real FBX, DAE, 3DS, OFF and X conversion",
  { skip: !process.env.GESTUR_PYTHON || !process.env.GESTUR_NATIVE_TESTS },
  async (t) => {
    const dir = await fixture(t),
      source = path.join(dir, "triangle.obj");
    await writeFile(source, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    const worker = await createImporter({
      modelsDir: path.join(dir, "models"),
    });
    t.after(() => worker.close());
    for (const [format, extension] of [
      ["fbxa", "fbx"],
      ["collada", "dae"],
      ["3ds", "3ds"],
      ["x", "x"],
    ]) {
      const exported = path.join(dir, `triangle.${extension}`);
      await runNative(
        process.env.GESTUR_ASSIMP ||
          (process.platform === "darwin"
            ? "/opt/homebrew/bin/assimp"
            : "/usr/bin/assimp"),
        ["export", source, exported, `-f${format}`],
        dir,
      );
      await worker.start(await readFile(exported), `triangle.${extension}`);
      await worker.idle();
      assert.equal(
        worker.current().state,
        "completed",
        `${extension}: ${worker.current().error}`,
      );
      assert.equal(worker.current().model.triangles, 1);
    }
    await worker.start(
      Buffer.from("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"),
      "triangle.off",
    );
    await worker.idle();
    assert.equal(worker.current().model.triangles, 1);
  },
);
test(
  "real lightweight worker simplifies geometry while retaining UVs, materials and texture pixels",
  { skip: !process.env.GESTUR_PYTHON || !process.env.GESTUR_NATIVE_TESTS },
  async (t) => {
    const dir = await fixture(t);
    assert.match(await verifyImporter(dir), /"ok":true/);
    const points = [],
      uv = [],
      faces = [],
      n = 20;
    for (let y = 0; y <= n; y++)
      for (let x = 0; x <= n; x++) {
        points.push(`v ${x / n} ${y / n} 0`);
        uv.push(`vt ${x / n} ${y / n}`);
      }
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x + 1,
          b = a + 1,
          c = a + n + 1,
          d = c + 1;
        faces.push(
          `f ${a}/${a} ${b}/${b} ${c}/${c}`,
          `f ${b}/${b} ${d}/${d} ${c}/${c}`,
        );
      }
    const model = [
      "mtllib material.mtl",
      ...points,
      ...uv,
      "usemtl stone",
      ...faces,
    ].join("\n");
    const worker = await createImporter({
      modelsDir: path.join(dir, "models"),
    });
    t.after(() => worker.close());
    await worker.start(
      await zip([
        ["grid.obj", model],
        ["material.mtl", "newmtl stone\nmap_Kd stone.png\n"],
        ["stone.png", png],
      ]),
      "grid.zip",
    );
    await worker.idle();
    assert.equal(worker.current().state, "completed", worker.current().error);
    const workspace = path.join(dir, "models", worker.current().id),
      before = readGlb(await readFile(path.join(workspace, "model.glb")));
    await simplifyModel({
      workspace,
      originalTriangles: 800,
      targetTriangles: 400,
    });
    const result = await preflightModel(path.join(workspace, "simplified.glb")),
      after = readGlb(await readFile(path.join(workspace, "simplified.glb")));
    assert.equal(result.triangles, 400);
    assert.equal(result.textures, 1);
    assert.equal(
      after.doc.meshes[0].primitives[0].attributes.TEXCOORD_0 >= 0,
      true,
    );
    const pixels = ({ doc, bin }) => {
      const view = doc.bufferViews[doc.images[0].bufferView];
      return bin.subarray(view.byteOffset, view.byteOffset + view.byteLength);
    };
    assert.deepEqual(pixels(after), pixels(before));
  },
);

test(
  "real glTF strips/fans normalize before Panda and TIFF/WebP textures convert losslessly to PNG",
  { skip: !process.env.GESTUR_PYTHON || !process.env.GESTUR_NATIVE_TESTS },
  async (t) => {
    const { default: sharp } = await import("sharp");
    const modelsDir = await fixture(t),
      worker = await createImporter({ modelsDir });
    t.after(() => worker.close());
    const positions = Buffer.from(
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]).buffer,
    );
    const tiff = await sharp(png).tiff({ compression: "lzw" }).toBuffer();
    const webp = await sharp(png).webp({ lossless: true }).toBuffer();
    for (const mode of [5, 6]) {
      const textureName = mode === 5 ? "texture.tif" : "texture.webp";
      const document = {
        asset: { version: "2.0" },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode }] }],
        buffers: [{ uri: "mesh.bin", byteLength: positions.length }],
        bufferViews: [{ buffer: 0, byteLength: positions.length }],
        accessors: [
          {
            bufferView: 0,
            componentType: 5126,
            count: 4,
            type: "VEC3",
            min: [0, 0, 0],
            max: [1, 1, 0],
          },
        ],
        images: [{ uri: `C:\\lost\\${textureName.toUpperCase()}` }],
      };
      await worker.start(
        await zip([
          ["scene.gltf", JSON.stringify(document)],
          ["mesh.bin", positions],
          [textureName, mode === 5 ? tiff : webp],
        ]),
        "strip.zip",
      );
      await worker.idle();
      const job = worker.current();
      assert.equal(job.state, "completed", job.error);
      assert.equal(job.model.triangles, 2);
      const output = readGlb(
        await readFile(path.join(modelsDir, job.model.id)),
      );
      assert.equal(output.doc.meshes[0].primitives[0].mode, 4);
      assert.equal(output.doc.images[0].mimeType, "image/png");
      const imageView = output.doc.bufferViews[output.doc.images[0].bufferView];
      const outputPixels = await sharp(
        output.bin.subarray(
          imageView.byteOffset,
          imageView.byteOffset + imageView.byteLength,
        ),
      )
        .ensureAlpha()
        .raw()
        .toBuffer();
      assert.deepEqual(
        outputPixels,
        await sharp(png).ensureAlpha().raw().toBuffer(),
      );
      assert.deepEqual(output.bin.subarray(0, positions.length), positions);
    }
  },
);
