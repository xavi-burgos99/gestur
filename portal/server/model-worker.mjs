// Run only as a disposable process with a bounded heap and no network or host data.
import { readFile, writeFile } from "node:fs/promises";
import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { simplify, weld } from "@gltf-transform/functions";
import { MeshoptSimplifier } from "meshoptimizer";
import sharp from "sharp";

sharp.concurrency(2);
sharp.cache(false);
const [operation, source, destination, ratioText] = process.argv.slice(2);
try {
  if (operation === "check") {
    await MeshoptSimplifier.ready;
    await sharp({
      create: { width: 1, height: 1, channels: 4, background: "#ffffff" },
    })
      .png()
      .toBuffer();
    console.log(JSON.stringify({ ok: true }));
  } else if (operation === "image") {
    const input = await readFile(source);
    // Raster input only: SVG/PDF can reference external resources.
    if (/^\s*(?:<\?xml|<svg|%PDF)/i.test(input.subarray(0, 1024).toString()))
      throw new Error("Unsupported image");
    const output = await sharp(input, { limitInputPixels: 67108864 })
      .png()
      .toBuffer();
    await writeFile(destination, output);
  } else if (operation === "simplify") {
    const ratio = Number(ratioText);
    if (!Number.isFinite(ratio) || ratio <= 0 || ratio >= 1)
      throw new Error("Invalid ratio");
    await MeshoptSimplifier.ready;
    const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
    // readBinary consumes only this already-contained GLB, never fetch URLs.
    const document = await io.readBinary(await readFile(source));
    await document.transform(
      weld(),
      simplify({
        simplifier: MeshoptSimplifier,
        ratio,
        error: 0.005,
        lockBorder: true,
      }),
    );
    await writeFile(destination, await io.writeBinary(document));
  } else throw new Error("Invalid operation");
} catch {
  console.error("GESTUR: no se pudo procesar el modelo o la textura.");
  process.exitCode = 1;
}
