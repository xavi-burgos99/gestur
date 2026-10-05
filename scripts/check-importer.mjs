#!/usr/bin/env node
// Exercise the installed converter and its isolation without touching the catalog.
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  convertModel,
  verifyImporter,
} from "../portal/server/native-model.mjs";
import { preflightModel } from "../portal/server/model-check.mjs";

const workspace = await mkdtemp(path.join(os.tmpdir(), "gestur-import-check-"));
try {
  await verifyImporter(workspace);
  await mkdir(path.join(workspace, "source"));
  await writeFile(
    path.join(workspace, "source", "triangle.obj"),
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
  );
  await convertModel({
    workspace,
    source: "triangle.obj",
    files: ["triangle.obj"],
    warnings: [],
  });
  const result = await preflightModel(path.join(workspace, "model.glb"));
  if (result.triangles !== 1)
    throw new Error(
      "La prueba de conversión no produjo el triángulo esperado.",
    );
  console.log(
    "Importador comprobado: Assimp, meshoptimizer, texturas, aislamiento y visualizador.",
  );
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
} finally {
  await rm(workspace, { recursive: true, force: true });
}
