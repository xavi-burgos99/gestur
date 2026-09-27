# Third-party notices

Gestur uses independently licensed software and recognition models. **Any license adopted for original Gestur material does not replace or restrict these upstream licenses.** The notices below, original source headers, and the complete texts referenced here must be preserved as required by their respective terms. Listing a component does not imply endorsement by its authors.

This file is an attribution/evidence index, not a new grant over third-party material, not a relicensing of all repository files, and not a completed corresponding-source offer for a preinstalled Raspberry Pi OS image. Consult the [licensing audit](docs/licensing-audit.md) for unresolved asset rights and binary-distribution work.

## Full package notices

The [dependency notice index](docs/licenses/DEPENDENCIES.md) links the original license/copyright/NOTICE files of all captured packages. The [inventory](docs/licenses/inventory.json) records package/version, dependency scope, original paths and SHA-256. Original texts are retained without rewriting in `docs/licenses/texts/`; copyright statements in those texts remain authoritative. Additional [upstream source texts](docs/licenses/upstream/sources.json) cover omissions in package archives and interpreter/bootstrap software.

The inventory includes development tools and optional platform variants as well as runtime packages. [Linux ARM64 Python wheel evidence](docs/licenses/arm64-wheel-evidence.json) and [Linux ARM64 Node archive evidence](docs/licenses/arm64-node-evidence.json) record the inspected target builds and additional notices. A package's declared top-level license does not automatically cover every embedded native library.

## Python runtime and graphics

| Component | Version in the audit | Terms and credit |
| --- | --- | --- |
| MediaPipe | 0.10.18 | Apache-2.0; Google/MediaPipe authors. The [upstream license](docs/licenses/upstream/mediapipe-0.10.18-LICENSE.txt) also retains the Lucent/Rob Pike/Ken Thompson UTF notice for the identified source portion. |
| Panda3D | 1.10.16 | Modified BSD, Copyright © 2008 Carnegie Mellon University; full installed text is linked in the package index. |
| panda3d-gltf / panda3d-simplepbr | 1.3.0 / 0.13.1 | BSD-3-Clause; preserve each package's own copyright notice. |
| NumPy / SciPy | 1.26.4 / 1.16.3 | BSD and bundled third-party notices, including OpenBLAS/LAPACK and GCC runtime terms where applicable. |
| opencv-contrib-python | 4.11.0.86 | Python packaging: MIT, Copyright Olli-Pekka Heinisuo. OpenCV: Apache-2.0. The original `LICENSE-3RD-PARTY.txt` covers FFmpeg, Qt and other bundled components. |
| protobuf | 4.25.9 | BSD-3-Clause, as distributed by its authors. |
| JAX / jaxlib | 0.7.1 | Apache-2.0 and applicable embedded component terms. |
| Matplotlib | 3.10.8 | Matplotlib license agreement and separate DejaVu/STIX font notices. |
| FlatBuffers | 25.12.19 | [Apache-2.0 license from the exact release](docs/licenses/upstream/flatbuffers-25.12.19-LICENSE.txt). |
| qrcode | 8.2 | BSD license; full copyright/conditions are in the index. |
| SentencePiece | 0.2.2 | [Apache-2.0 license from the exact release](docs/licenses/upstream/sentencepiece-0.2.2-LICENSE.txt). |

All other resolved Python dependencies and their full notices, including Pillow, CFFI, sounddevice, fonttools, ml_dtypes/Eigen, packaging and dateutil, are in the package index. This summary does not replace those texts. Font/library attributions must also accompany binary packages that contain them; the application uses FreeType functionality through its graphics/font stack.

## Web portal and model processing

React, React DOM, Mantine, Fastify and its plugins, glTF Transform, meshoptimizer, Ajv, yauzl and the listed frontend/build dependencies retain their original MIT or other stated package licenses. Exact versions and full texts are in the index.

- **Tabler Icons** / `@tabler/icons-react` 3.48.0: MIT; **Copyright (c) 2020–2026 Paweł Kuna**. The original text is preserved in the package index. This credit applies to the Tabler UI icons; the custom gesture SVGs have separately documented provenance.
- **meshoptimizer** 1.3.0: MIT; **Copyright (c) 2016–2026 Arseny Kapoulkine**. Preserve the package's complete notice with copies of its code/binary modules.
- **Sharp** 0.35.4: Apache-2.0. **libvips and its bundled libraries are independently licensed**, including LGPL components; their use is covered by those licenses, not a Gestur restriction.
- **sharp-libvips** 1.3.3, including **libvips 8.18.6**: see the [original upstream third-party table](docs/licenses/upstream/sharp-libvips-1.3.3-THIRD-PARTY-NOTICES.md), the archive's version list in the ARM64 evidence, and the [libvips license text](docs/licenses/upstream/libvips-8.18.6-LICENSE.txt). The bundle selects LGPLv3 via “or later” terms for several libraries. Copies of [LGPLv3](docs/licenses/upstream/LGPL-3.0.txt) and [GPLv3](docs/licenses/upstream/GPL-3.0.txt) are supplied. Library modification, replacement/relinking, debugging and corresponding-source rights must be preserved as applicable.
- **caniuse-lite** 1.0.30001810: CC-BY-4.0; retain its data attribution/license. No changes to the upstream package's data were made by this audit.
- **Rollup/esbuild native packages**: their enclosing exact-version [Rollup license bundle](docs/licenses/upstream/rollup-4.63.4-LICENSE.md) and [esbuild license](docs/licenses/upstream/esbuild-0.28.2-LICENSE.txt) are also supplied where the platform package omits the text.
- **react-remove-scroll-bar** 2.3.8 declares MIT; [current upstream full notice](docs/licenses/upstream/react-remove-scroll-bar-upstream-LICENSE.txt): Copyright (c) 2025 Anton Korzunov. This text was obtained from upstream, rather than represented as an original file inside the 2.3.8 npm archive.
- **abstract-logging** 2.0.1: James Sumners, MIT; [original README and license link](docs/licenses/upstream/abstract-logging-2.0.1-README.md), [exact notice from the author's linked license service](docs/licenses/upstream/abstract-logging-author-LICENSE.txt). The service automatically supplies the current year (2026 at retrieval); the preserved notice does not establish a historical copyright year for this package.

## Recognition model attribution and modification notice

Production uses Google MediaPipe **BlazePose/Pose Landmarker Lite**, **Palm Detection Lite** and **Hand Landmark Lite**, under Apache License 2.0 according to the official model cards:

- [BlazePose GHUM 3D model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf): credits Valentin Bazarevsky, Ivan Grishchenko and Eduard Gabriel Bazavan, Google.
- [Hand Tracking Lite/Full model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf).
- [MediaPipe Apache license and additional notice](docs/licenses/upstream/mediapipe-0.10.18-LICENSE.txt).

**Gestur modification notice:** `scripts/provision_models.py` adds/rewrites image input normalization metadata and output descriptions for `palm_detection_lite.tflite` and `hand_landmark_lite.tflite`, retaining neural-network weights and tensor ordering. It packages the resulting files as `hand_detector.tflite` and `hand_landmarks_detector.tflite` inside `hand_landmarker_lite.task`. Original source URLs, input hashes and modified-member hashes are recorded in [tracking_models/manifest.json](tracking_models/manifest.json). The pose task remains the verified upstream asset. Carry this notice with distributions of the modified hand bundle.

No Meta/Quest, MobRecon or MANO model is part of the production tracker. A model license is not a grant over unrelated training datasets or photographs.

## Vendored research code

The optional research directory contains HandMesh/MobRecon source from commit `3c87e958d4855f890e3884ec94bcfc0f99422c3d`, distributed under the authors' original [MIT license](scripts/research/mobrecon/source/LICENSE): **Copyright (c) 2021 chenxingyu**. Individual headers naming Xingyu Chen are preserved. The [research manifest](scripts/research/mobrecon/manifest.json) identifies exact files and hashes. These files were not modified by this audit and are not imported by the production application.

The upstream project acknowledges SpiralNet++ for SpiralConv. Its [original MIT notice](docs/licenses/upstream/spiralnet-plus-LICENSE.txt), **Copyright (c) 2019 swgong**, is also retained. References:

- Xingyu Chen et al., *MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image*, CVPR 2022.
- Shunwang Gong et al., *SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator*, ICCV Workshop 2019.

Weights, templates, regressors, dataset images and generated binary exports are not supplied in that tracked research directory. Links or hashes are not distribution permissions. In particular, any MANO-related data and derivatives remain subject to their applicable rights; Gestur does not grant a commercial or redistribution license for them.

## Interpreter, OS and native component notices

- [CPython 3.12.14 original license history](docs/licenses/upstream/python-3.12.14-LICENSE.txt); the actual python-build-standalone binary has additional component notices/build metadata.
- [Node v22.23.3 original combined notices](docs/licenses/upstream/node-22.23.3-LICENSE.txt), when the installer's upstream fallback is used.
- uv 0.12.18: [MIT](docs/licenses/upstream/uv-0.12.18-LICENSE-MIT.txt) or [Apache-2.0](docs/licenses/upstream/uv-0.12.18-LICENSE-APACHE.txt).
- [GCC Runtime Library Exception 3.1](docs/licenses/upstream/GCC-exception-3.1.txt) applies only to the covered runtime components, not arbitrary GPL code. [LGPL 2.1](docs/licenses/upstream/LGPL-2.1.txt) and the relevant full wheel notices are retained too.
- Assimp and other distribution packages keep their own copyright/NOTICE files. The [Assimp 5.4.3 source notice](docs/licenses/upstream/assimp-5.4.3-LICENSE.txt) is reference evidence; the installed apt version may differ.

For an installed OS, retain `/usr/share/doc/<package>/copyright`, `/usr/share/common-licenses` and any additional source-offer/build materials required for the exact redistributed packages. The operating system, firmware and external tools are not relicensed as Gestur. The audit's image-distribution checklist identifies incomplete corresponding-source/native evidence rather than pretending this index alone satisfies every binary obligation.

## Assets and historical rights

The capitel mesh/texture's original author and distribution permission are not established in the repository. No license for those files is created here. Likewise, the original gesture SVGs and archived generated raster images have their own provenance; this document makes no claim of exclusive copyright in generated output.

The root `LICENSE` present in historical commit `fc9fd2a` was Apache-2.0 with Open-MMLab copyright. The [original text is preserved](docs/licenses/upstream/historical-OpenMMLab-LICENSE.txt). Removing that file or adopting later terms does not revoke any valid historical grant. See the audit for scope uncertainty and academic-authorship questions.
