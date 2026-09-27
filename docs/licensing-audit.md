# Gestur dependency and rights audit

Audit date: **2026-09-27**. Baseline application revision: **ef1d383** on `gestur-integrated`; the dependency inventories, original notices, hashes and historical findings below describe that audit. This is a factual engineering audit, **not a legal opinion or a guarantee of redistribution clearance**.

**Subsequent maintainer decision, 2026-09-27:** the approved **Gestur Noncommercial Attribution License 1.0**, Copyright 2024–2026 Xavier Burgos, is active in [LICENSE](../LICENSE), prospectively from the commit that introduces it. The capitel files have been removed from the current source checkout and distribution and moved to the maintainer's Downloads directory. These decisions do not rewrite the audited evidence, license third-party material, revoke valid prior grants, erase Git history, or resolve the academic-ownership questions identified below.

## Conclusion

The adopted license grants noncommercial use of **the original Gestur material that its licensor actually owns**, requires attribution, and requires separate express written permission for commercial use; [LICENSE](../LICENSE) controls its precise scope and conditions. This choice cannot replace third-party licenses, withdraw earlier grants, or grant rights over an undocumented scan. Attribution in redistributed source is different from attribution visible to every visitor using a kiosk; the adopted text specifies the applicable requirements.

The runtime is not wholly MIT/BSD/Apache: Sharp includes LGPL libraries; the ARM64 OpenCV wheel includes FFmpeg and Qt; numerical wheels include GCC runtime code with an exception. These are manageable obligations, not evidence that every Gestur file must become GPL. A preinstalled image needs a separate binary/source compliance package. The files assembled here improve evidence and notices but do not yet constitute that complete image package.

Priority conditions for public distribution, with the subsequent asset decision recorded:

1. Confirm the licensor's rights in the academic work and comply with the attribution requirements in [LICENSE](../LICENSE). Git identities and acknowledgements are not assignment agreements; adopting a license does not settle ownership.
2. **Current checkout exclusion completed:** the capitel mesh, materials, photograph/texture and user archive are no longer included. Establish their separate rights before any future redistribution. Historical copies and hashes remain evidence, not a permission grant.
3. Preserve all upstream rights and satisfy the native library/source/replacement requirements for the exact image being shipped. Do not describe the entire image as noncommercial-only.
4. Keep MobRecon weights, geometry templates/regressors, training data and example photographs out of a generally licensed/commercial release until their separate rights are resolved. They are already absent from tracked research assets.
5. Explicitly preserve any valid historical Apache grants. A new restrictive license is prospective and cannot revoke those rights.

## Scope and reproducible evidence

Reviewed `requirements.txt`, installed `.venv` metadata and license files, `portal/package.json`, the full lockfile and installed `node_modules`, tracked resources and headers, both model manifests, installers/image preparation, and reachable Git history. No packages were installed and the Raspberry Pi was not accessed or changed for this audit.

Evidence:

- [Machine-readable inventory](licenses/inventory.json): 37 installed Python distributions, of which 31 are the environment's runtime dependency closure and six are development/environment tools; 265 Node lock entries, 192 installed on the audited macOS ARM64 host. Direct dependencies, dependency edges, versions, original notice locations, hashes and platform flags are retained.
- [Human-readable dependency index](licenses/DEPENDENCIES.md): every package, declared license and links to verbatim notice files. `texts/` is content-addressed; identical license text is stored only once, while package attribution/path mappings remain in the inventory.
- [ARM64 wheel evidence](licenses/arm64-wheel-evidence.json): 14 native CPython 3.12-compatible Linux ARM64 wheels downloaded from their exact PyPI URLs, checked against PyPI SHA-256, inspected as ZIPs without installing or executing, with original notices and native binary names retained. This supplements, rather than substitutes for, the installed-environment inventory.
- [ARM64 Node evidence](licenses/arm64-node-evidence.json): exact locked Sharp, libvips, esbuild and Rollup Linux ARM64 archives, checked against lockfile integrity. This includes the bundled libvips component versions.
- [Research environment metadata](licenses/research-environment.json): all 27 pins in the separate MobRecon preparation requirements, queried from their exact PyPI version records. This is metadata, not a full binary license audit of an installed research environment.
- [Additional original texts and provenance](licenses/upstream/sources.json): upstream license URLs, dates and SHA-256, including MediaPipe, CPython, Node, uv, missing wheel/package licenses, LGPL/GPL texts and SpiralNet++.
- [Distribution notices](../THIRD_PARTY_NOTICES.md): scope, major credits, links to full texts, and the MediaPipe model modification notice.

Regenerate the **local** package inventory after installing the project's ordinary requirements and `npm ci`:

```sh
.venv/bin/python scripts/audit_licenses.py
# Run on the actual Debian/Raspberry Pi image when preparing an image release:
/opt/gestur/.venv/bin/python /opt/gestur/scripts/audit_licenses.py \
  --system --output /tmp/gestur-image-licenses
```

The script performs read-only package inspection and writes the requested output directory. It does not import the application, run package lifecycle scripts, access credentials, contact the network, or collect corresponding source. `--system` adds `dpkg-query` versions/source-package metadata and the original `/usr/share/doc/*/copyright` and `/usr/share/common-licenses` texts. It is not run by the application or installer. It requires `packaging`, already present in the environment. Archived target evidence can be reproduced by downloading the recorded URL, checking the recorded digest/integrity and reading archive members; no installation is required.

Limitations are material: only direct Python packages are fully pinned in `requirements.txt`; some transitives can change on a new installation. OS apt package versions are not pinned. A platform lock entry does not prove that package is installed or executed. Notice metadata can omit embedded code, static libraries, fonts and build tools. No whole-disk/image SBOM, corresponding-source archive, patent analysis, institutional contract review or exhaustive similarity analysis has been completed.

## Direct Python dependencies

All thirteen direct pins were found at their requested versions. The following describes top-level terms; bundled native code and fonts have additional terms.

| Distribution | Version | Top-level terms / evidence |
| --- | --- | --- |
| mediapipe | 0.10.18 | Apache-2.0, plus the Lucent UTF notice included in its upstream license |
| numpy | 1.26.4 | BSD-3-Clause; bundled notices additionally cover OpenBLAS/LAPACK, runtime libraries and other code |
| opencv-contrib-python | 4.11.0.86 | OpenCV Apache-2.0; Python wheel packaging MIT, Copyright Olli-Pekka Heinisuo; native third-party license bundle |
| protobuf | 4.25.9 | BSD-3-Clause |
| Panda3D | 1.10.16 | Modified BSD / BSD-3-Clause, Copyright Carnegie Mellon University; separate native/plugin terms |
| panda3d-gltf | 1.3.0 | BSD-3-Clause |
| panda3d-simplepbr | 0.13.1 | BSD-3-Clause |
| qrcode | 8.2 | BSD license; preserve its exact text and attribution |
| jax | 0.7.1 | Apache-2.0 |
| jaxlib | 0.7.1 | Apache-2.0 at package level; compiled XLA/runtime third parties require binary review |
| scipy | 1.16.3 | BSD-3-Clause; separate bundled numerical/runtime notices |
| matplotlib | 3.10.8 | Matplotlib's PSF-derived license; preserve the actual agreement and font notices |
| flatbuffers | 25.12.19 | Apache-2.0; this installed wheel lacks a license file, so the exact-tag upstream license is supplied separately |

The separate 14-item ARM64 inspection includes native transitives and omits pure Python wheels; its count differs from the 13 direct requirements and the 31-package runtime closure.

Runtime transitives in this environment are absl-py 2.5.0; attrs 26.1.0; cffi 2.1.1; contourpy 1.3.3; cycler 0.12.1; fonttools 4.66.0; kiwisolver 1.5.1; ml_dtypes 0.5.4; opt_einsum 3.4.0; packaging 26.3; Pillow 12.3.0; pycparser 3.0; pyparsing 3.3.3; python-dateutil 2.9.0.post0; sentencepiece 0.2.2; six 1.17.0; sounddevice 0.5.6; typing_extensions 4.16.0. The index records exact terms rather than replacing long license bodies with a guessed SPDX identifier. SentencePiece's wheel also lacks a license text; the exact-tag Apache license is supplied separately. `ml_dtypes` additionally carries Eigen notices. Matplotlib includes DejaVu and STIX font terms; these are not covered by simply saying “Matplotlib license”.

The six extra local distributions are pip, pytest, pluggy, iniconfig, Pygments and Ruff. If the environment is changed during development, the inventory's explicit `environment-only` entries are authoritative. Their presence is not a declaration that they are part of the production application. Re-run the collector for the release environment.

## Node: direct and transitive dependencies

All 265 lock entries have a declared license field. The exact resolved versions, rather than only `^` ranges, are in the inventory. Direct production packages are:

| Packages | Locked versions | Declared terms |
| --- | --- | --- |
| @fastify/cookie / multipart / static | 11.1.2 / 9.4.0 / 10.1.4 | MIT |
| @gltf-transform/core / extensions / functions | 4.5.0 each | MIT |
| @mantine/core / hooks | 9.6.2 each | MIT |
| @tabler/icons-react | 3.48.0 | MIT |
| ajv / fastify | 8.20.0 / 5.12.5 | MIT |
| meshoptimizer | 1.3.0 | MIT |
| react / react-dom | 19.3.0 each | MIT |
| sharp | 0.35.4 | Apache-2.0; libvips dependency has separate terms |
| yauzl | 3.4.0 | MIT |

Direct development dependencies: @vitejs/plugin-react 5.2.0, Vite 7.3.6, Prettier 3.9.9 and yazl 3.3.1, all declared MIT. Development tools may still be redistributed if the installed tree is included in an image; frontend bundles contain runtime code and need their notices even when `node_modules` is absent.

Lockfile license distribution: 211 MIT, 11 ISC, six BSD-3-Clause, 15 Apache-2.0, five BlueOak-1.0.0, one 0BSD, one `(MIT OR CC0-1.0)`, one CC-BY-4.0, ten LGPL-3.0-or-later, three `Apache-2.0 AND LGPL-3.0-or-later`, one `Apache-2.0 AND LGPL-3.0-or-later AND MIT`. These are **entries, not unique executed libraries**: optional packages for other operating systems are counted separately.

Points hidden by a direct-dependency-only check:

- `@img/sharp-libvips-linux-arm64` **1.3.3** bundles libvips **8.18.6** and 28 other versioned components. Its upstream table chooses LGPLv3 under the “or later” option for libvips, glib, pango, fribidi, libexif, libheif, librsvg and proxy-libintl; Cairo is MPL-2.0; imagequant is the BSD-licensed fork listed there, not automatically the GPL variant. [Original notices](licenses/upstream/sharp-libvips-1.3.3-THIRD-PARTY-NOTICES.md), [version/ARM64 archive evidence](licenses/arm64-node-evidence.json).
- `caniuse-lite` **1.0.30001810** is CC-BY-4.0 data used by tooling. Preserve its attribution/license and modification information if distributing that data; do not treat CC as the license for Gestur's software.
- glob/minimatch/minipass/path-scurry and the nested lru-cache have BlueOak-1.0.0 terms. `type-fest` offers MIT or CC0; the package's complete notices are retained.
- The esbuild and Rollup native packages do not all carry a full license file; their exact-version upstream license bundles have been copied. `react-remove-scroll-bar` declares MIT in package/README but its version archive omits the full text; the current upstream license identifying Anton Korzunov is preserved and labelled **upstream**, not falsely claimed to have been in the old archive. `abstract-logging` links the author's MIT license service from its v2.0.1 README. The [exact plain-text notice](licenses/upstream/abstract-logging-author-LICENSE.txt), identifying James Sumners, was retrieved from that service and preserved with its source and hash. The service automatically inserts the current year (2026 at retrieval); this is the author's linked notice as retrieved, not evidence of the package's historical copyright year.

## Native libraries, linking and image distribution

### Sharp/libvips

The Linux ARM64 npm archive contains `libvips-cpp.so.8.18.6`; the native Sharp addon and this shared object are separate. The bundled object also includes dependencies, so it is not enough to audit the JavaScript wrapper. Sharp's [official installation guide](https://sharp.pixelplumbing.com/install/) documents selection of platform binaries and using a custom libvips.

LGPLv3 section 4 requires notices and the GPL/LGPL texts, permits terms of choice for an application within its conditions, and requires a suitable shared-library/relinking route. It forbids restricting modification of the library portion and reverse engineering for debugging those changes. Distribution of the covered library itself still needs the applicable corresponding-source arrangement; simply linking to a project homepage is not a completed source offer. Applicable installation information may be required when distributing a user product. The supplied [LGPLv3](licenses/upstream/LGPL-3.0.txt) and [GPLv3](licenses/upstream/GPL-3.0.txt) are original FSF texts. This is a compliance task for the exact distributed build, not a reason to relicense independent Gestur code as GPL.

### OpenCV, NumPy and SciPy on ARM64

The inspected OpenCV ARM64 wheel actually includes FFmpeg `libavcodec.so.59`, `libavformat.so.59`, `libavutil.so.57`, `libswscale.so.6`, `libswresample.so.4`, Qt **5.15.16** libraries and `libqxcb`, OpenSSL **1.1** libraries, OpenBLAS, libgfortran, libvpx and X11-related libraries. Its own `LICENSE-3RD-PARTY.txt` is preserved. The package's Apache metadata is not the complete license for those binaries. Even if Gestur never opens a Qt window, copying the non-headless wheel still redistributes Qt. Do not infer that the headless wheel is installed from application behavior.

NumPy/SciPy wheel notices identify BSD OpenBLAS/LAPACK and **GPL runtime code with the GCC Runtime Library Exception**, which explicitly allows qualifying combinations with independent non-GPL modules. This exception applies to identified runtime code, not to arbitrary GPL dependencies. The same wheel notice files mention LGPL libquadmath, but the audited aarch64 binary lists contain no libquadmath object: that is a platform-generic notice, not proof of a shipped ARM64 library. Preserve notices and inspect the actual build. [GCC's original exception](licenses/upstream/GCC-exception-3.1.txt), [ARM64 wheel members and notices](licenses/arm64-wheel-evidence.json).

### Panda3D and other compiled wheels

Panda3D's own [BSD license](https://www.panda3d.org/license/) is permissive. Its [third-party manual](https://docs.panda3d.org/1.10/python/distribution/thirdparty-licenses) explicitly distinguishes optional plugins and other licenses. The target wheel contains Assimp, FFmpeg and OpenAL plugin binaries, plus `deploy_libs` Python extension binaries. The wheel only supplies a top-level Panda license among the detected notice files. That does **not** establish that all statically linked/embedded components or deployment-support modules are BSD, nor does a plugin's presence prove it is loaded.

The macOS wheel includes `libCg.dylib`; the inspected ARM64 Linux wheel does not. Cg/FMOD discussions in generic documentation must not be asserted as Raspberry Pi dependencies. FreeType, JPEG/PNG, fonts, Assimp and whichever native components are actually distributed require their own notices/terms. Use the [Assimp 5.4.3 license](licenses/upstream/assimp-5.4.3-LICENSE.txt) only for that documented source version, not as proof of an uninspected apt package's version. Review the target wheel build recipes and native dependencies before releasing a preinstalled image. JAX/XLA, MediaPipe/TFLite and other compiled wheels likewise require their embedded component notices; their metadata alone is not an exhaustive native SBOM.

### Operating system and installer

`gestur.sh` uses **uv 0.12.18** to provision **CPython 3.12.14**. The portal installer uses an existing sufficiently recent Node or falls back to **Node v22.23.3**. Their original [uv MIT/Apache](licenses/upstream/sources.json), [CPython](licenses/upstream/python-3.12.14-LICENSE.txt) and [Node combined notices](licenses/upstream/node-22.23.3-LICENSE.txt) are preserved. A uv-managed Python binary comes from python-build-standalone, not solely the CPython source tree; its distribution has additional libraries and a full-archive `PYTHON.json` with license/build metadata. [Upstream archive specification](https://raw.githubusercontent.com/astral-sh/python-build-standalone/main/docs/distributions.rst). The actual build/source bundle still needs recording for image redistribution.

Installer-requested apt components, whose exact versions depend on the chosen OS repository:

| Function | Requested packages | License boundary |
| --- | --- | --- |
| Download/bootstrap | ca-certificates, curl, rsync, python3, python3-venv, sudo, xz-utils, optional npm | Independent tools/interpreters; retain distribution copyright files if shipping them. rsync and some tools are GPL; that does not itself relicense scripts that invoke them. |
| Desktop/kiosk | xserver-xorg, xinit, openbox, x11-xserver-utils | Separate X server/window-manager programs; X11 has permissive components, Openbox GPL. |
| Graphics/system/audio libraries | mesa-utils, libgl1-mesa-dri, libglx-mesa0, libegl1, libgles2, libglib2.0-0, libsm6, libxext6, libxrender1, libportaudio2, libgomp1 | In-process/dynamic dependencies as well as utilities: mixed permissive, LGPL and GCC-exception components; inspect exact package copyrights and dependency closure. |
| Network/device | network-manager, dnsmasq-base, avahi-daemon, python3-dbus, rfkill, util-linux | Services and command/DBus interfaces, not copied source in Gestur. NetworkManager/dnsmasq and parts of the system are GPL; D-Bus bindings/system libraries have separate terms. |
| Conversion/isolation | assimp-utils, bubblewrap | Separate converter/sandbox processes; Assimp is BSD with third-party additions, bubblewrap LGPL. |

Running an independent GPL executable through normal process/DBus interfaces is different from incorporating/linking its code. Conversely, subprocess separation is not a universal loophole for arbitrary tightly coupled combined programs. Keep the distinction factual and have the chosen distribution checked. A full Raspberry Pi OS image redistributes the kernel, firmware and many packages beyond this table. Preserve the original OS licensing materials and corresponding-source mechanism; this application audit is not a license for the OS. **Downloading dependencies on the end user's device** and **shipping a prepopulated image** are different distribution scenarios.

## Recognition models

The production manifest pins three original Google assets, sizes and SHA-256:

| Asset | SHA-256 | Source and terms |
| --- | --- | --- |
| pose_landmarker_lite.task | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` | Official MediaPipe model URL in [manifest](../tracking_models/manifest.json); Apache-2.0 in the linked BlazePose model card |
| palm_detection_lite.tflite | `e9a4aaddf90dda56a87235303cf00e4c2d3fb28725f68fd88772997dac905c18` | Official MediaPipe assets bucket; Hand Tracking Lite/Full model card explicitly Apache-2.0 |
| hand_landmark_lite.tflite | `d7fde8ac11f8ce03f8663775bfc323f4fc9f2a38062b4f4efa142874ef5b2a48` | Same Lite/Full model-card evidence |

Primary evidence: [official pose download guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker), [BlazePose GHUM 3D model card, page 2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf), [official hand guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), [Hand Tracking Lite/Full model card, page 2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf). This conclusion comes from the model cards, **not** from the website footer licensing its documentation or only from MediaPipe's code license.

`scripts/provision_models.py` adds/rewrites image normalization metadata and output labels to the two hand TFLite files, then packages them as `hand_detector.tflite` and `hand_landmarks_detector.tflite` inside `hand_landmarker_lite.task`. Network weights/tensor order are not changed. This is still a modified asset distribution: carry the original Apache terms/attribution and identify the modification. The output hashes are already recorded in the manifest, and the modification notice is now in `THIRD_PARTY_NOTICES.md`.

The model cards' intended-use limitations are not an invented extra noncommercial license. The underlying GHUM research assets and training images are not separately distributed or licensed by Gestur. The runtime does not contain a Meta/Quest model, MobRecon model, MANO model, or a new proprietary hand model.

## Vendored research and research dependencies

The seven Python files under `scripts/research/mobrecon/source/` and their MIT license match **every stored SHA-256** in the manifest at upstream commit `3c87e958d4855f890e3884ec94bcfc0f99422c3d`. Existing copyright headers are retained. Credit: **Copyright (c) 2021 chenxingyu**, with individual files also naming Xingyu Chen. These files remain under their upstream license; the adopted Gestur noncommercial terms do not apply to them. [Pinned upstream license](https://raw.githubusercontent.com/SeanChenxy/HandMesh/3c87e958d4855f890e3884ec94bcfc0f99422c3d/LICENSE).

The upstream [README](https://raw.githubusercontent.com/SeanChenxy/HandMesh/3c87e958d4855f890e3884ec94bcfc0f99422c3d/README.md) acknowledges SpiralNet++ for SpiralConv. Its MIT notice, **Copyright (c) 2019 swgong**, is now preserved in [the additional original text](licenses/upstream/spiralnet-plus-LICENSE.txt). This is an attribution addition; the vendored source was not reformatted or edited.

The experiment's `transform.pkl`, `j_reg.npy`, image and DenseStack checkpoint are **not tracked**. The manifest records their external URLs and hashes only. An ONNX export does not remove the rights/conditions of its source weights or embedded geometry. Upstream MobRecon tells users to accept MANO's license for its ordinary setup. Gestur's reduced experiment avoids `MANO_RIGHT.pkl`, but this does not establish independent rights to the templates/regressor/checkpoint. [MANO's current primary terms](https://mano.is.tue.mpg.de/license.html) limit purposes and redistribution and direct commercial users to a separate license. Whether each MobRecon auxiliary artifact incorporates protected MANO material is **unresolved**, not established merely by having a public GitHub URL. Do not infer MIT coverage of all data from a code repository's license.

Research-only preparation pins include PyTorch 2.14.0, ONNX 1.23.0, ONNX Runtime 1.30.0, OpenMesh 1.2.1, scikit-learn 1.9.1 and SciPy 1.17.1. PyPI metadata declares respectively a compound permissive license expression for PyTorch, Apache-2.0, MIT, BSD-3-Clause, BSD-3-Clause and SciPy's BSD terms; complete metadata for all 27 packages is archived. The OpenMesh Python package metadata is not proof that every version of the C++ OpenMesh library has identical terms. No PyTorch/ONNX research runtime is added to the production requirements. Benchmark result JSON and telemetry do not grant rights to the underlying models/images.

## Capitel, icons, fonts and generated images

| Resource | Finding / action |
| --- | --- |
| `examples/capitel/capitell.obj`, `.mtl`, `.jpg` | Tracked example at audited baseline `ef1d383`; subsequently removed from the current checkout/distribution and moved to Downloads. Original scan/texture authorship and license were not documented and remain unresolved. The OBJ/MTL say exported by Blender 3.6.2, which establishes neither rights to the scan nor a GPL license for the resulting model. Do not assign the scan to Xavier solely because it is in his Git history. Written permission/source credit is required before claiming a redistributable license. |
| `examples/capitel/capitell.zip` | Existing untracked user file at the audit, then left untouched and unlicensed by the audit. Subsequently moved out of the checkout to Downloads at the user's request, together with the other capitel assets; it is not part of the current distribution. |
| `portal/src/MotionIcon.jsx` | Repo-native SVG geometry, described in the design notes as original drawing inspired by Hugeicons' rounded stroke style. No Hugeicons package, downloaded SVG collection, or Hugeicons license found in the dependency tree. This records provenance; it is not a legal originality/similarity guarantee. |
| Other portal icons | `@tabler/icons-react` 3.48.0 is actually used and MIT-licensed. Preserve Paweł Kuna/Tabler's original package notice. Do not describe all portal icons as wholly custom. |
| `docs/design-assets/head-base.png`, `hand-base.png` | The prompt/provenance document records generated images from the integrated image tool. They are archived design references, not current portal public assets. Generated output provenance is not a guarantee of copyrightability, exclusivity or third-party clearance. |
| Welcome ring / QR | Procedural project geometry; QR encoding uses the BSD `qrcode` dependency. No third-party logo or model is loaded for the default ring. |
| Fonts | Portal uses the component/system font stack, with no extra downloaded web-font asset identified. Panda's default font and Matplotlib's distributed font data must be included in native/font notice review, not assumed covered solely by Python package metadata. |
| Research photographs | MediaPipe fixture URLs and the MobRecon crop are recorded in research reports/manifests, not tracked as production assets. A model/code license should not automatically be applied to those photos. |

Capitel SHA-256 for the audited tracked files: OBJ `dc6b46d861b0402a54041ffbed5ee7021dcc3f3872a7cd51599e9bd4b60d76d6`; JPG `440fd275205af9831b083ef899f8decf2f66ac24243f2131e302453b9ef0acd9`; MTL `c230b6110ce3f17b6bf61a29d1593729d30c0b14ad4ba12bdaedad2ff2d13202`. These identify the audited historical files; they do not certify a license. They are retained after removal of the assets from the current checkout. No history rewrite or clearance of historical redistribution is claimed.

## Git history and academic authorship

The README inspected during the audit credits Xavier Burgos, Escola d'Enginyeria/UAB, academic year 2024/2025, supervisor Fernando Vilariño/CVC, Fran Iglesias, Fundación Épica – La Fura dels Baus and Cátedra UAB–Cruïlla. The historical README at `f200fb8` identifies the work as a TFG and refers to unspecified academic terms. It does not include an executable public license grant. All observed commit author identities are variants of Xavi Burgos; Git authorship is not evidence of exclusive copyright or an institutional assignment.

There **was** a root license in reachable history:

- `fc9fd2a131df78463cccc60e37976e10eb03d954` (2025-06-01, “mmpose test”) added a root Apache-2.0 license headed **Copyright 2018–2020 Open-MMLab** and a 1,999-file tree dominated by MMPose. That snapshot also contains root custom-looking `head_viewer.py` and `main.py` alongside upstream files.
- `08bb72e` (2025-06-24, “mediapipe version”) removed the root `LICENSE`.
- Original license Git blob: `b712427afe4978c6084580f113cdc87f77564fd9`, preserved in [historical-OpenMMLab-LICENSE.txt](licenses/upstream/historical-OpenMMLab-LICENSE.txt).
- No path in the audited checkout had the same blob as the same path in that historical snapshot. This **does not prove** there are no derivatives, renamed files or retained portions. A scan of current project Python/JS headers found the explicit vendored HandMesh headers, not a current OpenMMLab header; that is not an exhaustive originality review.

The surrounding snapshot strongly indicates an upstream import, but a root license may have been understood to cover contributions in that revision. Do not assert that it was limited to MMPose without further evidence. Apache grants validly received for historical material continue under their own terms; removing the file and later adopting a noncommercial license does not retroactively revoke them. A maintainer can only license rights they hold, and must preserve relevant upstream notices on retained/derived portions. No copyright assignment, UAB agreement, CLA or institutional permission document was found in the tracked repository. Ownership and any academic/funding obligations need confirmation from the author/institution rather than inference.

## License choice and “attribution always”

The comparison below records the options considered during the audit. The maintainer subsequently approved the custom **Gestur Noncommercial Attribution License 1.0** now active in [LICENSE](../LICENSE); the standard licenses below are not alternative grants for original Gestur material.

| Option | Fits free noncommercial use? | Commercial consent only? | Attribution consequence |
| --- | --- | --- | --- |
| Unmodified PolyForm Noncommercial 1.0.0 | Yes, within its definitions | Commercial purposes generally outside the grant, but there is an explicit broad grant for listed noncommercial organizations regardless of funding | Recipients of copies receive the license and `Required Notice:` lines; it does **not** require a visible credit on every kiosk/web session |
| Custom noncommercial software license | Can state the exact intended scope | Can require a separate express written grant and define institutions, paid exhibitions, sponsorship, client services and internal business use | Can specify persistent visible credit, an About screen, documentation and modification requirements; requires coherent drafting and review |
| MIT / Apache / BSD | Yes | No: commercial permission already granted | Preserve notices on copies; no universal on-screen credit requirement |
| GPL / AGPL | Yes | No: commercial use is permitted | Copyleft/source obligations are different from requiring author consent for commercial use |
| CC BY-NC for software | Has a noncommercial restriction | Depends on its defined scope | CC itself recommends against its licenses for software; can be appropriate for separately owned non-code assets, not a substitute for software-specific linking/patent rules |

Primary texts: [PolyForm NC](https://polyformproject.org/licenses/noncommercial/1.0.0), [PolyForm source/branding instructions](https://github.com/polyformproject/polyform-licenses), [Creative Commons software FAQ](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software), [OSI definition, especially field-of-endeavour neutrality](https://opensource.org/osd). A noncommercial-only license is **source-available, not OSI open source**. Modifying PolyForm's terms requires removing its name/URL according to that project's instructions; do not silently bolt a custom display requirement onto text presented as the unchanged standard license.

The audit distinguished credit in copies, public installations, the administration portal, the viewer and every frame, because these are materially different requirements. The adopted [LICENSE](../LICENSE) now states the chosen attribution obligations; any separate commercial permission must also address attribution under its agreed terms. Third-party code, models, fonts, OS components, independently licensed assets and valid prior grants remain outside the new Gestur restrictions. Crediting Xavier does not claim ownership of those components or their authors' endorsement.

## Remaining release work

The source checkout contains the adopted Gestur license, third-party notices and a repeatable package inventory; the capitel assets have been excluded from the current distribution. Before releasing binaries/images: freeze the actual target dependencies; collect the target OS/native build notices and required corresponding sources; provide the LGPL library replacement/relinking/install route where applicable; verify licenses and required attribution remain accessible alongside distributed frontend bundles; and confirm academic rights. The scan's rights must be resolved before any future redistribution, including redistribution from historical revisions. The original audit did not alter the application, licenses, user assets, dependencies, or running Pi; subsequent license adoption and local asset removal are recorded above as separate maintainer decisions. Neither decision completes the native/image compliance work or changes the preserved historical evidence.
