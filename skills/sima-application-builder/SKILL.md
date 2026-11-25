---
name: sima-application-builder
description: Build or change a Neat application with the public C++ or Python APIs. Use for a `Model` on inputs the application supplies, a `Graph` over camera, file, or RTSP sources, multi-stream or multi-model graphs, Insight output, GenAI models, and Docker packaging.
---

# SiMa Application Builder

Build against the installed Neat Library. Its headers and Python bindings are the callable
contract, and Core docs and source at the same version explain the behavior. This skill teaches
the public APIs. The project's own rules decide layout, configuration, and packaging.

## Steps

1. **Locate the installed contract.** Run `neat` (or `neat --json`) where the application will
   run. It reports the installed Core, PyNeat, Scheduler, Runner, and LLiMa versions. In the
   Palette SDK (also called the Neat Development Environment), the SDK image decides which Core
   its sysroot holds: `/etc/sdk-release` reports `Neat Core = bundled` and the version in
   `Neat Core Requested`, or `Neat Core = not bundled` when the image holds no Neat Library to
   build against. The DevKit must run the same Core the SDK builds against. Headers live under
   `$SYSROOT/usr/include` in the Palette SDK and under `/usr/include` on the DevKit. Core source
   and docs live under `/neat-resources/core-src` in Core-bundled images. Otherwise use the Core
   source at the installed version's tag. To find the header that owns an API, use the
   [API surface map](references/api-surface-map.md). For Python, read the binding in Core's
   `python/src/module.cpp` or run `help(pyneat.<Name>)` on the target. Done when every class,
   function, and option field the planned code names has been found in its header or binding,
   and each one that could not be found is listed.

2. **Open a starting point.** For a focused change, start from the existing application. For a
   new one, open the closest Apps example: under `/neat-resources/apps-src/examples` in
   Core-bundled images, otherwise the public
   [Apps examples](https://github.com/sima-neat/apps/tree/main/examples). Two examples show the
   multi-stage patterns: `object-detection/high-density-multi-stream-object-detector` runs many
   RTSP streams on one shared model, and `object-detection/rfdetr-detection-segmentation` runs
   two models with host work between them. Where an example and the installed headers
   disagree, follow the headers. Read example code to adapt it: the application builds only
   against the installed Neat Library's headers and Python bindings, and never compiles,
   includes, or imports an example's source by path. Done when the starting code is open and
   either matches the installed Core (it came from the image's `/neat-resources/apps-src`, or
   from `sima-cli neat install apps`, which installs the Core it needs) or every API it calls
   has passed the step 1 check.

3. **Choose the API shape.** Classic models load a compiled model archive
   (`*_mpk.tar.gz`), not ONNX. GenAI models load a deployed LLiMa model directory.

   | The application | Use | Read |
   |---|---|---|
   | Supplies images or tensors and waits for each result | `Model::run(...)` | [Model](references/model.md) |
   | Supplies inputs and pushes and pulls many of them | `Model::build(...)`, which returns a `Model::Runner` | [Model](references/model.md) |
   | Lets Neat own a camera, file, or RTSP source, or chains several stages, streams, or models | `Graph`, whose `build(...)` returns a `Run` | [Graph topologies](references/graph-topologies.md), and the "Construct and configure", "Detection settings", and "Read outputs" sections of [Model](references/model.md) |
   | Runs an LLM, VLM, or ASR model, or serves one over HTTP | `GenAIModel`, `VisionLanguageModel`, `ASRModel`, or `GenAIServer` | [GenAI](references/genai.md) |

   Done when the artifact type and the owner of the input (the application or the `Graph`) are
   known and one row is chosen.

4. **Implement.** C++ includes `<neat.h>`. Python imports `pyneat`, and `pyneat.genai` for
   GenAI. Python spells functions in snake case (`pyneat.nodes.input`,
   `pyneat.groups.rtsp_encoded_input`), and names `Model::Options` and `Model::Runner`
   `pyneat.ModelOptions` and `pyneat.ModelRunner`. A long-running application handles `SIGINT`,
   `SIGTERM`, and `SIGHUP` by setting a stop flag that every pull loop checks between pulls,
   and closes every `Run` and `Model::Runner` on every exit path. For
   Insight video and metadata or a local window, read
   [application output](references/application-output.md). For Docker, read
   [containers](references/containers.md). For the C++ CMake file and the configure and build
   commands, read the C++ build section of [validation](references/validation.md). Done when
   the application builds against the installed headers (C++), or, in Python, when every options object it uses has been
   constructed and every field set on the target without an error. An import alone does not
   check attribute names. Check them with a short script that builds the options and graph
   objects, or with `dir()` and `help()` on each object.

5. **Validate.** Choose the checks with [validation](references/validation.md), run the
   application on the target, and inspect its output. Done when the output matches the
   application's contract on the target, or each check that could not run is named with its
   missing prerequisite. For Insight output, that means `MetadataSender.stats()` shows
   `datagrams_sent` rising on every channel and the Insight page shows video and boxes on every
   stream. Return the commands, the observed results, and those gaps.

## Related work

- Measuring or raising throughput, lowering latency, and removing copies belong to the
  `sima-application-optimizer` skill.
- Compiling a model belongs to the Model Compiler or LLiMa documentation. To download a model
  that is already compiled, follow [Model](references/model.md).
- A PCIe host application uses the `simaai::neat::pcie` and `pyneatpcie` APIs and the
  `neat-pcie-application-builder` skill.
- Insight viewer setup and device operations belong to the Insight and Neat Development
  Environment documentation.
