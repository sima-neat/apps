# Source Of Truth

Use the current Neat Development Environment's packaged source and installed
public surface before writing or changing application code. This skill is
guidance, not the API contract.

## API Contract Order

1. Packaged Neat Library source in the Neat Development Environment:
   - `/neat-resources/core-src`
2. Installed public headers:
   - `$SYSROOT/usr/include`
   - `/opt/toolchain/aarch64/modalix/usr/include`
   - target-device `/usr/include`
3. Local Neat docs in the packaged core source:
   - `docs/develop-apps/development-workflow/`
   - `docs/reference/`
   - `tutorials/`

## Implementation Starting Point

After the artifact, input owner, and API family are known, inspect the closest
current Apps example before writing application code:

   - `/neat-resources/apps-src/examples`
   - `https://github.com/sima-neat/apps/tree/main/examples`

Use its project structure, configuration, graph composition, build commands,
and runtime patterns. If it disagrees with the current installed headers or
docs, trust the installed Neat Library contract.

## Headers To Check

- Umbrella include: `include/neat.h`
- Classic model API: `include/model/Model.h`
- Graph composition: `include/pipeline/Graph.h`
- Runtime handle: `include/pipeline/Run.h`
- Tensor and sample contracts:
  - `include/pipeline/Tensor.h`
  - `include/pipeline/TensorCore.h`
  - `include/pipeline/TensorTypes.h`
- Diagnostics:
  - `include/pipeline/NeatError.h`
  - `include/pipeline/GraphReport.h`
- GenAI:
  - `include/neat/genai.h`
  - `include/genai/GenAIModel.h`
  - `include/genai/GenAIServer.h`
  - `include/genai/GenAITypes.h`
  - `include/genai/VisionLanguageModel.h`
  - `include/genai/ASRModel.h`
  - `include/genai/GraphFragments.h`

## Docs To Check

- `development-workflow/index.md`: end-to-end application loop.
- `development-workflow/overview.mdx`: direct model run vs graph/run.
- `development-workflow/model.mdx`: `Model`, model specs, and route fragments.
- `development-workflow/graph.mdx`: graph boundaries, `add`, `connect`, named endpoints.
- `development-workflow/core_types.mdx`: `Tensor` and `Sample`.
- `development-workflow/genai-model/index.mdx`: GenAI application entry point.
- `development-workflow/genai-model/direct-api.mdx`: direct model requests, results, and graph fragments.
- `development-workflow/genai-model/genai-server.mdx`: HTTP serving, lifecycle, and network boundary.
- `development-workflow/node.mdx`: public nodes and node groups.
- `development-workflow/pipeline.mdx`: built pipeline/runtime view.
- `advanced-concepts/application-design/graphs.md`: graph composition details.
- `advanced-concepts/application-design/video_sender.md`: video output patterns.
- `advanced-concepts/application-design/metadata_sender.md`: metadata output patterns.
- `advanced-concepts/data-model-contracts/data_formats.md`: data and media format contracts.
- `reference/`: generated C++ and Python API reference.

## Verification Rule

Before making a factual claim about an API, search the current source:

```bash
rg -n "Model|Graph|Run|GenAIModel|GenAIServer" /neat-resources/core-src/include /neat-resources/core-src/docs
```

Use the exact path available in the Neat Development Environment. If
`/neat-resources/core-src` is absent, inspect the installed headers and tell the
user what source was used.

For broader API discovery, read `api-surface-map.md` and search the exact header
families listed there.
