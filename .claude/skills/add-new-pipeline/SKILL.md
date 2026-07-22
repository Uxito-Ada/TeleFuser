---
name: add-new-pipeline
description: Integrate an external model or pipeline into TeleFuser while preserving upstream behavior and reusing TeleFuser's existing pipeline, stage, model-loading, configuration, example, CLI, and service interfaces. Use for new model support, new pipeline integration, or porting an upstream inference implementation.
---

# Add a New Pipeline

Treat interface compatibility and upstream parity as hard requirements. Use the repository as the API source of truth; do not copy static API templates from this skill.

## Establish the contract before editing

1. Read the upstream entry point, model definitions, checkpoint configuration, preprocessing, scheduler loop, and output handling.
2. Select the closest current TeleFuser pipeline, public example, and tests as structural baselines. Prefer a recently maintained implementation with the same task and loading pattern.
3. Read only the relevant canonical documentation:
   - `docs/en/adding_new_example.md`
   - `docs/en/adding_new_model.md`
   - `docs/en/adding_new_stage.md`
   - `docs/en/model_loading.md`
   - `docs/en/configuration.md`
   - `docs/en/service.md` when service support is in scope
4. Inventory the interfaces the integration will reuse: `BasePipeline`, `BaseStage`, `ModuleManager`, configuration dataclasses, example functions, contract/schema types, and service entry points.
5. Inventory required model-specific classes and configuration fields and map each one to upstream behavior and the selected baseline.
6. List every proposed framework-level or cross-pipeline interface, general-purpose configuration field, environment variable, loader, registry, or service schema deviation. The expected list is empty. If one is genuinely necessary, explain why the existing extension points cannot represent it and obtain user approval before adding it.

## Preserve upstream behavior

- Establish a minimal faithful path before splitting stages or optimizing operators.
- Preserve computation order, tensor shapes, conditioning paths, scheduler semantics, parameter meaning, default behavior, and output format.
- Read checkpoint metadata or configuration; never guess architecture dimensions or checkpoint-specific defaults.
- Keep preprocessing and postprocessing parity tests separate from end-to-end visual inspection.
- Do not combine integration with sparse attention, quantization, caching, refactoring, or performance tuning unless the user explicitly includes that work.

## Reuse TeleFuser interfaces

- Follow the selected baseline's constructor, `init`, stage wiring, `ModelRuntimeConfig`, `ModuleManager`, and configuration patterns.
- Put reusable model code in `telefuser/models/`, pipeline orchestration in `telefuser/pipelines/`, and model operations behind `telefuser.ops/`.
- Make `models/` import optimized kernels only through `telefuser.ops/`; never import `telefuser.kernel.triton` directly.
- Reuse existing config dataclasses and fields. Do not attach ad-hoc attributes or introduce a parallel config object.
- Reuse existing loading paths and module names when their semantics match. Do not create a second loader or registry for convenience.
- Match the nearest public example contract. Add `PPL_CONFIG`, `CONTRACT`, `get_pipeline`, `run`, and `run_with_file` only as required by the current example and service documentation.
- Keep model-specific public examples model-specific when the repository already follows that pattern; do not add runtime variant switches to avoid separate examples.

## Do not invent environment variables

- Do not add an environment variable during pipeline integration unless the user explicitly requests it or the same documented variable already serves the exact purpose.
- Use function parameters for request-scoped inputs, dataclass fields for pipeline/runtime configuration, CLI options for command-line inputs, and service schemas for API inputs.
- If a new process-level environment variable is unavoidable, obtain approval first, then document its name, scope, default, validation, and precedence and add tests for unset, valid, and invalid values.
- Never use an environment variable as a hidden fallback that changes model selection, tensor semantics, scheduling, or service behavior.

## Implement in parity-preserving increments

1. Add or adapt model loading and verify checkpoint keys and shapes.
2. Reproduce the upstream inference path with the smallest TeleFuser adapter surface.
3. Split work into stages only along existing lifecycle and data-flow boundaries.
4. Add the public example and service contract using the nearest baseline.
5. Add focused tests for configuration, loading, preprocessing, stage interfaces, output shapes, and service contract where applicable.
6. Run numerical or artifact parity against upstream before claiming completion.

Do not pause between these increments unless the user requested plan-first or stage-by-stage confirmation.

## Audit before completion

Inspect the final diff and report:

- New framework-level or cross-pipeline interfaces: expected `none`
- New general-purpose configuration fields: expected `none`
- Required model-specific classes and fields, with their upstream/baseline mapping
- New environment variables: expected `none`
- Intentional differences from upstream
- Intentional differences from the selected TeleFuser baseline
- Parity evidence and verification commands

Search the diff specifically for new `os.getenv`, `os.environ`, public definitions, dataclass fields, CLI options, and service schema fields. Treat unexplained additions as integration defects, not conveniences.
