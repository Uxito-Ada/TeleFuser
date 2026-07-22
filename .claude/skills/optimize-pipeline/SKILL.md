---
name: optimize-pipeline
description: Optimize an existing TeleFuser pipeline using measured bottlenecks and current repository-supported ops, parallelism, caching, quantization, compilation, or offload mechanisms. Use for latency, throughput, GPU memory, OOM, multi-GPU, or inference-performance work after a correct baseline exists.
---

# Optimize a Pipeline

Require a correct, reproducible baseline before changing performance behavior. Read current implementations and canonical docs rather than relying on hard-coded speedup or memory estimates.

## Define the target

Record the workload, model/checkpoint, task, shape, frame count, step count, dtype, attention backend, devices, warmup, measured latency/throughput, peak memory, and quality or parity criterion. Distinguish latency, throughput, capacity, and output cadence; they require different choices.

If no baseline exists, profile first. Use `.claude/skills/profile-pipeline/SKILL.md` and `docs/en/profiler.md`.

## Reuse supported mechanisms

Inspect the closest model and pipeline before choosing an optimization. Consult the relevant current docs:

- `docs/en/ops.md` and `docs/en/attention.md`
- `docs/en/parallel.md`
- `docs/en/offload.md`
- `docs/en/feature_cache.md`
- `docs/en/torch_compile_compatibility.md`
- `docs/en/configuration.md`

Apply these constraints:

- Route model operations through `telefuser.ops`; do not import Triton kernels directly from `models/`.
- Preserve exact semantics when replacing an op: layout, normalization, RoPE, masks, causal behavior, scale, dtype, and numerical tolerances must match.
- Use current `AttentionConfig`, `ModelRuntimeConfig`, `ParallelConfig`, `FeatureCacheConfig`, `CompileConfig`, `QuantConfig`, and `OffloadConfig` APIs as implemented in the repository.
- Confirm that the target model and stage implement the selected parallel or optimization path. A config field existing does not prove model support.
- Treat sparse attention, feature caching, quantization, distillation, and approximate computation as behavior or quality changes; require explicit user agreement and parity/quality evidence.
- Do not add a new public interface, configuration system, environment variable, loader, or parallel abstraction as part of optimization without first demonstrating a missing extension point and obtaining approval.

## Work one hypothesis at a time

1. Identify the dominant measured bottleneck.
2. State the optimization hypothesis and expected mechanism without promising a fixed speedup.
3. Make the smallest isolated change.
4. Re-run correctness and performance measurements with the same workload.
5. Keep the change only if the evidence meets the stated target without violating the quality criterion.
6. Record regressions and interactions before combining optimizations.

Do not present generic ratios such as "2x CFG speedup" or fixed VRAM savings as project facts. Report only measurements from the actual hardware and workload, or clearly label external estimates as unverified.

## Validate

Run checks proportional to the change:

- Unit and contract tests for changed code paths
- Numerical comparison against the baseline for exact optimizations
- Quality evaluation for approximate optimizations
- Warmed latency distribution, throughput, and peak allocated/reserved memory
- Multi-GPU correctness, process cleanup, and communication behavior when applicable
- `ruff check`, focused pytest, and `git diff --check`

Report the baseline and optimized results side by side, the exact configuration, any quality trade-off, unsupported paths not tested, and all new interfaces/configuration/environment variables. Those final three lists should normally be empty.
