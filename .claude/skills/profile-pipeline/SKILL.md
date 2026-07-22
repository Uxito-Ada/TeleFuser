---
name: profile-pipeline
description: Profile a TeleFuser pipeline progressively from stage timing to isolated kernel analysis and optional Nsight Compute diagnosis. Use when investigating latency, throughput, memory, output cadence, slow stages, or GPU kernel bottlenecks before optimization.
---

# Profile a Pipeline

Use the current profiler implementation and `docs/en/profiler.md` as the API and command source of truth. Do not reproduce profiler helpers or create new profiling environment variables.

## Capture a reproducible workload

Record the commit, model/checkpoint, task, input shape, frame count, diffusion steps, dtype, attention backend, devices, warmup, synchronization points, and whether compilation or caches are warm. Define the metric that matters: latency distribution, throughput, output cadence, or peak memory.

Check GPU availability and memory before profiling. Avoid concurrent workloads that invalidate the trace.

## Progress from broad to narrow

1. **Stage timing:** use the existing profiler flags and pipeline metrics documented in `docs/en/profiler.md`. Identify the stage dominating the target metric.
2. **Isolated stage analysis:** use the repository's `StageBenchHarness` and captured I/O signature when the stage can be reproduced faithfully. Verify that the harness input shapes and runtime state match the full pipeline.
3. **Kernel analysis:** inspect the PyTorch/Chrome trace and kernel breakdown for the isolated bottleneck. Account for synchronization, communication, memory copies, and launch overhead rather than ranking kernels by duration alone.
4. **NCU deep dive:** use Nsight Compute only when a specific reproducible kernel question remains. Keep launch counts small and document the selected kernel and metric set.

Stop when the evidence is sufficient to select or reject an optimization. Do not require a user checkpoint between layers unless the user requested stage-by-stage confirmation or the next layer is materially expensive.

## Interpret cautiously

- Compare warmed runs using identical inputs and configuration.
- Separate CPU launch overhead, GPU execution, communication, transfers, and synchronization.
- Do not infer resource groups from device placement when stages can overlap.
- For streaming, compare p95 chunk latency with the media duration represented by a chunk and retain transport/encoding margin.
- Do not assign a memory-bound or compute-bound diagnosis from one utilization percentage alone; use the relevant throughput, occupancy, stalls, and launch context.
- Do not claim a speedup until the proposed change is benchmarked on the same workload.

## Report

Provide the workload and environment, stage timing, dominant kernels or operations, evidence-backed diagnosis, uncertainty, raw artifact paths, and the smallest next experiment. Keep profiling-only work non-mutating unless the user also requested optimization.
