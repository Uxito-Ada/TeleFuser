---
name: profile-pipeline
description: Three-layer progressive pipeline profiling - stage timing, kernel analysis, and NCU deep dive. Each layer requires user confirmation before proceeding.
---

# Pipeline Profiling Skill

## Trigger

- User asks to profile ("profile xxx", "分析性能", "性能瓶颈")
- User wants timing analysis ("耗时在哪", "为什么慢")
- User mentions profiling ("性能分析", "profiling")

---

## Three-Layer Strategy

```
Layer 1: Stage Timing      → Identify bottleneck stage
Layer 2: Kernel Analysis    → Identify bottleneck kernel (isolated harness)
Layer 3: NCU Deep Dive      → Diagnose memory/compute/latency bound
```

---

## Pre-flight Checks

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
df -h .
```

---

## Layer 1: Stage Timing

**Goal:** Identify which stage consumes the most time.

**Execution:**

```bash
export TELEFUSER_PROFILE_DEBUG=true
export TELEFUSER_PIPELINE_NAME="wan21_t2v"  # Optional
CUDA_VISIBLE_DEVICES=0 python examples/<model>/<example>.py [options]

# Output: work_dirs/profiler_output/{pipeline_name}/{YYYYMMDD_HHMM}/timing.json
```

**Analyze:**

```python
import json
with open("work_dirs/profiler_output/wan21_t2v/20260402/timing.json") as f:
    report = json.load(f)
```

**🛑 LAYER 1 CHECKPOINT**

Report findings:

```
## Layer 1 Results

| Stage | Duration (ms) | Percentage |
|-------|---------------|------------|
| text_encoding | 150 | 3% |
| denoising | 4500 | 90% |
| vae_decode | 350 | 7% |

**Bottleneck:** denoising (90%)
```

Ask user:

```python
AskUserQuestion(questions=[{
    "question": "Layer 1 identified 'denoise' as bottleneck. Proceed to Layer 2?",
    "header": "Next Step",
    "options": [
        {"label": "Yes, profile bottleneck stage", "description": "Run isolated kernel analysis"},
        {"label": "Profile different stage", "description": "Specify another stage"},
        {"label": "Stop here", "description": "Stage timing sufficient"},
    ],
}])
```

---

## Layer 2: Kernel Analysis (Isolated Harness)

**Goal:** Profile bottleneck stage with minimal iterations (avoid 40+ DiT loops).

**Why Isolated Harness:**

| Aspect | Full Pipeline | Harness |
|--------|---------------|---------|
| Trace size | 100MB+ | <10MB |
| Iterations | 40+ | 1 |
| Reproducibility | Manual | Auto script |

**Step 1: Run Layer 1 (captures I/O signatures automatically)**

Layer 1 already generated `timing_io_signature.json` with tensor shapes.

**Step 2: Run isolated profiling**

```python
from telefuser.utils.stage_bench_harness import StageBenchHarness, HarnessConfig
from my_pipeline import get_pipeline

pipeline = get_pipeline()

harness = StageBenchHarness.from_signature_file(
    signature_path="work_dirs/profiler_output/wan21_t2v/20260402/timing_io_signature.json",
    stage_name="denoise",
    stage_instance=pipeline.denoise_stage,
    config=HarnessConfig(warmup=1, profile_steps=1),
)

harness.setup()
results = harness.profile()
```

**Output files:**
- `denoise_trace.json.gz` - Chrome trace (<10MB)
- `denoise_breakdown.json` - Top 50 kernels
- `profile_denoise.py` - Auto-generated reproducible script

**Analyze breakdown:**

```python
with open("denoise_breakdown.json") as f:
    breakdown = json.load(f)
# {"top_kernels": [{"name": "flash_attn_fwd", "ms": 75.0}, ...]}
```

**🛑 LAYER 2 CHECKPOINT**

Report findings:

```
## Layer 2 Results

| Kernel | Time (ms) | Percentage |
|--------|-----------|------------|
| flash_attn_fwd | 75 | 50% |
| ampere_fp16_s1688gemm | 50 | 33% |
| fused_add_rms_norm | 10 | 7% |

**Top Kernel:** flash_attn_fwd (50%)
```

Ask user:

```python
AskUserQuestion(questions=[{
    "question": "Layer 2 identified 'flash_attn_fwd' as top kernel. Proceed to Layer 3?",
    "header": "Next Step",
    "options": [
        {"label": "Yes, NCU analysis", "description": "Deep dive into kernel bottleneck"},
        {"label": "View Chrome trace", "description": "Visualize trace file first"},
        {"label": "Stop here", "description": "Kernel breakdown sufficient"},
    ],
}])
```

---

## Layer 3: NCU Deep Dive

**Goal:** Determine if kernel is memory/compute/latency bound.

**Execution:**

```bash
ncu --set full --kernel-name "flash_attn_fwd" \
    --launch-skip 10 --launch-count 3 \
    -o work_dirs/profiler_output/kernel_analysis \
    python examples/<model>/<example>.py [options]
```

**Bottleneck Diagnosis:**

| Pattern | Diagnosis | Recommendation |
|---------|-----------|----------------|
| DRAM > 80%, SM < 30% | Memory-bound | Fuse ops, reduce memory traffic, FP8 |
| DRAM < 50%, SM > 60% | Compute-bound | Reduce arithmetic, faster instructions |
| DRAM < 50%, SM < 30% | Latency-bound | Increase occupancy, check coalescing |

**🛑 LAYER 3 CHECKPOINT**

Report findings:

```
## Layer 3 Results

**Kernel:** flash_attn_fwd
**DRAM Throughput:** 85% (memory saturated)
**SM Throughput:** 12% (compute underutilized)
**Bottleneck:** Memory-bound

**Recommendations:**
1. Fuse attention with adjacent ops
2. Try FP8 attention for H100+
3. Check FlashAttn version
```

Ask user:

```python
AskUserQuestion(questions=[{
    "question": "Layer 3 complete. What next?",
    "header": "Next Step",
    "options": [
        {"label": "Apply optimizations", "description": "Implement suggested fixes"},
        {"label": "Generate report", "description": "Create summary report"},
        {"label": "End session", "description": "Have enough information"},
    ],
}])
```

---

## Final Report Template

```markdown
# Pipeline Profiling Report

## Summary
- **Model:** Wan2.2 14B T2V
- **Total Time:** 5000 ms
- **Bottleneck:** Denoising (90%)

## Layer 1: Stage Timing
| Stage | Time | % |
|-------|------|---|
| denoising | 4500ms | 90% |

## Layer 2: Kernel Analysis
| Kernel | Time | % |
|--------|------|---|
| flash_attn_fwd | 75ms | 50% |

## Layer 3: NCU
- **Bottleneck:** Memory-bound (85% DRAM)
- **Action:** Fuse ops, try FP8

## Output Files
- `timing.json`
- `timing_io_signature.json`
- `denoise_trace.json.gz`
- `denoise_breakdown.json`
- `profile_denoise.py`
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEFUSER_PROFILE_DEBUG` | Enable profiling (required) |
| `TELEFUSER_PIPELINE_NAME` | Pipeline name for output dir |
| `TELEFUSER_PROFILER_OUTPUT_DIR` | Override output directory |

**Output:** `work_dirs/profiler_output/{TELEFUSER_PIPELINE_NAME}/{YYYYMMDD_HHMM}/`

---

## Related Docs

- [Profiler Documentation](docs/en/profiler.md)
- [Stage Bench Harness](telefuser/utils/stage_bench_harness.py)