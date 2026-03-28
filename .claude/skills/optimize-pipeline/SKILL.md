---
name: optimize-pipeline
description: Three-phase pipeline optimization - reuse TeleFuser ops, multi-GPU inference, and custom optimization. Each phase ensures correctness before proceeding.
---

# Pipeline Optimization Skill

## Trigger Conditions

- User asks to optimize a pipeline ("优化 xxx pipeline", "speed up inference")
- User mentions performance issues ("OOM", "too slow", "out of memory")
- User wants to enable parallel inference ("multi-gpu", "distributed")
- User wants to reduce memory usage ("reduce vram", "fit in 24GB")

---

## Three-Phase Optimization Strategy

```
Phase 1: Reuse TeleFuser Ops          Phase 2: Multi-GPU Inference          Phase 3: Custom Optimization
┌─────────────────────────────┐       ┌─────────────────────────────┐       ┌─────────────────────────────┐
│ • Replace attention          │       │ • CFG Parallel              │       │ • Bottleneck analysis       │
│ • Replace normalization      │  ──►  │ • Sequence Parallel         │  ──►  │ • Custom solutions          │
│ • Replace FFN                │       │ • FSDP                      │       │ • New technique integration │
│ • Verify: Logic + Output     │       │ • Feature Cache             │       │ • Experimental features     │
│ • NO behavior change         │       │ • CPU Offload               │       │ • User-driven optimization  │
└─────────────────────────────┘       └─────────────────────────────┘       └─────────────────────────────┘
        ↓ Must Pass                          ↓ User Confirms                         ↓ User Requests
```

---

## Phase 1: Reuse TeleFuser Optimized Ops

### Goal
Replace standard PyTorch ops with TeleFuser optimized implementations while ensuring:
1. **Logic Consistency** - Same computational logic
2. **Model Consistency** - Same model architecture
3. **Output Consistency** - Numerically identical results

### CRITICAL: No Behavior Change

This phase ONLY replaces implementation details, NOT the algorithm:
- ✅ Replace `nn.LayerNorm` → `ops.normalization.LayerNorm`
- ✅ Replace `F.scaled_dot_product_attention` → `ops.attention.attention`
- ✅ Replace custom FFN → `ops.ffn.FeedForward`
- ❌ DO NOT change attention type (dense → sparse)
- ❌ DO NOT change model architecture
- ❌ DO NOT enable new features (cache, parallel)

### Available Optimized Ops

| Original | TeleFuser Replacement | Speedup | Notes |
|----------|----------------------|---------|-------|
| `nn.LayerNorm` | `ops.normalization.LayerNorm` | 1.2-1.5x | Triton kernel available |
| `nn.RMSNorm` | `ops.normalization.RMSNorm` | 1.3-1.8x | Triton kernel available |
| `F.scaled_dot_product_attention` | `ops.attention.attention` | 1.5-3x | FlashAttn 2/3/4, SageAttn |
| Custom FFN | `ops.ffn.FeedForward` | 1.2x | Fused ops |
| Custom SwiGLU | `ops.activations.SwiGLU` | 1.1x | Fused kernel |

### Process

#### Step 1: Identify Current Ops

Read the model file and identify ops that can be replaced:

```python
# Check for replaceable patterns
# Example: telefuser/models/flux2_dit.py

# Current (Phase 3 of add-new-pipeline):
class Flux2Attention(nn.Module):
    def forward(self, ...):
        # Uses F.scaled_dot_product_attention or custom attention

# Should check:
# 1. Is normalization using nn.LayerNorm or custom?
# 2. Is attention using standard SDPA?
# 3. Is FFN using standard nn.Linear?
```

#### Step 2: Check Op Compatibility

| Op Type | Check | Action |
|---------|-------|--------|
| Attention | QK normalization? RoPE? | Use `AttentionConfig` with matching settings |
| Normalization | affine? eps? | Match exact parameters |
| FFN | activation type? | Use matching activation |

#### Step 3: Replace Ops

**Example: Attention Replacement**

```python
# Before (Phase 3 style - standard PyTorch)
class Flux2Attention(nn.Module):
    def __init__(self, ...):
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim)
        self.norm_k = nn.RMSNorm(dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        # ... compute Q, K, V
        # Apply RoPE manually
        # Use F.scaled_dot_product_attention
        pass

# After (Phase 1 optimized - TeleFuser ops)
from telefuser.ops.attention import attention
from telefuser.ops.normalization import RMSNorm
from telefuser.core.config import AttentionConfig, AttnImplType

class Flux2Attention(nn.Module):
    def __init__(self, ...):
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim)  # TeleFuser RMSNorm
        self.norm_k = RMSNorm(dim)  # TeleFuser RMSNorm
        self.to_out = nn.Linear(dim, dim)

        # Attention config for optimized implementation
        self.attention_config = AttentionConfig.dense_attention(
            AttnImplType.FLASH_ATTN_2
        )

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        # ... compute Q, K, V
        # Apply RoPE (same logic, possibly Triton kernel)

        # Use TeleFuser attention
        hidden_states = attention(
            query=q,
            key=k,
            value=v,
            attention_config=self.attention_config,
            # ... other params
        )
        return hidden_states
```

**Example: Normalization Replacement**

```python
# Before
self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

# After
from telefuser.ops.normalization import LayerNorm

self.norm1 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
```

#### Step 4: Verify Output Consistency

**CRITICAL: Must verify numerical consistency before proceeding.**

```python
# Verification script
import torch

def verify_output_consistency(model_original, model_optimized, input_args):
    """Verify optimized model produces identical output."""

    model_original.eval()
    model_optimized.eval()

    with torch.no_grad():
        # Run both models
        out_original = model_original(**input_args)
        out_optimized = model_optimized(**input_args)

        # Check numerical difference
        diff = (out_original - out_optimized).abs().max().item()
        relative_diff = diff / out_original.abs().max().item()

    print(f"Max absolute diff: {diff:.2e}")
    print(f"Max relative diff: {relative_diff:.2e}")

    # Tolerance: 1e-5 for FP16, 1e-6 for BF16
    tolerance = 1e-5 if input_args.get('dtype') == torch.float16 else 1e-6

    if relative_diff < tolerance:
        print("✅ Output consistent - optimization verified")
        return True
    else:
        print("❌ Output differs - check implementation")
        return False
```

### 🛑 PHASE 1 CHECKPOINT

After completing Phase 1:

```
## Phase 1 Completed ✅

**Ops Replaced**:
| Op | Original | TeleFuser | Verified |
|----|----------|-----------|----------|
| Attention | F.scaled_dot_product_attention | ops.attention.attention (FlashAttn2) | ✅ |
| Normalization | nn.LayerNorm | ops.normalization.LayerNorm | ✅ |
| Normalization | nn.RMSNorm | ops.normalization.RMSNorm | ✅ |

**Verification Results**:
- Max absolute diff: 1.2e-6
- Max relative diff: 3.4e-7
- ✅ Output consistent within tolerance

**Expected Speedup**: 1.5-2x (FlashAttn2 + Triton kernels)

Ready for Phase 2 (Multi-GPU Inference)?
```

Call AskUserQuestion:
```python
AskUserQuestion(questions=[
    {
        "question": "Phase 1 ops optimization verified. Ready for Phase 2 (Multi-GPU inference)?",
        "header": "Proceed?",
        "options": [
            {"label": "Yes, continue to Phase 2", "description": "Explore parallel inference, feature cache, offload"},
            {"label": "No, stay in Phase 1", "description": "Need to review or fix the ops replacement"},
        ],
        "multiSelect": False,
    }
])
```

---

## Phase 2: Multi-GPU Inference Optimization

### Goal
Apply multi-GPU optimizations based on model size and hardware configuration:
- **CFG Parallel** - Parallel positive/negative prompt computation
- **Sequence Parallel** - Ulysses/Ring attention for long sequences
- **FSDP** - Fully Sharded Data Parallel for memory efficiency
- **Feature Cache** - AdaTaylor cache for acceleration
- **CPU Offload** - Memory reduction through weight offloading

### Decision Required: User Must Confirm

This phase changes behavior and requires user decisions:
- Memory vs Speed trade-off
- Number of GPUs to use
- Quality vs Speed for feature cache

### Model Size → Strategy Mapping

| Model Size | Single GPU (24GB) | Single GPU (80GB) | Multi-GPU |
|------------|-------------------|-------------------|-----------|
| < 3B | ✅ Direct | ✅ Direct | Optional speedup |
| 3-10B | ⚠️ Offload | ✅ Direct | ✅ Recommended |
| 10-20B | ⚠️ Offload + Cache | ⚠️ May need offload | ✅ Recommended |
| > 20B | ❌ May not fit | ⚠️ Offload needed | ✅ Required |

### Configuration Options

#### 2.1 CFG Parallelism

**When**: `cfg_scale > 1` (classifier-free guidance enabled)
**Effect**: 2x speedup when using 2 GPUs for CFG
**Trade-off**: Requires 2 GPUs

```python
# Before
for t in timesteps:
    noise_uncond = model(latents, uncond_embeds)
    noise_cond = model(latents, cond_embeds)
    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

# After (2 GPUs, CFG parallel)
# GPU 0: noise_uncond = model(latents, uncond_embeds)
# GPU 1: noise_cond = model(latents, cond_embeds)
# Both run in parallel

config = ParallelConfig(
    device_ids=[0, 1],
    cfg_degree=2,
)
```

#### 2.2 Sequence Parallelism (Ulysses)

**When**: Long sequences (video, high-res images)
**Effect**: Linear speedup with GPU count
**Trade-off**: Communication overhead, requires divisible head count

```python
# 2 GPUs, Ulysses SP
config = ParallelConfig(
    device_ids=[0, 1],
    sp_ulysses_degree=2,
)

# 4 GPUs, CFG + Ulysses
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

#### 2.3 Ring Attention

**When**: Very long sequences (long video, 100+ frames)
**Effect**: Supports arbitrary sequence length
**Trade-off**: Higher communication overhead

```python
# For very long videos
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    sp_ring_degree=4,
)
```

#### 2.4 FSDP (Fully Sharded Data Parallel)

**When**: Memory-constrained multi-GPU setup
**Effect**: Reduces memory per GPU by sharding model
**Trade-off**: Communication overhead

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    enable_fsdp=True,
)
```

#### 2.5 Feature Cache (AdaTaylor)

**When**: Speed is priority, minor quality loss acceptable
**Effect**: 1.5-2x speedup
**Trade-off**: Slight quality degradation

```python
pipe_config.dit_config.feature_cache_config = FeatureCacheConfig(
    enabled=True,
    model_type="wan21_14b",  # Model-specific cache parameters
    n_derivatives=1,
    taylor_threshold=2,
)
```

#### 2.6 CPU Offload

**When**: GPU memory insufficient
**Effect**: Reduces VRAM by 50-70%
**Trade-off**: 10-30% slower

```python
# Strategy selection based on memory pressure
from telefuser.core.config import OffloadConfig, WeightOffloadType

# Recommended: Async offload
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=1,
)

# For severe memory constraints
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD,
    pin_cpu_memory=True,
)
```

### Configuration Generator

```python
def generate_phase2_config(
    model_size_gb: float,
    gpu_count: int,
    gpu_memory_gb: float,
    has_cfg: bool,
    sequence_length: int,
    priority: str = "speed",  # "speed", "memory", "balanced"
) -> dict:
    """Generate Phase 2 optimization config based on constraints."""

    config = {
        "parallel": None,
        "offload": None,
        "feature_cache": None,
    }

    # Memory pressure check
    memory_pressure = model_size_gb / (gpu_memory_gb * gpu_count)

    # 1. Parallel strategy
    if gpu_count > 1:
        if memory_pressure < 0.6:
            # Plenty of memory - optimize for speed
            if has_cfg and gpu_count >= 2:
                if gpu_count == 2:
                    config["parallel"] = {"cfg_degree": 2}
                elif gpu_count == 4:
                    config["parallel"] = {"cfg_degree": 2, "sp_ulysses_degree": 2}
                else:  # 8+
                    config["parallel"] = {"cfg_degree": 2, "sp_ulysses_degree": 4}

            if sequence_length > 10000:
                # Long sequence - add Ring
                config["parallel"]["sp_ring_degree"] = gpu_count // 4

    # 2. Offload strategy
    if memory_pressure > 0.8:
        config["offload"] = {
            "type": "ASYNC_CPU_OFFLOAD",
            "prefetch_size": 2,
        }
    elif memory_pressure > 0.6:
        config["offload"] = {
            "type": "MODEL_CPU_OFFLOAD",
        }

    # 3. Feature cache
    if priority == "speed" and memory_pressure < 0.7:
        config["feature_cache"] = {
            "enabled": True,
            "model_type": "auto",  # Will be determined by model
        }

    return config
```

### 🛑 PHASE 2 CHECKPOINT

After analysis:

```
## Phase 2 Analysis

**Hardware**:
- GPUs: 2x H100 (80GB each)
- Total Memory: 160GB

**Model**:
- Size: 14B parameters (~28GB FP16)
- CFG: Yes (cfg_scale=4.0)
- Sequence: 4096 tokens

**Memory Pressure**: 28/160 = 17.5% (Low)

**Recommended Optimizations**:
| Option | Config | Expected Gain | Trade-off |
|--------|--------|---------------|-----------|
| CFG Parallel | cfg_degree=2 | 2x speedup | Uses 2 GPUs |
| Feature Cache | enabled=True | 1.5x speedup | Minor quality impact |

**Not Recommended**:
- Offload: Memory pressure is low, no need
- Ring Attention: Sequence length is moderate
- FSDP: Not needed with abundant memory
```

Call AskUserQuestion:
```python
AskUserQuestion(questions=[
    {
        "question": "Phase 2 analysis complete. Which optimizations do you want to apply?",
        "header": "Optimizations",
        "options": [
            {"label": "CFG Parallel (Recommended)", "description": "2x speedup with 2 GPUs, no trade-off"},
            {"label": "CFG Parallel + Feature Cache", "description": "3x speedup, minor quality impact"},
            {"label": "Custom selection", "description": "I'll specify which optimizations to apply"},
        ],
        "multiSelect": False,
    }
])
```

### Apply Selected Optimizations

Based on user selection, generate configuration:

```python
# Generate example file with selected optimizations
# examples/<model>/<model>_optimized.py

PPL_CONFIG = dict(
    # ... base config ...

    # Phase 2 optimizations
    parallel_config=ParallelConfig(
        device_ids=[0, 1],
        cfg_degree=2,  # User selected
    ),

    # Feature cache (if selected)
    feature_cache_config=FeatureCacheConfig(
        enabled=True,
        model_type="wan21_14b",
    ),
)
```

---

## Phase 3: Custom Optimization

### Goal
Address user-specific performance requirements through:
- Bottleneck analysis
- Custom solution exploration
- Experimental optimization techniques
- New technique integration

### Triggered By
- User reports specific performance issues
- User wants to push beyond Phase 1/2 optimizations
- User has unusual requirements (extreme memory constraints, specific latency targets)

### Process

#### 3.1 Profiling

```python
# Enable detailed profiling
pipeline.enable_metrics()

# Run inference
result = pipeline(prompt, ...)

# Get detailed metrics
print(pipeline.get_prometheus_metrics())

# Output example:
# stage_denoising_duration_seconds 45.2
# stage_vae_duration_seconds 3.1
# stage_text_encoding_duration_seconds 0.8
# attention_time_seconds 28.5
# ffn_time_seconds 12.3
```

#### 3.2 Bottleneck Identification

```
Time Distribution:
├── Denoising: 45.2s (91%)
│   ├── Attention: 28.5s (63%)
│   ├── FFN: 12.3s (27%)
│   └── Other: 4.4s (10%)
├── VAE Decode: 3.1s (6%)
└── Text Encoding: 0.8s (2%)

Primary Bottleneck: Attention (63% of time)
```

#### 3.3 Solution Exploration

| Bottleneck | Possible Solutions | Feasibility |
|------------|-------------------|-------------|
| Attention dominant | Sparse attention (Radial, Local) | Requires model support |
| | Attention quantization | Experimental |
| | KV cache | Not applicable for diffusion |
| FFN dominant | FFN quantization | Experimental |
| | MoE-style pruning | Model specific |
| Memory | Model pruning | Requires retraining |
| | Distillation | Requires retraining |

#### 3.4 Experimental Features

**Sparse Attention (for video):**

```python
# Radial attention - only attends to nearby tokens
from telefuser.core.config import SparseAttentionConfig

pipe_config.dit_config.sparse_attention_config = SparseAttentionConfig(
    sparse_impl="radial",
    dense_timesteps=40,  # Use dense for first 40 steps
    decay_factor=1.0,
)
```

**Attention Quantization:**

```python
# Experimental: QK quantization
from telefuser.ops.attention import attention

# Use lower precision for QK
output = attention(
    q=q.to(torch.float8_e4m3fn),
    k=k.to(torch.float8_e4m3fn),
    v=v,
    ...
)
```

### 🛑 PHASE 3 CHECKPOINT

After analysis:

```
## Phase 3 Analysis

**Bottleneck**: Attention (63% of denoising time)

**Potential Solutions**:
1. Radial Sparse Attention
   - Speedup: 2-3x
   - Trade-off: Quality degradation for long videos
   - Status: ✅ Available in TeleFuser

2. Attention Quantization (FP8)
   - Speedup: 1.3x
   - Trade-off: Minor quality impact
   - Status: ⚠️ Experimental

3. Model Distillation
   - Speedup: 2-5x
   - Trade-off: Requires retrained model
   - Status: ❌ Requires external work
```

Call AskUserQuestion:
```python
AskUserQuestion(questions=[
    {
        "question": "Phase 3 custom optimization analysis. Which approach do you want to explore?",
        "header": "Approach",
        "options": [
            {"label": "Radial Sparse Attention", "description": "2-3x speedup for videos, available now"},
            {"label": "Attention Quantization", "description": "1.3x speedup, experimental"},
            {"label": "No custom optimization needed", "description": "Phase 1+2 optimizations are sufficient"},
        ],
        "multiSelect": False,
    }
])
```

---

## Summary: Three-Phase Workflow

| Phase | Goal | Verification | User Decision |
|-------|------|--------------|---------------|
| 1. Reuse Ops | Replace with TeleFuser ops | Output must match exactly | Proceed after verification |
| 2. Multi-GPU | Apply standard optimizations | Performance metrics | User selects optimizations |
| 3. Custom | Address specific needs | Experimental | User chooses approach |

### Expected Outcomes

| Phase | Speedup | Memory | Risk |
|-------|---------|--------|------|
| 1 | 1.5-2x | Same | None (verified) |
| 2 | 2-4x | -50% to same | Low (standard configs) |
| 3 | 2-5x | Varies | Medium (experimental) |

---

## Related Documentation

- [Adding New Model](../../docs/en/adding_new_model.md)
- [Attention Configuration](../../docs/en/attention.md)
- [Parallel Inference](../../docs/en/parallel.md)
- [CPU Offloading](../../docs/en/offload.md)
- [Feature Cache](../../docs/en/feature_cache.md)