# Quantization: Principles, Implementation, and Practice

Quantization represents weights, activations, or caches at lower precision. It can reduce VRAM use and memory traffic and, when the GPU has matching low-precision hardware, accelerate matrix multiplication. It is not merely a dtype cast: format, scale granularity, execution kernel, model sensitivity, and GPU architecture all determine the result.

This guide lists only capabilities with concrete code entry points. It separates online quantization, pre-quantized checkpoints, offline conversion, and quantized operators. The presence of a value in `QuantType` does not by itself mean that generic online model loading implements it.

## Support matrix

| Layer | Algorithm or format | Quantized data | Main entry point | Current boundary |
| --- | --- | --- | --- | --- |
| Online model | TorchAO FP8 | Linear weights and dynamic activations, W8A8 | `QuantType.TORCHAO_FP8` | Wan, Qwen-Image, and LTX transformer blocks |
| Online model | bitsandbytes NF4 | Linear weights only, W4A16 | `QuantType.BNB_NF4` | Wan, Qwen-Image, and LTX transformer blocks |
| Pre-quantized model | scaled FP8 E4M3 | Linear weights plus dynamic FP8 activations | `torch_dtype=torch.float8_e4m3fn` | Use the supplied Qwen-Image/Wan FP8 checkpoint examples |
| Pipeline-specific | vLLM FP8 GEMM | Dynamic activations and lazily quantized weights, W8A8 | LiveAct `QuantConfig(enabled=True)` | CUDA; requires vLLM FP8 operators |
| MoE-specific | FP8 expert GEMM | Per-output-channel expert weights and per-token activations | `quantize_fp8_()` plus backend `fp8` | LingBot-Video MoE; requires `torch._scaled_mm` |
| Offline conversion | INT8 | 2D weights, per output row by default | `converter.py --linear_type int8` | Produces a checkpoint; not generic online `QuantType.INT8` |
| Offline conversion | FP8 E4M3FN | 2D weights, per output row by default | `--linear_type fp8` | Requires `qtorch` |
| Offline conversion | MXFP4/MXFP6/MXFP8 | 2D microscaled weights | `--linear_type mxfp4/mxfp6/mxfp8` | Requires quantizers from `lightx2v_kernel` |
| Offline conversion | NVFP4 | 2D weights with two-level scaling | `--linear_type nvfp4` | Requires `lightx2v_kernel`; targets the Blackwell FP4 path |
| Attention | SageAttention | INT8 Q/K with FP16 or FP8 P/V path | `SAGE_ATTN_2_8_16`, `SAGE_ATTN_2_8_8` | Architecture and wheel dependent |
| Cache | FP8 KV cache | Each token/head K/V vector | `KVCacheConfig(fp8_kv_cache=True)` | Currently wired into LiveAct |
| Primitive | Per-token FP8 | Any BF16 row vector | `per_token_quant_fp8` | Teaching/operator primitive; does not rewrite a model |

!!! warning "Configuration boundary"
    Generic model `enable_quant` implementations currently consume `BNB_NF4`, `TORCHAO_FP8`, and the legacy FP8-checkpoint path on selected models. `INT8`, `MXFP8`, `MXFP6`, `MXFP4`, and `NVFP4` exist in `QuantType` but are not wired into generic online loading. Use the offline converter and a matching model/kernel contract.

    `QuantConfig.kernel_backend` currently expresses intent; model code dispatches on `quant_type`. The generic branches also do not consume `weight_block_size`, `group_size`, or `keep_fp16_weight` yet.

## Quantization state

### Integer quantization

Symmetric integer quantization is commonly defined as:

\[
s = \frac{\max |x|}{q_{\max}}, \qquad
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right), q_{\min}, q_{\max}\right)
\]

and dequantization as:

\[
\hat{x}=s q
\]

TeleFuser's offline INT8 converter uses symmetric zero-point-0 quantization. In normal mode, each output row of a 2D weight receives an `absmax/127` scale. ComfyUI mode uses one scale for the entire tensor.

### Floating-point quantization

FP8, FP6, and FP4 retain sign, exponent, and mantissa fields. They offer more dynamic range than an integer at the same width but less mantissa precision. Scaled FP8 still uses a scale:

\[
s = \frac{\max |x|}{\operatorname{max}(\mathrm{FP8})}, \qquad
q = \operatorname{cast}_{\mathrm{FP8}}\left(\operatorname{clip}(x/s)\right)
\]

E4M3 has four exponent bits and three mantissa bits. GEMM inputs may use E4M3 while accumulators and outputs stay in BF16, FP16, or FP32.

### Scale granularity

| Granularity | One scale covers | Trade-off |
| --- | --- | --- |
| Per-tensor | A complete tensor | Least metadata; most exposed to outliers |
| Per-channel | One weight row/output channel | Common weight accuracy/overhead balance |
| Per-token | One token hidden vector | Adapts to activations; requires runtime absmax reduction |
| Per-block/group | A small contiguous group | Better for 4/6 bit; more scale metadata and addressing |

Finer scales usually reduce error but increase reduction work, metadata, and kernel complexity.

### Weight-only versus W8A8

- **Weight-only (NF4)** keeps weights in 4-bit storage while activations remain BF16. Its primary benefit is capacity; decoding overhead and GEMM shape determine speed.
- **W8A8 (dynamic FP8)** sends both weights and activations through scaled FP8 GEMM. Weights can be cached, while activations are quantized per forward call.
- **Cache quantization** compresses persistent K/V without changing model weights. The current cache path dequantizes before attention computation.

## Online TorchAO FP8

### Principle

TeleFuser calls TorchAO's in-place `quantize_` API and prefers dynamic-activation FP8 plus FP8 weight configuration. The Linear weight is converted once; each forward computes activation scales and runs scaled FP8 GEMM. By default, only transformer blocks are targeted, and sensitive names such as `head`, `time_embedding`, `time_projection`, and `patch_embedding` are excluded.

Dynamic scales adapt to prompt, timestep, and token ranges at the cost of an absmax/scale operation on every forward.

### Practice

Run the complete Qwen-Image example:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --prompt "A cat playing piano" \
  --aspect_ratio 1:1 \
  --num-inference-steps 16 \
  --seed 42 \
  --output qwen_image_fp8.png
```

Minimal loading configuration:

```python
import torch

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager

quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.TORCHAO_FP8,
    kernel_backend=QuantKernelBackend.TORCHAO,
    # Filters are substring matches relative to the transformer blocks.
    quantize_modules=("attn", "mlp"),
    skip_modules=("head", "time_embedding", "time_projection", "patch_embedding"),
)

manager = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
manager.load_model(
    dit_paths,
    device="cuda",                 # Real weights must exist on the target device first.
    torch_dtype=torch.bfloat16,
    quant_config=quant_config,
)
```

Wan, Qwen-Image, and LTX DiTs currently implement this branch. Hopper/H100 is the primary target; verify the installed TorchAO version and kernels on other architectures.

## Online bitsandbytes NF4

### Principle

NF4 (NormalFloat4) is not uniform INT4. Its non-uniform 16-value codebook is designed for approximately normally distributed neural-network weights.

TeleFuser replaces selected `nn.Linear` modules with `bitsandbytes.nn.Linear4bit`:

- `Params4bit(quant_type="nf4")` stores the weight;
- BF16 is the default compute dtype, making this W4A16;
- `compress_statistics=True` applies double quantization to quantization statistics;
- bias remains BF16;
- only matching Linear modules under transformer blocks are replaced.

NF4 normally saves more weight memory than FP8, but is not guaranteed to be faster because 4-bit decoding may offset bandwidth savings.

### Practice

```bash
export BNB_CUDA_VERSION=128
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/qwen_image/qwen_image_t2i_telefuser_nf4_h100.py \
  --prompt "A cat playing piano" \
  --aspect_ratio 1:1 \
  --num-inference-steps 16 \
  --seed 42 \
  --output qwen_image_nf4.png
```

The corresponding custom configuration is:

```python
quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.BNB_NF4,
    kernel_backend=QuantKernelBackend.BITSANDBYTES,
    quantize_modules=("attn", "mlp"),
)
manager.load_model(dit_paths, device="cuda", torch_dtype=torch.bfloat16, quant_config=quant_config)
```

If import fails, verify the bitsandbytes wheel against CUDA and inspect `CUDA_HOME` and `LD_LIBRARY_PATH`. CUDA 13 wheels also require `libnvJitLink.so.13` to be discoverable.

## Scaled FP8 checkpoints and TeleFuser FP8 Linear

### Principle

This differs from TorchAO online quantization. A compatible checkpoint already contains FP8 E4M3FN weights and scales, and TeleFuser's `LinearFP8` consumes them:

- weight scales are normally per output channel with shape `[out_features, 1]`;
- activations are dynamically quantized per token;
- if available, `tf_kernel` provides `tf_per_token_quant_fp8` and `fp8_scaled_mm`;
- otherwise the implementation attempts vLLM quantization and CUTLASS scaled GEMM;
- output returns to the input/autocast BF16 or FP16 dtype.

This is scaled W8A8 GEMM, not a whole-model FP8 cast. Normalization, bias, softmax, and other sensitive operations remain at higher precision.

### Practice

Start with checkpoints and examples that implement the same scale/layout contract:

```bash
python examples/qwen_image/qwen_image_t2i_lightning_fp8_h100.py \
  --prompt "A studio portrait with soft window light" \
  --output qwen_fp8.jpg

python examples/wan_video/wan22_14b_image_to_video_distill_fp8_h100.py \
  --prompt "A boat crossing a calm lake at sunrise"
```

The Wan example writes its result under `TELEAI_EXAMPLE_OUTPUT_DIR` (the current directory by default).

The key load argument is:

```python
manager.load_model(fp8_checkpoint, device="cuda", torch_dtype=torch.float8_e4m3fn)
```

Do not treat an arbitrary BF16 checkpoint as a scaled-FP8 checkpoint by changing this dtype alone. The artifact must carry the expected scale names and layouts.

## LiveAct dynamic FP8 GEMM

LiveAct uses the separate vLLM-style `enable_fp8_gemm` wrapper. It quantizes and caches Linear weights during wrapping or the first CUDA forward and quantizes activations per token. The default options discard the original high-precision weight after FP8 materialization to minimize steady-state VRAM.

```python
from telefuser.ops.fp8_gemm import FP8GemmOptions, enable_fp8_gemm

enable_fp8_gemm(
    model,
    options=FP8GemmOptions(
        fp16_weight_storage="discard",  # keep and cpu_offload are alternatives
        materialize_fp8_on_wrap=True,
        cast_inputs=True,
        cast_output_back=True,
    ),
)
```

In the full pipeline:

```python
config.dit_config.quant_config = QuantConfig(enabled=True, quant_type=QuantType.FP8)
```

Then run `examples/liveact/liveact_s2v_h100.py`. In this pipeline, `enabled=True` activates LiveAct-specific wrapping; the current code does not dispatch further on `quant_type` or `kernel_backend`.

## LingBot-Video MoE FP8

### Principle

Expert weights have shape `[experts, out_features, in_features]`. TeleFuser computes one scale per expert output channel:

\[
s_{e,o}=\frac{\max_k |W_{e,o,k}|}{\operatorname{max}(\mathrm{FP8})}
\]

Weights are quantized once. At execution time, tokens are sorted by expert; inputs and intermediate activations are dynamically quantized per row, and three expert GEMMs run through `torch._scaled_mm`. The final route-weight reduction uses FP32.

### Practice

```python
from telefuser.models.lingbot_video_moe import LingBotVideoGroupedExperts

for module in model.modules():
    if isinstance(module, LingBotVideoGroupedExperts):
        module.quantize_fp8_()
        module.set_execution_backend("fp8")
```

The CUDA build must provide `torch._scaled_mm`. Calling the FP8 backend before `quantize_fp8_()` produces an explicit error.

## Offline checkpoint conversion

Offline conversion produces weights and scales before deployment, avoiding repeated startup quantization and making the artifact auditable.

### Installation and common command

`tools/convert/quant.py` imports `qtorch` at module load. MX and NVFP4 additionally need the corresponding `scaled_*_quant` functions from `lightx2v_kernel.gemm`.

```bash
pip install qtorch
```

Common command template:

```bash
python tools/convert/converter.py \
  --source /path/to/source \
  --output /path/to/output \
  --model_type wan_dit \
  --quantized \
  --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 \
  --single_file
```

!!! note "CLI names"
    The current option is `--linear_type`, with choices `int8`, `fp8`, `nvfp4`, `mxfp4`, `mxfp6`, and `mxfp8`. `--bits` currently accepts only 8 and does not select 4/6-bit formats; select those formats with `--linear_type`.

Supported model types are `wan_dit`, `wan_animate_dit`, `qwen_image_dit`, `hunyuan_dit`, `wan_t5`, `wan_clip`, and `qwen25vl_llm`. Only targeted 2D tensors are quantized; other tensors are converted to `--non_linear_dtype`.

Default metadata naming is:

```text
<original_weight_key>               quantized weight
<original_weight_key>_scale         scale
<original_weight_key>_global_scale  NVFP4 only
```

ComfyUI mode uses `.scale_weight` and per-tensor scaling for INT8/FP8.

### INT8

```bash
python tools/convert/converter.py \
  -s /path/to/Wan2.1-I2V-14B-480P \
  -o /path/to/wan-int8 \
  -t wan_dit \
  --quantized \
  --linear_type int8 \
  --non_linear_dtype torch.bfloat16 \
  --save_by_block
```

This is symmetric per-output-row INT8 with zero point 0. It has uniform intervals and less dynamic range than FP8. The command creates a checkpoint; inference acceleration still requires an INT8 Linear kernel that understands the same scale/layout contract.

### FP8 E4M3FN

```bash
python tools/convert/converter.py \
  -s /path/to/Qwen-Image-2512/transformer \
  -o /path/to/qwen-fp8 \
  -t qwen_image_dit \
  --quantized \
  --linear_type fp8 \
  --non_linear_dtype torch.bfloat16 \
  --single_file
```

The converter takes an absmax per output row, scales into the finite `torch.float8_e4m3fn` range, applies nearest rounding, and saves the FP8 weight and scale.

### MXFP4, MXFP6, and MXFP8

MX (microscaling) formats combine low-precision values in a small block with a shared scale. Local scales isolate outliers better than one channel-wide scale, at the cost of scale metadata and a backend-specific packed layout.

```bash
for quant_type in mxfp4 mxfp6 mxfp8; do
  python tools/convert/converter.py \
    -s /path/to/source \
    -o "/path/to/output-${quant_type}" \
    -t wan_dit \
    --quantized \
    --linear_type "${quant_type}" \
    --non_linear_dtype torch.bfloat16 \
    --single_file
done
```

The installed `lightx2v_kernel` defines block size, element encoding, and packed tensor layout. The deployment kernel and converter versions must agree; a PyTorch tensor dtype alone does not reveal the effective packed width.

### NVFP4

NVFP4 combines FP4 element values, local block scales, and a model/tensor-level global scale. TeleFuser first computes:

\[
g=\frac{2688}{\max |W|}
\]

It then calls `scaled_nvfp4_quant` and stores the packed weight, local scales, and `<weight>_global_scale`.

```bash
python tools/convert/converter.py \
  -s /path/to/source \
  -o /path/to/output-nvfp4 \
  -t wan_dit \
  --quantized \
  --linear_type nvfp4 \
  --non_linear_dtype torch.bfloat16 \
  --single_file
```

This path targets Blackwell FP4 kernels. Successful conversion does not imply that the current GPU can execute NVFP4 GEMM; validate compute capability, extension build target, and checkpoint layout on the deployment host.

## Quantized attention: SageAttention

SageAttention quantizes runtime Q/K rather than persistent model weights, then selects an FP16 or FP8 P/V path. Per-block INT8 scales reduce Q/K traffic and enable integer matrix multiplication while higher-precision softmax statistics and accumulation control error.

```python
from telefuser.core.config import AttentionConfig, AttnImplType

config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_16)      # Q/K INT8, P/V FP16
config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8)       # Q/K INT8, P/V FP8
config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90)  # Hopper route
```

See [Attention](./attention.md#sageattention) and [TF-Kernel](./tf_kernel.md) for architecture-specific kernels, installation, the Blackwell FP4 build, and known SM90 wheel constraints. Weight quantization and quantized attention are orthogonal and can be combined, but accuracy and performance must be validated separately.

## FP8 KV cache

TeleFuser computes one scale along the final head dimension for each `[batch, sequence, head]` K/V vector:

\[
s=\max(|x|)/\operatorname{max}(\mathrm{FP8}), \qquad q=\operatorname{cast}_{\mathrm{FP8}}(x/s)
\]

K/V payloads use FP8 and scales use FP32. `load()` moves data to the requested device and dequantizes to BF16/FP16. This saves persistent storage and offload traffic; the current attention kernel does not consume the FP8 cache directly.

LiveAct wiring:

```python
from telefuser.pipelines.liveact import LiveActPipelineConfig

config = LiveActPipelineConfig()
config.fp8_kv_cache = True
config.offload_cache = False
```

Standalone round trip:

```python
import torch
from telefuser.cache import KVCache

k = torch.randn(1, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cache = KVCache(fp8_kv_cache=True)
cache.store(k, v)
k_hat, v_hat = cache.load("cuda", torch.bfloat16)

print((k.float() - k_hat.float()).abs().max())
print(cache.k.element_size(), k.element_size())
```

Include FP32 scale storage in memory estimates; the real saving is below the naive 50% implied by 1-byte FP8 versus 2-byte BF16 payloads.

## Triton per-token FP8 teaching primitive

`telefuser/kernel/triton/quant.py` implements a standalone per-token FP8 round trip. Each row gets an absmax scale. The BF16 scale is encoded in the final two FP8 byte positions, so the quantized last dimension is `H + 2`.

```python
import torch
from telefuser.kernel.triton.quant import per_token_dequant_fp8, per_token_quant_fp8

x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
x_q = per_token_quant_fp8(x)
x_hat = per_token_dequant_fp8(x_q)

assert x_q.shape == (32, 4098)
print((x.float() - x_hat.float()).abs().mean())
```

This is an operator primitive. It does not replace Linear modules and is not interchangeable with checkpoint/kernel formats that store scales in separate tensors.

## Choosing a path

| Goal | Suggested starting point | Why |
| --- | --- | --- |
| Balance throughput and memory on H100 | TorchAO FP8 | Online W8A8 with an existing Qwen example |
| Minimize weight VRAM | BNB NF4 | 4-bit weight storage |
| Produce a distributable fixed artifact | Offline FP8/INT8 | No repeated startup quantization; auditable checkpoint |
| Explore lowest precision on Blackwell | NVFP4/MXFP4 | Low-bit weights and hardware FP4 path, with extension dependency |
| Attention dominates runtime | SageAttention | Dynamically quantizes Q/K without changing the checkpoint |
| Long-lived cache dominates memory | FP8 KV cache | Compresses only persistent K/V |

Do not compare checkpoint size alone. Measure peak loading memory, steady-state VRAM, post-warmup latency, throughput, and generation quality.

## Correctness and performance validation

### Confirm that conversion happened

Online paths log replacement counts. You can also inspect module types and parameter dtypes:

```python
from collections import Counter

print(Counter(type(module).__name__ for module in model.modules()))
print(Counter(str(param.dtype) for param in model.parameters()))
```

If the replacement count is zero, check that the model implements the requested `enable_quant`, name filters match, and `quant_config` was passed to the DiT `load_model` call.

### Compare against a high-precision baseline

Hold prompt, input, seed, scheduler, and inference steps constant. For an operator, record:

\[
\text{max error}=\max |y_q-y_{ref}|,
\qquad
\text{relative L2}=\frac{\|y_q-y_{ref}\|_2}{\|y_{ref}\|_2}
\]

For images and videos, save BF16 and quantized outputs at the same seed, inspect them, and use task-appropriate quality metrics. A successful process exit is not an accuracy test.

### Measure after warmup

- Run at least one warmup so lazy weight quantization, compilation, and autotuning are complete.
- Use CUDA events or the profiler rather than unsynchronized wall-clock time.
- Record artifact size, load-time peak, steady-state VRAM, and end-to-end latency separately.
- Weight-only quantization may save memory without speeding up; W8A8 is faster only on suitable shapes and hardware.

Relevant unit tests:

```bash
pytest -q tests/unit/quantize/test_quantized_linear.py
pytest -q tests/unit/models/test_lingbot_video_moe.py
```

## Troubleshooting

### Why did `QuantType.INT8` not change any Linear modules?

INT8 currently exists as an enum and offline conversion format, not a generic online Wan/Qwen/LTX branch. Use `tools/convert/converter.py --linear_type int8` and a deployment kernel that understands the resulting layout.

### Why did VRAM not fall in proportion to bit width?

Scales, bias, unquantized modules, activations, CUDA workspaces, and retained high-precision weights all consume memory. TorchAO, bitsandbytes, and vLLM wrappers also have different storage policies.

### Why is quantized inference slower?

Typical causes are small matrices, dynamic scale reductions, 4-bit decode overhead, backend fallback, first-run compilation, or hardware without a throughput advantage for the target format. Confirm the actual kernel with a profiler first.

### In what order should LoRA and quantization run?

The offline converter performs format conversion and LoRA merging before quantization. Online flows should likewise merge LoRA into high-precision weights first. Modifying already quantized weights invalidates their scales unless the backend explicitly supports quantized LoRA.
