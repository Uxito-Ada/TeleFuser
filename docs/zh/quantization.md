# 量化：原理、实现与实践

量化用较低精度表示权重、激活或缓存，以减少显存占用和内存带宽，并在硬件具备相应低精度算力时加速矩阵乘。它不是简单的 dtype 转换：可用性和效果同时取决于量化格式、缩放粒度、执行内核、模型敏感度和 GPU 架构。

本章只把代码中存在实际入口的能力列为支持项，并区分在线量化、预量化 checkpoint、离线转换和量化算子。`QuantType` 中存在枚举不代表通用模型加载器已经实现了对应的在线路径。

## 支持矩阵

| 层级 | 算法或格式 | 量化对象 | 主要入口 | 当前边界 |
| --- | --- | --- | --- | --- |
| 在线模型量化 | TorchAO FP8 | Linear 权重和动态激活，W8A8 | `QuantType.TORCHAO_FP8` | Wan、Qwen-Image、LTX transformer blocks |
| 在线模型量化 | bitsandbytes NF4 | Linear 仅权重，W4A16 | `QuantType.BNB_NF4` | Wan、Qwen-Image、LTX transformer blocks |
| 预量化模型 | scaled FP8 E4M3 | Linear 权重 + 动态 FP8 激活 | `torch_dtype=torch.float8_e4m3fn` | 使用项目提供的 Qwen-Image/Wan FP8 checkpoint 示例 |
| Pipeline 专用 | vLLM FP8 GEMM | 动态激活 + 延迟量化权重，W8A8 | LiveAct 的 `QuantConfig(enabled=True)` | CUDA；需要 vLLM FP8 算子 |
| MoE 专用 | FP8 expert GEMM | expert 权重按输出通道、激活按 token | `quantize_fp8_()` + backend `fp8` | LingBot-Video MoE；需要 `torch._scaled_mm` |
| 离线转换 | INT8 | 2D 权重，默认按输出行 | `tools/convert/converter.py --linear_type int8` | 生成量化 checkpoint，不是通用在线 `QuantType.INT8` |
| 离线转换 | FP8 E4M3FN | 2D 权重，默认按输出行 | `--linear_type fp8` | 需要 `qtorch` |
| 离线转换 | MXFP4/MXFP6/MXFP8 | 2D 权重，microscaling | `--linear_type mxfp4/mxfp6/mxfp8` | 需要 `lightx2v_kernel` 的量化算子 |
| 离线转换 | NVFP4 | 2D 权重，两级缩放 FP4 | `--linear_type nvfp4` | 需要 `lightx2v_kernel`；目标是 Blackwell FP4 路径 |
| Attention | SageAttention | Q/K INT8，P/V 使用 FP16 或 FP8 路径 | `SAGE_ATTN_2_8_16`、`SAGE_ATTN_2_8_8` | 架构和 wheel 相关，详见注意力文档 |
| Cache | FP8 KV cache | 每个 token/head 的 K/V 向量 | `KVCacheConfig(fp8_kv_cache=True)` | 当前在 LiveAct pipeline 接线 |
| 底层算子 | per-token FP8 | 任意 BF16 行向量 | `per_token_quant_fp8` | 教学/算子构件，不会自动改写模型 |

!!! warning "配置边界"
    通用模型 `enable_quant` 当前真正消费的是 `BNB_NF4`、`TORCHAO_FP8`，以及部分模型的 legacy FP8 checkpoint 路径。`INT8`、`MXFP8`、`MXFP6`、`MXFP4`、`NVFP4` 虽已出现在 `QuantType`，但尚未接入通用在线加载流程；请使用离线转换器和与格式匹配的模型/内核。

    `QuantConfig.kernel_backend` 当前主要用于表达配置意图，模型分支实际按 `quant_type` 路由；`weight_block_size`、`group_size` 和 `keep_fp16_weight` 也尚未被这些通用模型分支消费。不要依赖它们改变当前执行行为。

## 先理解量化状态

### 整数量化

对称整数数量化通常写成：

\[
s = \frac{\max |x|}{q_{\max}}, \qquad
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right), q_{\min}, q_{\max}\right)
\]

反量化为：

\[
\hat{x}=s q
\]

TeleFuser 离线 INT8 转换使用零点为 0 的对称量化。普通模式对 2D 权重的每个输出行分别计算 `absmax/127`；ComfyUI 模式使用整个 tensor 的单个 scale。

### 浮点量化

FP8、FP6、FP4 仍保留符号、指数和尾数。它们比同位宽整数拥有更宽的动态范围，但尾数更短。scaled FP8 仍需要 scale：

\[
s = \frac{\max |x|}{\operatorname{max}(\mathrm{FP8})}, \qquad
q = \operatorname{cast}_{\mathrm{FP8}}\left(\operatorname{clip}(x/s)\right)
\]

E4M3 使用 4 个指数位和 3 个尾数位，适合矩阵乘中的权重和激活；累加和最终输出通常保持 BF16/FP16/FP32。

### Scale 粒度

| 粒度 | 一个 scale 覆盖的范围 | 特点 |
| --- | --- | --- |
| Per-tensor | 整个 tensor | 元数据最少，容易被离群值主导 |
| Per-channel | 一行权重或一个输出通道 | 权重精度与开销的常见平衡点 |
| Per-token | 一个 token 的 hidden vector | 动态适应激活范围，推理时需要求 absmax |
| Per-block/group | 一小组连续元素 | 更适合 4/6 bit，scale 元数据和精度之间折中 |

scale 越细，通常误差越小，但量化 reduction、scale 存储和 kernel 寻址开销越大。

### Weight-only 与 W8A8

- **Weight-only（例如 NF4）**：权重保持 4 bit 存储，计算前由内核解码；激活仍是 BF16。主要目标是省显存，是否加速取决于解码开销和矩阵形状。
- **W8A8（例如动态 FP8）**：权重和激活都进入 FP8 GEMM。权重可预量化，激活在每次 forward 动态按 token 量化。硬件支持良好时既省带宽又能提高吞吐。
- **Cache quantization**：只压缩长期保存的 K/V，不改变模型权重；使用缓存前通常反量化回计算 dtype。

## 在线 TorchAO FP8

### 原理

TeleFuser 调用 TorchAO 的 in-place `quantize_` API，优先选择动态激活 FP8 + FP8 权重配置。权重在转换后以 FP8 表示；每次 Linear forward 根据当前激活范围动态求 scale，再执行 scaled FP8 GEMM。默认只改写 transformer blocks，并跳过 `head`、`time_embedding`、`time_projection` 和 `patch_embedding` 等敏感模块。

动态激活量化可适应不同 prompt、时间步和 token 的范围，代价是每次 forward 都需要一次 absmax/scale 计算。

### 实践

项目提供了完整的 Qwen-Image 示例：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --prompt "A cat playing piano" \
  --aspect_ratio 1:1 \
  --num-inference-steps 16 \
  --seed 42 \
  --output qwen_image_fp8.png
```

最小配置方式：

```python
import torch

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager

quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.TORCHAO_FP8,
    kernel_backend=QuantKernelBackend.TORCHAO,
    # 名字过滤是 substring match，并且相对于 transformer blocks。
    quantize_modules=("attn", "mlp"),
    skip_modules=("head", "time_embedding", "time_projection", "patch_embedding"),
)

manager = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
manager.load_model(
    dit_paths,
    device="cuda",                 # 在线改写前权重必须已经真实落到目标设备
    torch_dtype=torch.bfloat16,
    quant_config=quant_config,
)
```

源码目前在 Wan、Qwen-Image 和 LTX DiT 的 `enable_quant` 中接通该配置。Hopper/H100 是此路径的主要目标；其他架构应以 TorchAO 当前版本的 kernel 支持和实测结果为准。

## 在线 bitsandbytes NF4

### 原理

NF4（NormalFloat4）不是均匀 INT4。它使用为近似正态分布权重设计的非均匀 4-bit codebook，让有限的 16 个表示值更贴近神经网络权重的高密度区间。

TeleFuser 将选中的 `nn.Linear` 替换为 `bitsandbytes.nn.Linear4bit`：

- 权重使用 `Params4bit(quant_type="nf4")`；
- 计算 dtype 默认为 BF16，因此属于 W4A16；
- `compress_statistics=True`，会进一步压缩量化统计量，也称 double quantization；
- bias 保持 BF16；
- 默认只替换 transformer blocks 中通过过滤器的 Linear。

NF4 通常比 FP8 更省权重显存，但运行速度不一定更快。小 batch 或不理想的 kernel shape 下，4-bit 解码成本可能抵消带宽收益。

### 实践

```bash
# CUDA 版本按本机 bitsandbytes 构建调整。
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

自定义加载只需更换配置：

```python
quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.BNB_NF4,
    kernel_backend=QuantKernelBackend.BITSANDBYTES,
    quantize_modules=("attn", "mlp"),
)
manager.load_model(dit_paths, device="cuda", torch_dtype=torch.bfloat16, quant_config=quant_config)
```

如果导入失败，先检查 `bitsandbytes` wheel 是否匹配 CUDA，再检查 `CUDA_HOME`、`LD_LIBRARY_PATH`；CUDA 13 wheel 还需要动态链接器能找到 `libnvJitLink.so.13`。

## Scaled FP8 checkpoint 与 TeleFuser FP8 Linear

### 原理

这条路径与 TorchAO 在线量化不同。checkpoint 已经包含 FP8 E4M3FN 权重及 scale，TeleFuser 使用 `LinearFP8` 消费它们：

- 权重默认按输出通道保存 scale，形状为 `[out_features, 1]`；
- 激活在 forward 中动态按 token 量化为 FP8；
- `tf_kernel` 可用时优先调用 `tf_per_token_quant_fp8` 和 `fp8_scaled_mm`；
- 否则尝试 vLLM 的 `scaled_fp8_quant` 与 CUTLASS scaled GEMM；
- 输出回到输入或 autocast 对应的 BF16/FP16 dtype。

这是 W8A8 scaled GEMM，不等价于把整个模型直接 cast 为 FP8。归一化、bias、softmax 等敏感算子仍应保持更高精度。

### 实践

使用与该加载协议匹配的预量化 checkpoint 和仓库示例：

```bash
python examples/qwen_image/qwen_image_t2i_lightning_fp8_h100.py \
  --prompt "A studio portrait with soft window light" \
  --output qwen_fp8.jpg

python examples/wan_video/wan22_14b_image_to_video_distill_fp8_h100.py \
  --prompt "A boat crossing a calm lake at sunrise"
```

Wan 示例将结果写入 `TELEAI_EXAMPLE_OUTPUT_DIR`（默认当前目录）。

关键加载参数是：

```python
manager.load_model(
    fp8_checkpoint,
    device="cuda",
    torch_dtype=torch.float8_e4m3fn,
)
```

不要把任意 BF16 checkpoint 仅通过上述 dtype 参数当作 scaled FP8 checkpoint。权重文件必须包含实现期望的 scale 命名和布局，实际部署应从仓库对应示例开始。

## LiveAct 动态 FP8 GEMM

LiveAct 使用另一套 vLLM 风格的包装器：`enable_fp8_gemm`。它在包装时或第一次 CUDA forward 时量化 Linear 权重并缓存 FP8 权重，激活按 token 动态量化。默认 `FP8GemmOptions` 会在 FP8 materialize 后丢弃原始高精度权重，以降低 steady-state VRAM。

```python
from telefuser.ops.fp8_gemm import FP8GemmOptions, enable_fp8_gemm

enable_fp8_gemm(
    model,
    options=FP8GemmOptions(
        fp16_weight_storage="discard",  # 也可为 keep 或 cpu_offload
        materialize_fp8_on_wrap=True,
        cast_inputs=True,
        cast_output_back=True,
    ),
)
```

完整 pipeline 使用：

```python
config.dit_config.quant_config = QuantConfig(enabled=True, quant_type=QuantType.FP8)
```

然后运行 `examples/liveact/liveact_s2v_h100.py`。在该 pipeline 中 `enabled=True` 触发的是 LiveAct 专用 FP8 GEMM 包装逻辑，不是通用 `QuantType` 路由；当前代码不会按 `quant_type` 或 `kernel_backend` 进一步分派。

## LingBot-Video MoE FP8

### 原理

MoE expert 权重形状为 `[experts, out_features, in_features]`。TeleFuser 对每个 expert 的每个输出通道计算 scale：

\[
s_{e,o}=\frac{\max_k |W_{e,o,k}|}{\operatorname{max}(\mathrm{FP8})}
\]

权重只量化一次。执行时，把路由到同一 expert 的 token 排序到一起；输入和中间激活分别按行动态 FP8 量化，再用 `torch._scaled_mm` 完成三个 expert GEMM。路由权重的最终 reduction 使用 FP32。

### 实践

```python
from telefuser.models.lingbot_video_moe import LingBotVideoGroupedExperts

for module in model.modules():
    if isinstance(module, LingBotVideoGroupedExperts):
        module.quantize_fp8_()
        module.set_execution_backend("fp8")
```

该路径要求 CUDA build 提供 `torch._scaled_mm`。在切换 backend 前必须调用 `quantize_fp8_()`，否则 forward 会明确报错。

## 离线 checkpoint 转换

离线转换在部署前生成权重和 scale，避免每次启动重复量化，也便于分发固定 artifact。

### 安装与通用命令

`tools/convert/quant.py` 顶层依赖 `qtorch`；MX 和 NVFP4 还需要能够导入 `lightx2v_kernel.gemm` 中对应的 `scaled_*_quant`：

```bash
pip install qtorch
```

统一命令模板：

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

!!! note "参数名"
    当前 CLI 参数是 `--linear_type`，可选 `int8`、`fp8`、`nvfp4`、`mxfp4`、`mxfp6`、`mxfp8`。`--bits` 目前只接受 8，并不用于选择 4/6 bit 格式；低位格式同样通过 `--linear_type` 选择。

支持的 `--model_type` 包括 `wan_dit`、`wan_animate_dit`、`qwen_image_dit`、`hunyuan_dit`、`wan_t5`、`wan_clip` 和 `qwen25vl_llm`。转换器只量化目标模块中的 2D tensor；其他 tensor 转为 `--non_linear_dtype`。

输出 metadata 默认采用：

```text
<original_weight_key>        quantized weight
<original_weight_key>_scale  scale
<original_weight_key>_global_scale  NVFP4 only
```

ComfyUI 模式改用 `.scale_weight` 命名，并对 INT8/FP8 使用 per-tensor scale。

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

实现是 per-output-row symmetric INT8，零点为 0。它比 FP8 有更均匀的量化间隔，但动态范围较窄，对行内离群值敏感。该命令生成 checkpoint；要获得推理加速，还必须由理解相同 scale/layout 的 INT8 Linear kernel 消费它。

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

实现按输出行取 absmax，将值缩放到 `torch.float8_e4m3fn` 的有限范围，使用 nearest rounding 后保存 FP8 权重和 scale。

### MXFP4、MXFP6 与 MXFP8

MX（microscaling）格式把连续小 block 中的元素低精度值与一个共享 scale 组合起来。相比整个通道共用一个 scale，局部共享 scale 能更好地隔离离群值；代价是更多 scale 元数据和特定 packed layout。

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

具体 block 大小、元素编码和 packed tensor layout 由安装的 `lightx2v_kernel` 实现决定。不要只根据输出 tensor 的 PyTorch dtype 推断真实位宽；部署内核必须和转换器版本匹配。

### NVFP4

NVFP4 使用 FP4 元素值、局部 block scale 和 model/tensor 级 global scale。TeleFuser 转换器先计算：

\[
g=\frac{2688}{\max |W|}
\]

再把 `W` 和 global scale 交给 `scaled_nvfp4_quant` 生成 packed 权重与局部 scale，并额外保存 `<weight>_global_scale`。

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

该路径面向 Blackwell FP4 kernel。转换成功不代表当前 GPU 能执行 NVFP4 GEMM；应在部署机上同时验证计算能力、扩展构建目标和 checkpoint layout。

## 量化 Attention：SageAttention

SageAttention 不量化持久化模型权重，而是在 attention kernel 中动态量化 Q/K，并选择 FP16 或 FP8 的 P/V 计算路径。Q/K 的 INT8 scale 按 block 求取，从而降低 `QK^T` 的带宽并使用整数矩阵乘；softmax 的稳定统计和输出累加仍使用更高精度策略控制误差。

```python
from telefuser.core.config import AttentionConfig, AttnImplType

# Q/K INT8，P/V FP16 路径
config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_16)

# Q/K INT8，P/V FP8 路径
config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8)

# Hopper 专用路由
config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90)
```

不同 GPU 架构对应的 kernel、安装方式、FP4 Blackwell build 和已知 SM90 wheel 限制见[注意力机制](./attention.md#sageattention)与 [TF-Kernel](./tf_kernel.md)。量化权重和量化 attention 是正交能力，可以同时使用，但必须分别验证误差和性能。

## FP8 KV Cache

### 原理

KV cache 会跨生成步骤长期驻留。TeleFuser 沿最后一个 head dimension 为每个 `[batch, sequence, head]` 向量计算 scale：

\[
s=\max(|x|)/\operatorname{max}(\mathrm{FP8}), \qquad q=\operatorname{cast}_{\mathrm{FP8}}(x/s)
\]

K/V 以 FP8 保存，scale 以 FP32 保存。`load()` 时先转到目标设备，再反量化到请求的 BF16/FP16 dtype。因此它主要节省缓存存储和 offload 带宽，当前实现并不是直接让 attention kernel 消费 FP8 cache。

### 实践

LiveAct pipeline 已接线：

```python
from telefuser.pipelines.liveact import LiveActPipelineConfig

config = LiveActPipelineConfig()
config.fp8_kv_cache = True
config.offload_cache = False  # 可与 CPU offload 独立组合
```

也可以单独验证 round trip：

```python
import torch
from telefuser.cache import KVCache

k = torch.randn(1, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cache = KVCache(fp8_kv_cache=True)
cache.store(k, v)
k_hat, v_hat = cache.load("cuda", torch.bfloat16)

print((k.float() - k_hat.float()).abs().max())
print(cache.k.element_size(), k.element_size())  # FP8 payload 1 byte，BF16 2 bytes
```

显存估算必须包含 FP32 scale；实际收益小于单纯按 1 byte/2 bytes 得到的 50%。

## Triton per-token FP8 教学算子

`telefuser/kernel/triton/quant.py` 提供独立的 per-token FP8 round trip。每行求一次 absmax，并在输出最后两个 FP8 字节的位置编码 BF16 scale，因此量化 tensor 的最后一维是 `H + 2`。

```python
import torch
from telefuser.kernel.triton.quant import per_token_dequant_fp8, per_token_quant_fp8

x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
x_q = per_token_quant_fp8(x)
x_hat = per_token_dequant_fp8(x_q)

assert x_q.shape == (32, 4098)
print((x.float() - x_hat.float()).abs().mean())
```

这是底层构件，不会自动替换 Linear，也不能与要求独立 scale tensor 的 checkpoint/kernel 格式混用。

## 如何选择

| 目标 | 建议起点 | 原因 |
| --- | --- | --- |
| H100 上兼顾吞吐和显存 | TorchAO FP8 | 在线 W8A8、配置清晰、已有 Qwen 示例 |
| 显存最紧张 | BNB NF4 | 4-bit 权重存储，通常比 FP8 更省 |
| 需要可分发的固定 artifact | 离线 FP8/INT8 | 启动不重复量化，checkpoint 可审计 |
| Blackwell 上探索最低精度 | NVFP4/MXFP4 | 低位权重与硬件 FP4 路径，但依赖专用扩展 |
| Attention 占主要耗时 | SageAttention | 动态量化 Q/K，不改变模型 checkpoint |
| 长序列缓存占主要显存 | FP8 KV cache | 只压缩长期驻留 K/V |

不要仅比较 checkpoint 文件大小。最终选择至少需要同时测量峰值显存、warmup 后延迟、吞吐和生成质量。

## 正确性与性能验证

### 1. 确认实际发生了量化

在线转换函数会记录替换数量。还可以检查模块类型和 dtype：

```python
from collections import Counter

print(Counter(type(module).__name__ for module in model.modules()))
print(Counter(str(param.dtype) for param in model.parameters()))
```

替换数量为 0 时，优先检查目标模型是否实现了对应 `enable_quant`、`quantize_modules` 是否匹配，以及量化配置是否传给了 DiT 的 `load_model`。

### 2. 与高精度基线对比

固定 prompt、输入、seed、scheduler 和推理步数。对于单算子比较：

\[
\text{max error}=\max |y_q-y_{ref}|,
\qquad
\text{relative L2}=\frac{\|y_q-y_{ref}\|_2}{\|y_{ref}\|_2}
\]

图像/视频还应保存相同 seed 的 BF16 与量化输出，做视觉检查并使用项目适合的质量指标。只检查程序没有异常不足以证明量化质量可接受。

### 3. 测量 warmup 后性能

- 先运行至少一次，让权重量化、Triton/CUDA 编译和 autotune 完成；
- 使用 CUDA event 或 profiler 测量，不用未同步的 wall clock；
- 分别记录 checkpoint 大小、加载峰值、steady-state VRAM 和端到端延迟；
- weight-only 量化省显存但可能不加速，W8A8 也只在 shape 和硬件适合时更快。

相关单元测试：

```bash
pytest -q tests/unit/quantize/test_quantized_linear.py
pytest -q tests/unit/models/test_lingbot_video_moe.py
```

## 常见问题

### 配置了 `QuantType.INT8`，为什么 Linear 没有变化？

`INT8` 当前是枚举和离线转换格式，还没有接入 Wan/Qwen/LTX 的通用在线 `enable_quant`。使用 `tools/convert/converter.py --linear_type int8`，并确保部署端存在匹配该 checkpoint layout 的执行内核。

### 为什么量化后显存没有按位宽同比下降？

scale、bias、未量化模块、临时激活、CUDA workspace 和可能保留的高精度权重都会占显存。TorchAO、bitsandbytes 和 vLLM 包装器的保存策略也不同。

### 为什么量化后反而更慢？

常见原因是矩阵太小、动态 scale reduction 成本较高、4-bit 解码开销、发生 fallback、首次运行包含编译，或 GPU 没有目标低精度吞吐优势。先用 profiler 确认实际 kernel，再判断算法。

### LoRA 和量化的顺序是什么？

离线转换器先完成格式转换和 LoRA merge，再执行量化。在线路径也应先把 LoRA 合并到高精度权重，再量化；量化后直接修改低精度权重会破坏原有 scale，除非 backend 明确支持量化 LoRA。
