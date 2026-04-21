<div align="center">
  <img src="assets/telefuser_logo.png" width="80%">
</div>

<p align="center">
  <a href="README_zh.md">中文</a> | English
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-green" alt="CUDA">
</p>

TeleFuser is a high-performance framework for multimodal generation model inference, supporting image generation, video generation, and video super-resolution pipelines.

## Overview

TeleFuser is designed to provide optimized inference for large-scale multimodal generation models. The framework features a modular architecture with stage-based pipeline composition, unified API design for both local and server deployment, and comprehensive optimization techniques.

## Features

- **Stage-based Pipeline Architecture**: Modular pipeline design composed of reusable stages (text encoding, denoising, VAE, etc.) for flexible model orchestration
- **Unified API Design**: Consistent API interface for both local Python calls and server deployment
- **Asynchronous Pipeline Scheduling**: Stage-level async execution with parallel group scheduling and shared resource locking for concurrent request processing
- **Distributed Parallelism**: Multi-GPU processing with Ray framework, supporting data parallelism, tensor parallelism, and sequence parallelism (Ulysses/Ring/USP)
- **Quantization Support**: FP8 quantization for reduced memory footprint
- **LoRA Loading**: Runtime LoRA weight loading for inference with fine-tuned models
- **Custom Attention Implementations**: Optimized attention kernels including SageAttention, FlashAttention, and sparse attention variants
- **Memory Optimization**: Flexible CPU offloading strategies and intelligent weight caching for large model support
- **Feature Caching**: AdaTaylorCache for Wan2.1/2.2 models with precomputed skip steps and hybrid Taylor-residual approximation
- **REST API Server**: FastAPI server with OpenAI-compatible endpoints (`/v1/images`, `/v1/videos`) and native API (`/v1/tasks/*`)
- **HuggingFace Format Support**: Load models from HuggingFace Hub or local folders in Diffusers format

## Installation

```bash
pip install -e .
```

For development installation:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline
import torch

# Load from HuggingFace Hub
pipe = Wan21VideoPipeline.from_pretrained(
    model_id_or_path="Wan-AI/Wan2.1-T2V-1.3B",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# Generate video
video = pipe(
    prompt="A cat playing piano",
    num_frames=81,
    height=480,
    width=832,
)
```

### Examples

📁 **See the `examples/` directory for complete working examples:**

| Pipeline | Task | Examples | Features & Performance |
|----------|------|----------|------------------------|
| **WanVideo** | T2V, I2V, FL2V | `examples/wan_video/` | [README](examples/wan_video/README.md) |
| **Qwen-Image** | T2I, Edit | `examples/qwen_image/` | [README](examples/qwen_image/README.md) |
| **Z-Image** | T2I | `examples/z_image/` | [README](examples/z_image/README.md) |
| **HunyuanVideo** | T2V, I2V | `examples/hunyuan_video/` | [README](examples/hunyuan_video/README.md) |
| **FlashVSR** | VSR | `examples/flashvsr/` | [README](examples/flashvsr/README.md) |
| **LongCat-Video** | T2V, I2V | `examples/longcat_video/` | [README](examples/longcat_video/README.md) |
| **LTX Video** | I2V + Audio | `examples/ltx_video/` | [README](examples/ltx_video/README.md) |
| **Flux2 Klein** | T2I | `examples/flux2_klein/` | [README](examples/flux2_klein/README.md) |
| **LiveAct** | S2V | `examples/liveact/` | Speech-to-Video (talking head) |

> 💡 **Tip:** Click the README links above to see detailed features, model sources, and performance metrics for each pipeline.

Run any example with `--help` to see available options:

```bash
python examples/wan_video/wan21_1_3b_text_to_video_hf.py --help
```

Examples that support `telefuser serve` can also declare a pipeline contract through `PIPELINE_MANIFEST`,
`PIPELINE_CONTRACT`, or the corresponding factory functions. The server uses that contract to determine supported
tasks, required file inputs, and user-facing request parameters instead of exposing every internal pipeline knob.

📖 See [Adding New Example](docs/en/adding_new_example.md) for authoring rules and [Service Documentation](docs/en/service.md)
for how the server consumes the contract.

## Architecture Highlights

### Asynchronous Pipeline Scheduling

TeleFuser provides a stage-based asynchronous scheduling engine for concurrent request processing:

- **Parallel Stage Groups**: Independent stages (e.g., text encoding and VAE encoding) execute concurrently within parallel groups
- **Shared Resource Locking**: Stages accessing shared resources use configurable lock groups to prevent contention
- **Request Isolation**: Each request maintains independent state and queues
- **Event-Driven Execution**: Non-blocking async/await patterns throughout the pipeline

Example async usage:
```python
from telefuser.pipelines.wan_video import AsyncWan22VideoPipeline

pipe = AsyncWan22VideoPipeline(device="cuda")
pipe.init(module_manager, config)

async for event in pipe.agenerate(
    request_id="demo-1",
    prompt="A beautiful landscape",
    input_image=image,
):
    if event["type"] == "stage_end":
        print(f"Stage {event['stage']['name']} completed")
    elif event["type"] == "final":
        print(f"Video saved to {event['payload']['artifacts'][0]['uri']}")
```

## CLI Usage

TeleFuser provides a command-line interface for easy access to its functionality:

```bash
# Start API server
telefuser serve ./examples/wan_video/wan21_14b_image_to_video_h100.py --port 8000

# With multi-GPU support
telefuser serve ./examples/wan_video/wan21_14b_image_to_video_h100.py --task i2v --port 8000 --gpu-num 2

# Validate a pipeline file
telefuser validate /path/to/pipeline.py

# Scan directory for pipeline files
telefuser scan /path/to/pipelines/
```

All example scripts support command-line parameters. Use `--help` to see available options:

```bash
python examples/wan_video/wan21_1_3b_text_to_video_h100.py --help
```

📖 **For detailed CLI usage, API reference, and server configuration, see [Service Documentation](docs/en/service.md).**

## API Server

TeleFuser provides a FastAPI-based REST API server with dual API support:

- **TeleFuser Native API** (`/v1/tasks/*`) - Async task management
- **OpenAI Compatible API** (`/v1/images`, `/v1/videos`) - Industry-standard format

```bash
# Start the server
telefuser serve /path/to/pipeline.py --port 8000
```

When a pipeline file provides a manifest/contract, `telefuser serve` uses it as the source of truth for:

- supported tasks such as `t2v`, `i2v`, `fl2v`, and `vc`
- required upload inputs such as images or videos
- user-facing request defaults and required fields
- service metadata returned by `/v1/service/metadata`

Keep user-facing parameters in the contract, and keep internal tuning values in `PPL_CONFIG` or implementation code.

📖 **For detailed API documentation, client SDK, and examples, see [Service Documentation](docs/en/service.md).**

## Configuration

### Model Paths

Configure model paths in your pipeline configuration:

```python
PPL_CONFIG = dict(
    dit_path=["/path/to/dit/model-*.safetensors"],
    vae_path=["/path/to/vae/model.safetensors"],
    text_encoder_path=["/path/to/text_encoder/model-*.safetensors"],
)
```

### HuggingFace Format Loading

Load models directly from HuggingFace Hub or local directories:

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline

pipe = Wan21VideoPipeline.from_pretrained(
    model_id_or_path="Wan-AI/Wan2.1-T2V-1.3B",  # HF Model ID or local path
    device="cuda",
    torch_dtype=torch.bfloat16,
)
```

### Performance Optimization

- Use `attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90` for optimized attention on H100 GPUs
- Enable FP8 quantization for reduced memory usage
- Configure parallel processing for multi-GPU setups

📖 **For more configuration details, see:**
- [Model Loading Guide](docs/en/model_loading.md) - Using ModuleManager for model loading
- [Adding New Models](docs/en/adding_new_model.md) - Guide for integrating new models
- [Ops Module Documentation](docs/en/ops.md) - Neural network operator implementations
- [Parallel Inference Guide](docs/en/parallel.md) - Distributed parallel inference architecture and usage
- [Attention Implementation Guide](docs/en/attention.md) - Attention configuration and backends
- [Logging Guide](docs/en/logging.md) - Logging system configuration and usage
- [Metrics Guide](docs/en/metrics.md) - Metrics collection and monitoring

## Supported Pipelines

| Pipeline | Task | Features & Performance |
|----------|------|------------------------|
| **WanVideo** (Wan2.1/2.2) | T2V, I2V, FL2V | [Examples README](examples/wan_video/README.md) |
| **Qwen-Image** | T2I, Edit | [Examples README](examples/qwen_image/README.md) |
| **Z-Image** | T2I | [Examples README](examples/z_image/README.md) |
| **HunyuanVideo** | T2V, I2V | [Examples README](examples/hunyuan_video/README.md) |
| **FlashVSR** | VSR | [Examples README](examples/flashvsr/README.md) |
| **LongCat-Video** | T2V, I2V | [Examples README](examples/longcat_video/README.md) |
| **LTX Video** | I2V + Audio | [Examples README](examples/ltx_video/README.md) |
| **Flux2 Klein** | T2I | [Examples README](examples/flux2_klein/README.md) |
| **LiveAct** | S2V | Speech-to-Video (talking head generation) |

> 💡 See each pipeline's README for detailed feature support matrix and performance benchmarks.

## Known Limitations

- **Feature Cache**: Currently supports Wan2.1, Wan2.2, Qwen-Image, and HunyuanVideo models
- **Compile**: Torch compile support is experimental, see [torch.compile Compatibility Guide](docs/en/torch_compile_compatibility.md)
- **Multi-Machine**: Distributed inference across multiple machines is not yet tested
- **GPU Requirements**: Some features (FP8, SageAttention) require specific GPU architectures (H100+)
- **Model Coverage**: Only selected models are tested; other Diffusers models may require adaptation

## Development

We adopt PEP8 as our code style.

Quick setup:

```bash
pip install -e ".[dev]"
pre-commit install
```

📖 **For detailed contribution guidelines, testing documentation, and development workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).**

## License

Apache 2.0 License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please see the project repository for contribution guidelines.

## Acknowledgements

This project builds upon and is inspired by many excellent open-source projects:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Diffusion model engine by ModelScope Community
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) - High-performance diffusion inference engine
- [LightX2V](https://github.com/ModelTC/LightX2V) - Lightweight image/video generation inference framework
- [cache-dit](https://github.com/vipshop/cache-dit) - PyTorch-native caching for DiT models
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan2.2](https://github.com/Wan-Video/Wan2.2) - Video generation models
- [diffusers](https://github.com/huggingface/diffusers) - State-of-the-art diffusion models library
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image) - Image generation foundation model
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) - Photorealistic image generation with bilingual text rendering
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) - Real-time diffusion-based video super-resolution

## Support

For issues and questions, please check the project documentation or create an issue in the repository.
