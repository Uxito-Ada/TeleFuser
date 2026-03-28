# Z-Image Example

High-speed text-to-image generation using Z-Image-Turbo model.

## Model Source

| Platform | Link |
|----------|------|
| HuggingFace | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| ModelScope | [Tongyi-MAI/Z-Image-Turbo](https://modelscope.cn/models/ZhipuAI/Z-Image-Turbo) |

## Feature Support

| Feature | Support |
|---------|---------|
| CFG Parallel (CFGP) | ✔️ |
| Ulysses Sequence Parallel (USP) | ✔️ |
| LoRA | ❔ |
| FP8 Quantization | ✔️ |
| FSDP | ✔️ |
| Encoder Parallel | N/A |
| Async Pipeline | N/A |
| Feature Cache (AdaTaylor) | N/A |
| Distilled Model | ✔️|
| Server API | ✔️ |

## Files

### z_image_turbo_t2i_h100.py

TeleFuser optimized text-to-image generation example.

**Purpose:** High-speed image generation with TeleFuser's internal pipeline implementation.

**Usage:**
```bash
# Basic usage
python examples/z_image/z_image_turbo_t2i_h100.py

# Custom prompt
python examples/z_image/z_image_turbo_t2i_h100.py --prompt "A beautiful landscape"

# Custom aspect ratio
python examples/z_image/z_image_turbo_t2i_h100.py --aspect_ratio 16:9

# Multi-GPU inference
python examples/z_image/z_image_turbo_t2i_h100.py --gpu_num 2
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aspect_ratio` | 16:9 | Image aspect ratio (1:1, 16:9, 9:16, 4:3, etc.) |
| `--gpu_num` | 1 | Number of GPUs |
| `--prompt` | (default prompt) | Text prompt |
| `--output` | auto | Output filename |

**Features:**
- 9 inference steps (8 DiT forwards) for fast generation
- BF16 precision
- CFG scale = 0 (no classifier-free guidance for Turbo models)

## Notes

- Z-Image-Turbo is optimized for 4-8 step inference
- No CFG needed for Turbo models (guidance_scale = 0)
- Supports various aspect ratios with automatic resolution calculation