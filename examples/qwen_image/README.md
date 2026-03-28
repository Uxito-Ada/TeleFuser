# Qwen-Image Example

Text-to-Image and Image Editing using Qwen-Image models.

## Model Source

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| Qwen-Image | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | [Qwen/Qwen-Image](https://modelscope.cn/models/Qwen/Qwen-Image) |
| Qwen-Image-Lightning | [Qwen/Qwen-Image-Lightning](https://huggingface.co/Qwen/Qwen-Image-Lightning) | [Qwen/Qwen-Image-Lightning](https://modelscope.cn/models/Qwen/Qwen-Image-Lightning) |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | [Qwen/Qwen-Image-Edit](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) |

## Feature Support

| Feature | Support |
|---------|---------|
| CFG Parallel (CFGP) | ✔️ |
| Ulysses Sequence Parallel (USP) | ✔️ |
| LoRA | ✔️ |
| FP8 Quantization | ✔️ |
| FSDP | ✔️ |
| Encoder Parallel | N/A |
| Async Pipeline | ❔ |
| Feature Cache (AdaTaylor) | ✔️ |
| Distilled Model | ✔️ |
| Server API | ✔️ |

## Files

### Text-to-Image Examples

#### qwen_image_t2i_h100.py

Standard T2I generation with Qwen-Image.

**Purpose:** High-quality text-to-image generation.

**Usage:**
```bash
# Basic usage
python examples/qwen_image/qwen_image_t2i_h100.py

# Custom prompt
python examples/qwen_image/qwen_image_t2i_h100.py --prompt "A beautiful sunset over mountains"

# Custom aspect ratio
python examples/qwen_image/qwen_image_t2i_h100.py --aspect_ratio 16:9

# Multi-GPU inference
python examples/qwen_image/qwen_image_t2i_h100.py --gpu_num 2
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aspect_ratio` | 16:9 | Image aspect ratio |
| `--gpu_num` | 1 | Number of GPUs |
| `--prompt` | (default prompt) | Text prompt |
| `--output` | auto | Output filename |

**Features:**
- 50 inference steps
- CFG scale 4.0
- Async CPU offloading for memory efficiency
- CFG parallel for multi-GPU

#### qwen_image_t2i_lora_h100.py

T2I with Lightning LoRA acceleration.

**Purpose:** Fast generation using LoRA-distilled weights.

**Usage:**
```bash
python examples/qwen_image/qwen_image_t2i_lora_h100.py --prompt "A portrait photo"
```

**Features:**
- 16 inference steps with Lightning LoRA
- CFG scale 1.0 (no guidance needed with distilled model)
- Faster generation with comparable quality

#### qwen_image_t2i_lightning_fp8_h100.py

T2I with FP8 quantized Lightning model.

**Purpose:** Maximum speed with FP8 quantization.

**Usage:**
```bash
python examples/qwen_image/qwen_image_t2i_lightning_fp8_h100.py --prompt "A landscape photo"
```

**Features:**
- FP8 (float8_e4m3fn) quantized weights
- 16 inference steps
- Reduced memory footprint
- Multi-image generation per prompt

### Image Editing Examples

#### qwen_image_edit_plus_h100.py

Image editing with Qwen-Image-Edit.

**Purpose:** Edit images based on text instructions.

**Usage:**
```bash
python examples/qwen_image/qwen_image_edit_plus_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "Change the background to a beach scene"
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aspect_ratio` | 1:1 | Image aspect ratio |
| `--gpu_num` | 1 | Number of GPUs |
| `--prompt` | (default prompt) | Edit instruction |
| `--image_path` | (default image) | Input image path |
| `--output` | auto | Output filename |

**Features:**
- 40 inference steps
- CFG scale 4.0
- Supports complex editing instructions

### Cache Calibration

#### qwen_image_cache_calibrate.py

Calibration tool for Qwen-Image T2I AdaTaylorCache.

**Purpose:** Generate calibration parameters for text-to-image feature caching.

**Usage:**
```bash
python examples/qwen_image/qwen_image_cache_calibrate.py \
    --model_root /path/to/Qwen-Image-2512/ \
    --num_inference_steps 50 \
    --output_path ./cache_params.json
```

**Output:**
Generates a JSON file with magnitude ratios for skip decisions.

#### qwen_image_edit_plus_cache_calibrate.py

Calibration tool for Qwen-Image-Edit-Plus AdaTaylorCache.

**Purpose:** Generate calibration parameters for image editing feature caching.

**Usage:**
```bash
python examples/qwen_image/qwen_image_edit_plus_cache_calibrate.py \
    --model_root /path/to/Qwen-Image-Edit-2511/ \
    --num_inference_steps 40 \
    --output_path ./cache_params.json
```

**Features:**
- Uses `is_edit_plus=True` mode for optimal editing calibration
- 40 inference steps with CFG scale 4.0

## Performance

### Text-to-Image

| Config | Device | Attn Type| Steps | CFG | Resolution | Dit Time (s) /iter | Max VRAM (GB) |
|--------|--------|-------|-----|----|------------|----------|---------------|
| T2I BF16 | H100*1 | SAGE_ATTN_2_8_8_SM90|40 | 4.0 | 1328x1328 | 0.72 | 45 |
| T2I Lightning FP8 | H100*1 |  SAGE_ATTN_2_8_8_SM90|16 | 1.0 | 1328x1328 | 0.33 | 38 |

### Image Editing

| Config | Device | Steps | CFG | Resolution | Dit Time (s) /iter | Max VRAM (GB) |
|--------|--------|-------|-----|------------|----------|---------------|
| Edit BF16 | H100*1 | 40 | 4.0 | 1184x896 | 1 |  60|

## Supported Aspect Ratios

| Aspect Ratio | Resolution |
|--------------|------------|
| 1:1 | 1328x1328 |
| 16:9 | 1664x928 |
| 9:16 | 928x1664 |
| 4:3 | 1472x1104 |
| 3:4 | 1104x1472 |
| 3:2 | 1584x1056 |
| 2:3 | 1056x1584 |