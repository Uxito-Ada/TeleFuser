# HunyuanVideo Example

Text-to-Video and Image-to-Video generation using HunyuanVideo-1.5 model.

## Model Source

| Platform | Link |
|----------|------|
| HuggingFace | [tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| ModelScope | [Tencent-Hunyuan/HunyuanVideo-1.5](https://modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-1.5) |

## Feature Support

| Feature | Support |
|---------|---------|
| CFG Parallel (CFGP) | ❔ |
| Ulysses Sequence Parallel (USP) | ✔️ |
| LoRA | ❔ |
| FP8 Quantization | ❔ |
| FSDP | ❔ |
| Encoder Parallel | ✔️ |
| Async Pipeline | ❔ |
| Feature Cache (AdaTaylor) | ✔️ |
| Distilled Model | N/A |
| Server API | ✔️ |

## Files

### hunyuan_video_t2v.py

Text-to-Video generation example.

**Purpose:** Generate video from text prompt using HunyuanVideo-1.5.

**Features:**
- Optional ByT5 for glyph text rendering (text in quotes will be rendered in video)
- Optional Super-Resolution (SR) for 480p -> 720p upscaling

**Usage:**
```bash
# Basic usage
python examples/hunyuan_video/hunyuan_video_t2v.py --prompt "A beautiful sunset"

# With glyph text rendering
python examples/hunyuan_video/hunyuan_video_t2v.py --enable_byt5 --prompt 'A sunset scene with "Hello" in the sky'

# With Super-Resolution (480p -> 720p)
python examples/hunyuan_video/hunyuan_video_t2v.py --enable_sr --prompt "A beautiful sunset"

# Multi-GPU inference
python examples/hunyuan_video/hunyuan_video_t2v.py --gpu_num 2 --prompt "A beautiful sunset"
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs (1, 2, 4, 8) |
| `--prompt` | (Chinese prompt) | Text prompt |
| `--negative_prompt` | "" | Negative prompt |
| `--seed` | 42 | Random seed |
| `--resolution` | 720p | Target resolution |
| `--aspect_ratio` | 16:9 | Aspect ratio |
| `--model_root` | /root/models/HunyuanVideo-1.5 | Model path |
| `--enable_byt5` | False | Enable glyph text rendering |
| `--enable_sr` | False | Enable super-resolution |

### hunyuan_video_i2v.py

Image-to-Video generation example.

**Purpose:** Generate video from a reference image and text prompt.

**Usage:**
```bash
# Basic usage
python examples/hunyuan_video/hunyuan_video_i2v.py --image_path /path/to/image.jpg --prompt "Make this image move naturally"

# With custom model path
python examples/hunyuan_video/hunyuan_video_i2v.py --image_path /path/to/image.jpg --model_root /path/to/HunyuanVideo-1.5
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs |
| `--image_path` | None | Path to reference image |
| `--prompt` | (Chinese prompt) | Text prompt |
| `--negative_prompt` | "" | Negative prompt |
| `--seed` | 42 | Random seed |
| `--model_root` | /root/models/HunyuanVideo-1.5 | Model path |

### Cache Calibration

#### hunyuan_video_t2v_cache_calibrate.py

Calibration tool for HunyuanVideo T2V AdaTaylorCache.

**Purpose:** Generate calibration parameters for text-to-video feature caching.

**Usage:**
```bash
python examples/hunyuan_video/hunyuan_video_t2v_cache_calibrate.py \
    --model_root /path/to/HunyuanVideo-1.5/ \
    --num_inference_steps 50 \
    --sigma_shift 7.0 \
    --output_path ./cache_params.json
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs |
| `--prompt` | (Chinese prompt) | Text prompt |
| `--seed` | 42 | Random seed |
| `--resolution` | 720p | Target resolution |
| `--aspect_ratio` | 16:9 | Aspect ratio |
| `--model_root` | /root/models/HunyuanVideo-1.5 | Model path |
| `--model_name` | HunyuanVideo-T2V | Model name for output |
| `--output_path` | None | JSON output path |

#### hunyuan_video_i2v_cache_calibrate.py

Calibration tool for HunyuanVideo I2V AdaTaylorCache.

**Purpose:** Generate calibration parameters for image-to-video feature caching.

**Usage:**
```bash
python examples/hunyuan_video/hunyuan_video_i2v_cache_calibrate.py \
    --model_root /path/to/HunyuanVideo-1.5/ \
    --num_inference_steps 50 \
    --sigma_shift 5.0 \
    --output_path ./cache_params.json
```

**Features:**
- Uses 480p_i2v transformer version by default
- Includes vision encoder for image conditioning
- 50 inference steps with CFG scale 6.0

## Performance

### Text-to-Video (hunyuan_video_t2v.py)

| Config | Device | Attn Type|  Steps | Frames | Resolution | Dit Time (s) / iter | Max VRAM (GB) |
|--------|--------|-------|--------|--------|------------|----------|---------------|
| 720p t2v | H100*1|SAGE_ATTN_2_8_8_SM90| 50 | 121 | 1280x720 | 20 | 41 |
| 720p t2v(480p) + SR | H100*1|SAGE_ATTN_2_8_8_SM90| 50 + 6 | 121 | 1280x720 | base 5.36, sr 9 | 35 |

### Image-to-Video (hunyuan_video_i2v.py)

| Config | Device | Attn Type|  Steps | Frames | Resolution | Dit Time (s) / iter | Max VRAM (GB) |
|--------|--------|-------|--------|--------|------------|----------|---------------|
| 720p i2v | H100*1 | SAGE_ATTN_2_8_8 |50 | 121 | 1280*720 | 20 | 45 |

## Notes
- ByT5 enables rendering text within generated videos
- Super-Resolution uses distilled model for fast 480p -> 720p upscaling