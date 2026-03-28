# Flux2 Klein Example

High-quality text-to-image generation using FLUX.2-Klein model from Black Forest Labs.

## Model Source

|Model| Platform | Link |
|-----|-----|------|
| Flux2-klein 9B distill 4 step | HuggingFace | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |

## Feature Support

| Feature | Support |
|---------|---------|
| CFG Parallel (CFGP) | N/A |
| Ulysses Sequence Parallel (USP) | N/A |
| LoRA | ❔ |
| FP8 Quantization | ❔ |
| FSDP | N/A |
| Encoder Parallel | N/A |
| Async Pipeline | N/A |
| Feature Cache (AdaTaylor) | N/A |
| Server API | ✔️ |

## Files

### flux2_klein_text_to_image_h100.py

TeleFuser optimized text-to-image generation example.

**Purpose:** High-quality image generation with TeleFuser's internal pipeline implementation.

**Usage:**
```bash
# Basic usage (requires local model path)
python examples/flux2_klein/flux2_klein_text_to_image_h100.py --model_root /path/to/FLUX.2-klein-base-9B

# Custom prompt
python examples/flux2_klein/flux2_klein_text_to_image_h100.py --model_root /path/to/model --prompt "A beautiful landscape"

```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs |
| `--prompt` | (default prompt) | Text prompt |
| `--seed` | 42 | Random seed |
| `--height` | 1024 | Image height (divisible by 16) |
| `--width` | 1024 | Image width (divisible by 16) |
| `--model_root` | None | Local path to model directory (required) |

**Model Directory Structure:**
```
model_root/
├── transformer/     # DiT weights (.safetensors)
├── vae/             # VAE weights
├── text_encoder/    # Text encoder weights
└── tokenizer/       # Tokenizer files
```

**Features:**
- 4 inference steps with CFG scale = 1.0
- BF16 precision
- TORCH_SDPA attention implementation
- Support for multi-GPU parallel inference

### flux2_klein_text_to_image_official.py

Original diffusers pipeline for comparison.

**Purpose:** Reference implementation using diffusers Flux2KleinPipeline.

**Usage:**
```bash
# Basic usage
python examples/flux2_klein/flux2_klein_text_to_image_official.py

# Custom settings
python examples/flux2_klein/flux2_klein_text_to_image_official.py --prompt "A cat" --guidance_scale 2.0
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | (default prompt) | Text prompt |
| `--seed` | 42 | Random seed |
| `--height` | 1024 | Image height (divisible by 16) |
| `--width` | 1024 | Image width (divisible by 16) |
| `--num_inference_steps` | 4 | Number of inference steps |
| `--guidance_scale` | 1.0 | CFG guidance scale |
| `--model_id` | black-forest-labs/FLUX.2-klein-base-9B | HuggingFace model ID or local path |
| `--cache_dir` | None | Cache directory for downloads |

## Performance

### Text-to-Image

| Config | Device | Attn Type | Steps | CFG | Resolution | Total Time (s) /iter | Max VRAM (GB) |
|--------|--------|-----------|-------|-----|------------|-------------------|---------------|
| official BF16 | H100*1 | TORCH_SDPA | 4 | 1.0 | 1024x1024 | 0.93s | 38G |
| T2I BF16 | H100*1 | TORCH_SDPA | 4 | 1.0 | 1024x1024 | 0.84s | 38G |

## Notes

- FLUX.2-Klein is a 9B parameter model requiring significant GPU memory
- Recommended: H100 or A100 with 80GB for single GPU inference
- Use `--gpu_num` for multi-GPU inference to distribute memory load
- Image dimensions must be divisible by 16