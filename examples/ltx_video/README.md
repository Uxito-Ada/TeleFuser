# LTX 2.3 Example (Two-Stage Image-to-Video + Audio)

Two-stage Image-to-Video generation using the LTX 2.3 22B checkpoint. This example produces a **single `.mp4`** that
contains **both video and audio**.

## Feature Support

| Feature | Support |
|---------|---------|
| Audio-Video Generation | ✔️ |
| Two-Stage (Stage1 + Upsample + Stage2) | ✔️ |
| Ulysses Sequence Parallel (SP) | ✔️ |
| FSDP | ✔️ |
| LoRA (Stage2 distilled) | ✔️ |
| VAE Parallel | ✔️ |
| Ray VAE Actor | ❔ |

## Files

### ltx23_22b_image_to_video_two_stage_h100.py

**Purpose:** Generate an audio-video clip from a reference image and a prompt using LTX2.3 two-stage denoising.

**Output:** A single `.mp4` with an AAC audio track.

**Usage:**
```bash
# Basic (2 GPUs by default)
python examples/ltx_video/ltx23_22b_image_to_video_two_stage_h100.py \
  --image_path /path/to/image.png \
  --prompt "A stylish little girl gently caressing her dog in a sunny backyard."

# Multi-GPU
python examples/ltx_video/ltx23_22b_image_to_video_two_stage_h100.py \
  --gpu_num 4 \
  --image_path /path/to/image.png \
  --prompt "A cinematic outdoor scene with natural motion."

# Custom model root
python examples/ltx_video/ltx23_22b_image_to_video_two_stage_h100.py \
  --model_root /path/to/LTX-2.3 \
  --image_path /path/to/image.png \
  --prompt "A slow dolly-in shot, soft lighting, realistic motion."
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 2 | Number of GPUs for parallel inference |
| `--image_path` | `examples/data/101235-video-720_0.png` | Reference image path |
| `--prompt` | (preset) | Positive text prompt |
| `--negative_prompt` | `""` | Extra negative prompt appended to the built-in negative prompt |
| `--model_root` | `/nvfile-heatstorage/model_zoo/modelscope/LTX-2.3` | Directory containing checkpoint files |
| `--resolution` | `1080p` | Target resolution preset (`720p`, `1080p`, `2k`, `4k`) |
| `--num_inference_steps` | 30 | Denoising steps for both stages |
| `--num_frames` | 121 | Number of video frames to generate |
| `--seed` | 42 | Random seed |

**Output location:**
- Uses `TELEAI_EXAMPLE_OUTPUT_DIR` if set, otherwise saves to the current directory.

## Model Files Expected

Under `--model_root`, the example expects at least:
- `ltx-2.3-22b-dev.safetensors` (main checkpoint)
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (2x spatial upsampler)
- Gemma text encoder shards (see `gemma_path_list` in the script)
- `ltx-2.3-22b-distilled-lora-384.safetensors` (stage2 LoRA; optional but recommended for quality)

