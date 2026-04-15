# FlashVSR Example

Video Super-Resolution using FlashVSR model with streaming inference support.

## Model Source

| Platform | Link |
|----------|------|
| HuggingFace | [lzx1413/FlashVSR-v1.1-BF16](https://huggingface.co/lzx1413/FlashVSR-v1.1-BF16) |
| ModelScope | [lzx1413/FlashVSR-v1.1-BF16](https://modelscope.cn/models/lzx1413/FlashVSR-v1.1-BF16) |

## Feature Support

| Feature | Support |
|---------|---------|
| CFG Parallel (CFGP) | N/A |
| Ulysses Sequence Parallel (USP) | ✔️ |
| LoRA | N/A |
| FP8 Quantization | ❔ |
| FSDP | ✔️ |
| Encoder Parallel | N/A |
| Async Pipeline | ❔ |
| Feature Cache (AdaTaylor) | N/A |
| Distilled Model | N/A |
| Server API | N/A |

## Start

1. install block sparse attention
````bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install

````
2. install tf-kernel refer to tf-kernel sub-folder

## Model Files

The model root directory should contain:
- `flashvsr11_dit_streaming_dmd_5dc619.safetensors` - DiT model
- `TCDecoder.ckpt` - VAE decoder

## Files

### flashvsr_stream.py

Streaming video super-resolution example that processes video in chunks.

**Purpose:** Demonstrates streaming inference for video super-resolution, suitable for processing long videos with limited GPU memory.

**Usage:**
```bash
# Basic usage (auto-detect resolution)
python examples/flashvsr/flashvsr_stream.py -i /path/to/video.mp4 -s 4

# Specify input resolution
python examples/flashvsr/flashvsr_stream.py -i /path/to/video.mp4 -s 4 --height 480 --width 854

# Multi-GPU inference
python examples/flashvsr/flashvsr_stream.py -i /path/to/video.mp4 -s 4 --gpu_num 2

# Custom model path and output
python examples/flashvsr/flashvsr_stream.py \
    -i /path/to/video.mp4 \
    -s 4 \
    --model_root /path/to/FlashVSR-v1.1 \
    -o output.mp4
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i, --input_video` | (required) | Path to input low-quality video |
| `-s, --scale` | 4 | Upscaling factor |
| `-h, --height` | auto | Input video height (auto-detect if not specified) |
| `-w, --width` | auto | Input video width (auto-detect if not specified) |
| `--gpu_num` | 1 | Number of GPUs to use |
| `--model_root` | /dev/shm/zuoxin/flashvsr | Root directory containing model files |
| `-o, --output` | auto | Output video path |
| `--seed` | 0 | Random seed |

## Performance

| Config | Device | Target Resolution | Speed | Max VRAM (GB) |
|--------|--------|------------|-------|---------------|
| flashvsr_stream | 5090*1 | 1920x1080 |1.5it/s (per chunk) | 22 |
| flashvsr_stream | 5090*2 | 1920x1080 |2.1it/s (per chunk) | 20 |

## Notes

- The example uses streaming mode for memory-efficient inference
- `local_range` parameter: 9 for sharper details, 11 for more stable results
- Multi-GPU uses Ulysses sequence parallelism