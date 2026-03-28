"""Audio utilities for saving waveforms."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


def save_wav(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    path: str,
) -> None:
    """Save a waveform as a PCM16 WAV file.

    Args:
        waveform: Shape (samples,), (channels, samples) or (samples, channels). Values are assumed in [-1, 1].
        sample_rate: Sampling rate in Hz.
        path: Destination file path.
    """
    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach().cpu().float().numpy()
    else:
        wav = np.asarray(waveform, dtype=np.float32)

    if wav.ndim == 1:
        wav = wav[None, :]
    elif wav.ndim == 2:
        # Accept both (channels, samples) and (samples, channels).
        if wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
            pass
        elif wav.shape[1] <= 8 and wav.shape[0] > wav.shape[1]:
            wav = wav.T
    else:
        raise ValueError(f"Expected waveform with 1 or 2 dimensions, got shape {wav.shape}.")

    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).round().astype(np.int16)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_channels = int(pcm16.shape[0])
    with wave.open(str(out_path), "wb") as writer:
        writer.setnchannels(num_channels)
        writer.setsampwidth(2)  # 16-bit PCM
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm16.T.tobytes())
