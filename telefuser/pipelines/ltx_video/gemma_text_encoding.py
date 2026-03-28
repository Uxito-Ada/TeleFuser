from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx_dit import find_matching_file
from telefuser.models.ltx_gemma_text_encoder import LTXEmbeddingsProcessor
from telefuser.models.t5_tokenizer import HuggingfaceTokenizer
from telefuser.utils.profiler import ProfilingContext4Debug

ImageSource = str | Image.Image | torch.Tensor


class GemmaTextEncodingStage(BaseStage):
    """Text encoding stage for LTX video using Gemma and embeddings processor."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        max_txt_length: int = 1024,
    ):
        super().__init__(name, model_runtime_config)
        text_encoder_model_and_path = module_manager.fetch_module("gemma_text_encoder", require_model_path=True)
        self.text_encoder, text_encoder_paths = text_encoder_model_and_path
        tokenizer_path = os.path.dirname(text_encoder_paths[0])
        self.tokenizer = HuggingfaceTokenizer(tokenizer_path, max_txt_length, None)
        self.embeddings_processor: LTXEmbeddingsProcessor = module_manager.fetch_module("ltx_embeddings_processor")
        self.model_names = ["text_encoder", "embeddings_processor"]

    @staticmethod
    def _decode_image(image: ImageSource) -> np.ndarray:
        if isinstance(image, str):
            return np.array(Image.open(image).convert("RGB"))[..., :3]
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))[..., :3]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported image source type: {type(image)!r}")

        image_tensor = image.detach().cpu()
        if image_tensor.ndim == 4:
            if image_tensor.shape[0] != 1:
                raise ValueError(f"Expected a single image tensor batch, got shape {tuple(image_tensor.shape)}.")
            image_tensor = image_tensor[0]
        if image_tensor.ndim != 3:
            raise ValueError(f"Expected an image tensor with 3 dimensions, got shape {tuple(image_tensor.shape)}.")
        if image_tensor.shape[-1] not in {1, 3, 4} and image_tensor.shape[0] in {1, 3, 4}:
            image_tensor = image_tensor.permute(1, 2, 0)
        if image_tensor.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"Unsupported image tensor shape {tuple(image_tensor.shape)}.")

        if image_tensor.is_floating_point():
            if torch.amin(image_tensor) < 0:
                image_tensor = (image_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5
            elif torch.amax(image_tensor) <= 1.0:
                image_tensor = image_tensor.clamp(0.0, 1.0) * 255.0
            else:
                image_tensor = image_tensor.clamp(0.0, 255.0)
        else:
            image_tensor = image_tensor.clamp(0, 255)
        return image_tensor.to(torch.uint8).numpy()[..., :3]

    @staticmethod
    def _resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")
        _, _, src_h, src_w = tensor.shape
        scale = max(height / src_h, width / src_w)
        new_h = int(np.ceil(src_h * scale))
        new_w = int(np.ceil(src_w * scale))
        tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        crop_top = (new_h - height) // 2
        crop_left = (new_w - width) // 2
        return tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width].unsqueeze(2)

    def _resize_aspect_ratio_preserving(self, image: torch.Tensor, long_side: int) -> torch.Tensor:
        height, width = image.shape[-3:-1]
        scale = long_side / float(max(height, width))
        resized = self._resize_and_center_crop(image, int(height * scale), int(width * scale))
        result = resized.permute(0, 2, 3, 4, 1)[0]
        return result[0] if result.shape[0] == 1 else result

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def _default_gemma_t2v_system_prompt() -> str:
        return GemmaTextEncodingStage._load_system_prompt("gemma_t2v_system_prompt.txt")

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def _default_gemma_i2v_system_prompt() -> str:
        return GemmaTextEncodingStage._load_system_prompt("gemma_i2v_system_prompt.txt")

    @staticmethod
    def _load_system_prompt(prompt_name: str) -> str:
        prompts_dir = Path(__file__).resolve().parents[3] / "models" / "prompts"
        with open(prompts_dir / prompt_name, "r") as f:
            return f.read()

    @staticmethod
    def _cat_with_padding(tensor: torch.Tensor, padding_length: int, value: int) -> torch.Tensor:
        return torch.cat(
            [
                tensor,
                torch.full(
                    (1, padding_length),
                    value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=1,
        )

    @classmethod
    def _pad_inputs_for_attention_alignment(
        cls,
        model_inputs: dict[str, torch.Tensor],
        pad_token_id: int = 0,
        alignment: int = 8,
    ) -> dict[str, torch.Tensor]:
        """Pad sequence length to a multiple of `alignment` for Flash Attention compatibility."""
        seq_len = model_inputs["input_ids"].shape[1]
        padded_len = ((seq_len + alignment - 1) // alignment) * alignment
        padding_length = padded_len - seq_len

        if padding_length > 0:
            model_inputs["input_ids"] = cls._cat_with_padding(model_inputs["input_ids"], padding_length, pad_token_id)
            model_inputs["attention_mask"] = cls._cat_with_padding(model_inputs["attention_mask"], padding_length, 0)
            token_type_ids = model_inputs.get("token_type_ids")
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = cls._cat_with_padding(token_type_ids, padding_length, 0)

        return model_inputs

    def encode_prompt(self, prompt: str):
        """Encode a single prompt into multimodal context embeddings."""
        input_ids, attention_mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return self.embeddings_processor.process_hidden_states(hidden_states, attention_mask)

    @with_model_offload(["text_encoder", "embeddings_processor"])
    @ProfilingContext4Debug("ltx text_encoding")
    @torch.inference_mode()
    def process(
        self,
        prompt_list: list[str],
    ) -> list[object]:
        """Encode prompts and return contexts in input order."""
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            return [self.encode_prompt(prompt) for prompt in prompt_list]
