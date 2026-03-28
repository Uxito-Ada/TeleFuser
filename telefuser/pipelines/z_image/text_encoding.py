from __future__ import annotations

import os
from typing import List

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.z_image_text_encoder import ZImageTextEncoder
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


class TextEncodingStage(BaseStage):
    """Text encoding stage for Z-Image with chat template support.

    Applies chat template with thinking enabled and extracts
    second-to-last hidden states for embeddings.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.text_encoder = module_manager.fetch_module("zimage_text_encoder")
        self.tokenizer = module_manager.fetch_module("tokenizer")
        self.model_names = ["text_encoder"]

    def _encode_prompt(
        self,
        prompt: str | List[str],
        device: torch.device | None = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        """Encode prompt(s) with chat template and return embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Apply chat template with thinking enabled
        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Use second-to-last hidden states for embeddings
        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int = 1,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Encode prompts and optionally negative prompts."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=self.device,
            max_sequence_length=max_sequence_length,
        )

        if negative_prompt is not None:
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=self.device,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        # Duplicate for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]
        return prompt_embeds, negative_prompt_embeds
