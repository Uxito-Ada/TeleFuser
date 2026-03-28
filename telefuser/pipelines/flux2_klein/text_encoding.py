"""Text encoding stage for Flux2 Klein using Qwen3."""

from __future__ import annotations

import os
from typing import List

import torch
from transformers import Qwen2TokenizerFast

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.utils.profiler import ProfilingContext4Debug


class TextEncodingStage(BaseStage):
    """Text encoding stage for Flux2 Klein using Qwen3ForCausalLM.

    Encodes text prompts using Qwen3 with intermediate layer extraction
    (layers 9, 18, 27 by default) and generates 4D position IDs for RoPE.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        text_encoder_and_path = module_manager.fetch_module("text_encoder", require_model_path=True)
        self.text_encoder = None
        self.tokenizer: Qwen2TokenizerFast | None = None

        if text_encoder_and_path is not None:
            self.text_encoder, text_encoder_path = text_encoder_and_path
            if isinstance(text_encoder_path, list):
                text_encoder_path = text_encoder_path[0]
            if self.tokenizer is None:
                tokenizer_path = os.path.join(os.path.dirname(text_encoder_path), "tokenizer")
                if os.path.exists(tokenizer_path):
                    self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)

        self.model_names = ["text_encoder"]
        self.tokenizer_max_length = 512
        # Default intermediate layers for Qwen3
        self.default_hidden_states_layers = (9, 18, 27)

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
        """Generate 4D position coordinates (T, H, W, L) for text tokens.

        Text uses T=0, H=0, W=0, L=[0..seq_len-1].

        Args:
            x: Text embeddings of shape (B, L, D)

        Returns:
            Position IDs of shape (B, L, 4)
        """
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1)  # [0]
            h = torch.arange(1)  # [0]
            w = torch.arange(1)  # [0]
            seq_len = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, seq_len)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def _get_qwen3_prompt_embeds(
        self,
        prompt: str | List[str],
        max_sequence_length: int = 512,
        hidden_states_layers: tuple[int, ...] = (9, 18, 27),
    ) -> torch.Tensor:
        """Encode prompts using Qwen3 with intermediate layer extraction.

        Args:
            prompt: Text prompt or list of prompts
            max_sequence_length: Maximum sequence length for tokenization
            hidden_states_layers: Layer indices to extract and concatenate

        Returns:
            Prompt embeddings of shape (B, seq_len, 3 * hidden_dim)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.device)

        # Forward pass through Qwen3
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract intermediate layers and stack
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=self.torch_dtype, device=self.device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        prompt: str | List[str],
        negative_prompt: str | List[str] | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        hidden_states_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode prompts with optional negative prompts for CFG.

        Args:
            prompt: Positive prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            num_images_per_prompt: Number of images per prompt
            max_sequence_length: Maximum sequence length
            hidden_states_layers: Layer indices to extract

        Returns:
            Tuple of (positive_embeds, positive_ids, negative_embeds, negative_ids)
            where embeds shape is (B*num_images, seq_len, 15360)
            and ids shape is (B*num_images, seq_len, 4)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # Encode positive prompts
        prompt_embeds = self._get_qwen3_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=hidden_states_layers,
        )

        # Expand for num_images_per_prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # Generate position IDs
        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(self.device)

        # Handle negative prompts for CFG
        neg_prompt_embeds = None
        neg_text_ids = None
        if negative_prompt is not None:
            # Expand negative prompt to match positive prompt batch size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) == 1 and batch_size > 1:
                # Broadcast single negative prompt to match batch size
                negative_prompt = negative_prompt * batch_size

            neg_prompt_embeds = self._get_qwen3_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
            neg_prompt_embeds = neg_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            neg_prompt_embeds = neg_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            neg_text_ids = self._prepare_text_ids(neg_prompt_embeds)
            neg_text_ids = neg_text_ids.to(self.device)

        return prompt_embeds, text_ids, neg_prompt_embeds, neg_text_ids
