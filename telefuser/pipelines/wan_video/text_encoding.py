from __future__ import annotations

import os

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.t5_tokenizer import HuggingfaceTokenizer
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


class TextEncodingStage(BaseStage):
    """Text encoding stage for Wan video using T5-XXL text encoder."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        max_txt_length: int = 512,
    ):
        super().__init__(name, model_runtime_config)
        text_encoder_model_and_path = module_manager.fetch_module("wan_video_text_encoder", require_model_path=True)
        self.text_encoder, text_encoder_path = text_encoder_model_and_path
        tokenizer_path = os.path.join(os.path.dirname(text_encoder_path), "google/umt5-xxl")
        self.tokenizer = HuggingfaceTokenizer(tokenizer_path, max_txt_length, "whitespace")
        if model_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            logger.info("enable sequential cpu offload for wan text encoder")
            self.text_encoder.enable_sequential_cpu_offload(device=self.device, torch_dtype=self.torch_dtype)
        self.model_names = ["text_encoder"]

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode single prompt to embeddings."""
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(self, prompt_list: list[str]):
        """Encode list of prompts to text embeddings."""
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            text_embedding_list = []
            for prompt in prompt_list:
                text_embedding = self.encode_prompt(prompt)
                text_embedding_list.append(text_embedding)
        return text_embedding_list
