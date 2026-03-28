from __future__ import annotations

import html

import ftfy
import regex as re
import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text: str) -> str:
    text = whitespace_clean(basic_clean(text))
    return text


class LongCatTextEncodingStage(BaseStage):
    """Text encoding stage for LongCat video using UMT5-XL text encoder."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        max_sequence_length: int = 512,
    ):
        super().__init__(name, model_runtime_config)
        self.max_sequence_length = max_sequence_length

        self.tokenizer = module_manager.fetch_module("longcat_tokenizer")
        self.text_encoder = module_manager.fetch_module("longcat_text_encoder")

        self.model_names = ["text_encoder"]

        if model_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            logger.info("enable sequential cpu offload for longcat text encoder")
            from telefuser.models.wan_video_text_encoder import T5LayerNorm, T5RelativeEmbedding
            from telefuser.offload import AutoWrappedLinear, AutoWrappedModule, enable_sequential_cpu_offload

            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_sequential_cpu_offload(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

    def _get_prompt_embeds(
        self,
        prompt: str,
        num_videos_per_prompt: int = 1,
    ):
        prompt = prompt_clean(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)

        prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state

        prompt_embeds = prompt_embeds.to(dtype=self.torch_dtype, device=self.device)
        mask = mask.to(device=self.device)

        # [batch, seq, hidden] -> [batch, 1, seq, hidden]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(1 * num_videos_per_prompt, 1, seq_len, -1)

        return prompt_embeds, mask

    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    @with_metrics
    def process(self, prompt_list: list[str]):
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            results = []
            for prompt in prompt_list:
                prompt_embeds, attention_mask = self._get_prompt_embeds(prompt)
                results.append((prompt_embeds, attention_mask))
        return results
