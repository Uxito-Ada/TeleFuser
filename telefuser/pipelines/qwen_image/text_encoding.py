from __future__ import annotations

import os
from typing import List, Optional

import torch
from transformers import Qwen2TokenizerFast, Qwen2VLProcessor

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import DeviceMesh, create_device_mesh_from_config, get_tp_ranks
from telefuser.distributed.tp_parallelize import parallelize_module
from telefuser.metrics import with_metrics
from telefuser.models.qwen_image_text_encoder import QwenImageTextEncoder
from telefuser.utils.profiler import ProfilingContext4Debug


class TextEncodingStage(BaseStage):
    """Text encoding stage for Qwen-Image with prompt templating support.

    Supports both text-only generation and image-conditioned editing with
    specialized prompt templates and vision token handling.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        text_encoder_model_and_path = module_manager.fetch_module("qwen_image_text_encoder", require_model_path=True)
        self.text_encoder: QwenImageTextEncoder | None = None
        self.tokenizer: Qwen2TokenizerFast = module_manager.fetch_module("tokenizer")
        self.processor: Qwen2VLProcessor = module_manager.fetch_module("processor")

        if text_encoder_model_and_path is not None:
            self.text_encoder, text_encoder_path = text_encoder_model_and_path
            if isinstance(text_encoder_path, list):
                text_encoder_path = text_encoder_path[0]
            if self.tokenizer is None:
                tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(text_encoder_path)), "tokenizer")
                self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
            if self.processor is None:
                processor_path = os.path.join(os.path.dirname(os.path.dirname(text_encoder_path)), "processor")
                if os.path.exists(processor_path):
                    self.processor = Qwen2VLProcessor.from_pretrained(processor_path)
        self.model_names = ["text_encoder"]
        self.tokenizer_max_length = 1024
        # Qwen-Image prompt templates for text-only generation
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa
        self.prompt_template_encode_start_idx = 34
        # Qwen-Image-Edit prompt templates with vision tokens
        self.edit_system_prompt = (
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the original."
        )
        self.edit_prompt_template_encode = (
            "<|im_start|>system\n"
            + self.edit_system_prompt
            + "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"  # noqa
        )
        # Qwen-Image-EditPlus uses text-only template
        self.edit_plus_prompt_template_encode = (
            "<|im_start|>system\n"
            + self.edit_system_prompt
            + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )

        self.edit_prompt_template_encode_start_idx = 64

    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text encoding")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        prompt: str | List[str],
        negative_prompt: str | List[str] | None = None,
        num_images_per_prompt: int = 1,
        edit_image: List[torch.Tensor] | None = None,
        is_edit_plus: bool = False,
    ):
        """Encode prompts with optional image conditioning for editing."""
        if edit_image is None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt=prompt, num_images_per_prompt=num_images_per_prompt
            )
            if negative_prompt is not None:
                neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(
                    prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt
                )
            else:
                neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        else:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt_with_image(
                prompt=prompt, num_images_per_prompt=num_images_per_prompt, image=edit_image, is_edit_plus=is_edit_plus
            )
            if negative_prompt is not None:
                neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt_with_image(
                    prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    image=edit_image,
                    is_edit_plus=is_edit_plus,
                )
            else:
                neg_prompt_embeds, neg_prompt_embeds_mask = None, None

        return prompt_embeds, prompt_embeds_mask, neg_prompt_embeds, neg_prompt_embeds_mask

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract hidden states based on attention mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def encode_prompt_with_image(
        self,
        prompt: str | List[str],
        image: List[torch.Tensor],
        num_images_per_prompt: int = 1,
        is_edit_plus: bool = True,
    ):
        """Encode prompt with image conditioning for editing tasks."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt)
        template = self.edit_prompt_template_encode
        drop_idx = self.edit_prompt_template_encode_start_idx
        if not is_edit_plus:
            template = self.edit_prompt_template_encode
            texts = [template.format(txt) for txt in prompt]
        else:
            # EditPlus: include image tokens in the prompt
            template = self.edit_plus_prompt_template_encode
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(image))])
            texts = [template.format(img_prompt + e) for e in prompt]

        model_inputs = self.processor(text=texts, images=image, padding=True, return_tensors="pt").to(self.device)
        hidden_states = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_emb = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        prompt_emb_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
        seq_len = prompt_emb.shape[1]
        # Duplicate for num_images_per_prompt
        prompt_emb = prompt_emb.repeat(1, num_images_per_prompt, 1)
        prompt_emb = prompt_emb.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_emb_mask = prompt_emb_mask.repeat(1, num_images_per_prompt, 1)
        prompt_emb_mask = prompt_emb_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_emb, prompt_emb_mask

    def encode_prompt(
        self,
        prompt: str | List[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ):
        """Encode text-only prompts using templated format."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        prompt_embeds_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

        prompt_embeds = prompt_embeds.to(dtype=self.torch_dtype, device=self.device)
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    def parallel_models(self):
        """Configure tensor parallelism for text encoder."""
        if self.model_runtime_config.parallel_config.tp_degree > 1:
            parallel_cfg = self.model_runtime_config.parallel_config
            device_mesh = create_device_mesh_from_config(parallel_cfg, self.device_type)
            self.text_encoder = parallelize_module(
                self.text_encoder,
                device_mesh=DeviceMesh(self.device_type, torch.tensor(get_tp_ranks(device_mesh))),
                parallelize_plan=self.text_encoder.get_tp_plan(),
            )
