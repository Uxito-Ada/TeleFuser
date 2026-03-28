"""Text encoding stage for HunyuanVideo pipeline.

This stage works with the TextEncoder from HunyuanVideo repository:
    from hyvideo.models.text_encoders import TextEncoder

Optionally supports ByT5 for glyph text rendering:
    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_byt5 import extract_glyph_texts
from telefuser.utils.logging import logger


class HunyuanVideoTextEncodingStage(BaseStage):
    """Text encoding stage for HunyuanVideo using LLM-based encoder from HunyuanVideo.

    This stage wraps the TextEncoder from hyvideo.models.text_encoders.
    The TextEncoder handles tokenization and encoding with prompt templates.

    Optionally supports ByT5 for glyph text rendering (text in quotes).

    Example:
        from hyvideo.models.text_encoders import TextEncoder, PROMPT_TEMPLATE

        text_encoder = TextEncoder(
            text_encoder_type='llm',
            tokenizer_type='llm',
            text_encoder_path=text_encoder_path,
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE['li-dit-encode-image-json'],
            prompt_template_video=PROMPT_TEMPLATE['li-dit-encode-video-json'],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
        )
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.text_encoder = module_manager.fetch_module("text_encoder")
        self.model_names = ["text_encoder"]

        # Optional ByT5 for glyph text rendering
        self.byt5_model = None
        self.byt5_tokenizer = None
        self.byt5_max_length = 256

        # Try to fetch ByT5 components if available
        try:
            self.byt5_model = module_manager.fetch_module("byt5_model")
            self.byt5_tokenizer = module_manager.fetch_module("byt5_tokenizer")
            logger.info("ByT5 model found, glyph text rendering enabled")
        except KeyError:
            logger.info("ByT5 model not found, glyph text rendering disabled")

    def _process_single_byt5_prompt(self, prompt_text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process.
            device: Target device for tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - byt5_embeddings: Encoded embeddings tensor.
                - byt5_mask: Attention mask tensor.
        """
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device, dtype=self.torch_dtype)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.long)

        if self.byt5_model is None or self.byt5_tokenizer is None:
            return byt5_embeddings, byt5_mask

        glyph_texts = extract_glyph_texts(prompt_text)

        if len(glyph_texts) > 0:
            # Format glyph texts for ByT5 tokenizer
            text_ids, text_mask = self._get_byt5_text_tokens(glyph_texts)
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)

            # Move model to target device if needed
            if self.byt5_model.device != device:
                self.byt5_model = self.byt5_model.to(device)

            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0].to(dtype=self.torch_dtype)
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    def _get_byt5_text_tokens(self, glyph_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize glyph texts for ByT5.

        The text must be formatted as 'Text "{text}"' before tokenization,
        matching the original HunyuanVideo format_prompt logic.

        Args:
            glyph_texts: List of glyph text strings.

        Returns:
            tuple of (text_ids, text_mask)
        """
        # Format text as "Text \"{text}\". " matching HunyuanVideo format_prompt
        # Original: text_prompt = f'Text "{text}"'
        formatted_text = ""
        for text in glyph_texts:
            formatted_text += f'Text "{text}". '

        text_inputs = self.byt5_tokenizer(
            formatted_text,
            max_length=self.byt5_max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return text_inputs.input_ids, text_inputs.attention_mask

    def _prepare_byt5_embeddings(
        self,
        prompts: list[str],
        device: torch.device,
        do_classifier_free_guidance: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Prepare byT5 embeddings for prompts.

        During CFG, DiT forward is called sequentially. We return separate embeddings
        for conditional and unconditional passes.

        Args:
            prompts: List of prompt strings.
            device: Target device for tensors.
            do_classifier_free_guidance: Whether CFG is enabled.

        Returns:
            Dictionary containing:
                - byt5_text_states: Positive ByT5 embeddings (B, L, D)
                - byt5_text_mask: Positive ByT5 attention mask (B, L)
                - byt5_text_states_nega: None for CFG unconditional (DiT skips processing)
                - byt5_text_mask_nega: None for CFG unconditional
        """
        if self.byt5_model is None:
            return {}

        positive_embeddings = []
        positive_masks = []

        for prompt in prompts:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

        byt5_text_states = torch.cat(positive_embeddings, dim=0)
        byt5_text_mask = torch.cat(positive_masks, dim=0)

        result = {
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }

        # Prepare negative (None) for CFG unconditional forward
        # DiT checks `if byt5_text_states is not None` and skips processing
        if do_classifier_free_guidance:
            result["byt5_text_states_nega"] = None
            result["byt5_text_mask_nega"] = None

        return result

    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def process(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        data_type: str = "video",
        max_length: int | None = None,
        cfg_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Encode text prompts to hidden states using HunyuanVideo TextEncoder.

        This method uses the HunyuanVideo TextEncoder which handles:
        1. Prompt template application (for LLM-based encoders)
        2. Tokenization with proper padding and truncation
        3. Encoding with hidden state skip layer support
        4. Optional ByT5 encoding for glyph text rendering

        Args:
            prompt: Positive prompt text (str or list of str)
            negative_prompt: Negative prompt text for CFG
            data_type: 'image' or 'video' - determines which prompt template to use
            max_length: Maximum token length (uses TextEncoder's default if None)
            cfg_scale: CFG scale (determines if CFG is enabled)

        Returns:
            Dictionary containing:
                - prompt_emb_posi: Positive prompt embeddings (B, L, D)
                - prompt_emb_nega: Negative prompt embeddings (B, L, D) or None
                - attention_mask: Attention mask (B, L)
                - byt5_text_states: Positive ByT5 embeddings (B, L, D) or None
                - byt5_text_mask: Positive ByT5 attention mask (B, L) or None
                - byt5_text_states_nega: None for CFG unconditional (DiT skips processing)
                - byt5_text_mask_nega: None for CFG unconditional
        """
        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] if negative_prompt else [""]

        # Get max_length from TextEncoder if not specified
        if max_length is None and hasattr(self.text_encoder, "max_length"):
            max_length = self.text_encoder.max_length

        # Encode positive prompts using HunyuanVideo TextEncoder
        # The TextEncoder's text2tokens handles prompt templates
        batch_encoding = self.text_encoder.text2tokens(prompt, data_type=data_type, max_length=max_length or 1000)

        # Encode to hidden states
        outputs = self.text_encoder.encode(
            batch_encoding,
            data_type=data_type,
            device=self.text_encoder.device,
        )
        prompt_emb_posi = outputs.hidden_state
        attention_mask = outputs.attention_mask

        # Encode negative prompts
        # When CFG is enabled, we MUST encode the uncond prompt (even if empty string)
        # The text encoder handles empty string by using the template with empty user content
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            # Use empty string if negative_prompt is not provided (matching original HunyuanVideo behavior)
            if not negative_prompt or not negative_prompt[0]:
                negative_prompt = [""]

            uncond_encoding = self.text_encoder.text2tokens(
                negative_prompt, data_type=data_type, max_length=max_length or 1000
            )
            neg_outputs = self.text_encoder.encode(
                uncond_encoding,
                data_type=data_type,
                is_uncond=True,
                device=self.text_encoder.device,
            )
            prompt_emb_nega = neg_outputs.hidden_state
            nega_attention_mask = neg_outputs.attention_mask

        else:
            prompt_emb_nega = None
            nega_attention_mask = None

        result = {
            "prompt_emb_posi": prompt_emb_posi,
            "prompt_emb_nega": prompt_emb_nega,
            "attention_mask": attention_mask,
            "nega_attention_mask": nega_attention_mask,
            "byt5_text_states": None,
            "byt5_text_mask": None,
            "byt5_text_states_nega": None,
            "byt5_text_mask_nega": None,
        }

        # Prepare ByT5 embeddings if available
        if self.byt5_model is not None:
            byt5_result = self._prepare_byt5_embeddings(
                prompts=prompt,
                device=self.device,
                do_classifier_free_guidance=do_cfg,
            )
            result["byt5_text_states"] = byt5_result["byt5_text_states"]
            result["byt5_text_mask"] = byt5_result["byt5_text_mask"]
            result["byt5_text_states_nega"] = byt5_result.get("byt5_text_states_nega")
            result["byt5_text_mask_nega"] = byt5_result.get("byt5_text_mask_nega")

        return result
