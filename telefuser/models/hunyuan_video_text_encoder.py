"""HunyuanVideo Text Encoder implementation for TeleFuser.

Uses LLM-based text encoder with chat template support.
Based on the original HunyuanVideo-1.5 implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils.generic import ModelOutput

# Prompt templates for HunyuanVideo
PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the image by detailing the following aspects: "
        "1. The main content and theme of the image. "
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
        "3. The background environment, light, style and atmosphere.",
    },
    {"role": "user", "content": "{}"},
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the video by detailing the following aspects: "
        "1. The main content and theme of the video. "
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
        "4. background environment, light, style and atmosphere. "
        "5. camera angles, movements, and transitions used in the video.",
    },
    {"role": "user", "content": "{}"},
]

PROMPT_TEMPLATE = {
    "li-dit-encode-image-json": {"template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON, "crop_start": -1},
    "li-dit-encode-video-json": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, "crop_start": -1},
}

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """Output from text encoder.

    Args:
        hidden_state: Hidden states from the encoder (B, L, D)
        attention_mask: Attention mask (B, L)
        hidden_states_list: Tuple of hidden states from all layers (optional)
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None


def use_default(value, default):
    """Utility: return value if not None, else default."""
    return value if value is not None else default


class HunyuanVideoTextEncoder(nn.Module):
    """Text encoder using LLM for HunyuanVideo.

    This is compatible with the original HunyuanVideo TextEncoder API.

    Args:
        text_encoder_type: Type of text encoder (must be "llm")
        max_length: Maximum sequence length
        text_encoder_precision: Precision for text encoder ("fp16", "bf16", "fp32")
        text_encoder_path: Path to text encoder model
        tokenizer_type: Type of tokenizer
        tokenizer_path: Path to tokenizer
        output_key: Key for output from model
        use_attention_mask: Whether to use attention mask
        prompt_template: Prompt template dict for image encoding
        prompt_template_video: Prompt template dict for video encoding
        hidden_state_skip_layer: Number of layers to skip for hidden state
        apply_final_norm: Whether to apply final norm when using skip layer
        reproduce: Whether to use deterministic sampling
        device: Device to load model on
    """

    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.use_attention_mask = use_attention_mask

        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert isinstance(self.prompt_template, dict) and "template" in self.prompt_template, (
                f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            )
            assert "{}" in str(self.prompt_template["template"]), (
                f"`prompt_template['template']` must contain a placeholder {{}} for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert isinstance(self.prompt_template_video, dict) and "template" in self.prompt_template_video, (
                    f"`prompt_template_video` must be a dictionary with a key 'template', got \
                    {self.prompt_template_video}"
                )
            assert "{}" in str(self.prompt_template_video["template"]), (
                f"`prompt_template_video['template']` must contain a placeholder {{}} for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if text_encoder_type != "llm":
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.output_key = output_key or "last_hidden_state"

        # Load model
        self.model = AutoModel.from_pretrained(self.model_path)

        # Handle different model structures
        if hasattr(self.model, "language_model"):
            self.model = self.model.language_model
        self.model.final_layer_norm = self.model.norm

        # Apply precision
        if self.precision is not None:
            self.model = self.model.to(dtype=PRECISION_TO_TYPE[self.precision])

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, padding_side="right")

        # Freeze model
        self.model.requires_grad_(False)
        self.model.eval()

        if device is not None:
            self.model = self.model.to(device)

        # Pre-calculate crop_start for image and video
        if self.use_template and self.prompt_template is not None:
            self.text2tokens("a photo of a cat", data_type="image")
        if self.use_video_template and self.prompt_template_video is not None:
            self.text2tokens("a photo of a cat", data_type="video")

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """Apply text to template.

        Args:
            text: Input text.
            template: Template string or list of chat conversation.
            prevent_empty_text: If True, prevent empty user text by adding a space.
        """
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            # For JSON list template format (chat conversation)
            template_copy = copy.deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    item["content"] = item["content"].format(text if text else (" " if prevent_empty_text else ""))
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def calculate_crop_start(self, tokenized_input) -> int:
        """Automatically calculate the crop_start position.

        This identifies where the user prompt starts after the system message.

        Args:
            tokenized_input: Output from tokenizer containing input_ids

        Returns:
            Position where the actual prompt content begins
        """
        input_ids = tokenized_input["input_ids"][0].tolist()

        marker = "<|im_start|>user\n"

        # Tokenize just the marker to get its token IDs
        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]

        # Find the end position of the marker in the input sequence
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i : i + len(marker_tokens)] == marker_tokens:
                return i + len(marker_tokens)

        # If marker not found, try to find based on special tokens
        if hasattr(self.tokenizer, "special_tokens_map"):
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if "user" in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1

        # Default fallback: return 0 (no cropping)
        return 0

    def text2tokens(self, text, data_type="image", max_length=None):
        """Tokenize the input text.

        Args:
            text: Input text (str or list)
            data_type: 'image' or 'video'
            max_length: Maximum sequence length
        """
        max_length = max_length or self.max_length
        tokenize_input_type = "str"

        if self.use_template or self.use_video_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")

            if isinstance(text, (list, tuple)):
                text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

            # First pass: tokenize to find crop_start
            if crop_start == -1:
                temp_kwargs = dict(
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                    return_tensors="pt",
                )

                if tokenize_input_type == "str":
                    temp_tokenized = self.tokenizer(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **temp_kwargs,
                    )
                elif tokenize_input_type == "list":
                    temp_tokenized = self.tokenizer.apply_chat_template(
                        text,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **temp_kwargs,
                    )

                crop_start = self.calculate_crop_start(temp_tokenized)

                # Store the calculated crop_start
                if data_type == "image":
                    self.prompt_template["crop_start"] = crop_start
                else:
                    self.prompt_template_video["crop_start"] = crop_start
        else:
            crop_start = 0

        # Second pass: tokenize with proper max_length
        kwargs = dict(
            truncation=True,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding="max_length",
            return_tensors="pt",
        )

        if tokenize_input_type == "str":
            tokenized_output = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            tokenized_output = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

        return tokenized_output

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        device=None,
        is_uncond=False,
    ) -> TextEncoderModelOutput:
        """Encode tokenized input to hidden states.

        Args:
            batch_encoding: Batch encoding from tokenizer
            use_attention_mask: Whether to use attention mask
            output_hidden_states: Whether to output hidden states
            do_sample: Whether to sample (for decoder-only LLMs)
            hidden_state_skip_layer: Number of layers to skip
            return_texts: Whether to return decoded texts
            data_type: 'image' or 'video'
            device: Target device
            is_uncond: Whether this is unconditional encoding
        """
        device = device if device is not None else self.device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        do_sample = use_default(do_sample, not self.reproduce)

        attention_mask = batch_encoding["attention_mask"].to(device) if use_attention_mask else None

        outputs = self.model(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
        )

        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Apply final norm for intermediate layers
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens
        if self.use_template:
            if data_type == "image":
                crop_start = self.prompt_template.get("crop_start", 0)
            elif data_type == "video":
                crop_start = self.prompt_template_video.get("crop_start", 0)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")

            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None

        if output_hidden_states:
            return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs.hidden_states)

        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        """Full forward: tokenize and encode."""
        batch_encoding = self.text2tokens(text, max_length=self.max_length)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "HunyuanVideoTextEncoder":
        """Load HunyuanVideoTextEncoder from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: Path to text encoder model
            torch_dtype: Model precision (default: bfloat16)
            **kwargs: Ignored for compatibility

        Returns:
            Loaded HunyuanVideoTextEncoder instance
        """
        precision = "bf16" if torch_dtype == torch.bfloat16 else "fp16"

        return cls(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision=precision,
            text_encoder_path=pretrained_model_name_or_path,
            tokenizer_type="llm",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
        )
