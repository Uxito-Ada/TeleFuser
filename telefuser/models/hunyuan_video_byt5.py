"""HunyuanVideo ByT5 Glyph Encoder for text rendering in videos.

Based on the Glyph-SDXL-v2 implementation for rendering text in generated videos.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

from telefuser.utils.logging import logger


def extract_glyph_texts(prompt: str) -> list[str]:
    """Extract text within quotes from prompt.

    Glyph texts are quoted strings in the prompt that will be rendered
    with special font/color styling in the generated video.

    Args:
        prompt: Input prompt string containing quoted text.

    Returns:
        List of extracted text strings (deduplicated while preserving order).
    """
    pattern = r'\"(.*?)\"|"(.*?)"'
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    return list(dict.fromkeys(result)) if len(result) > 1 else result


@dataclass
class ByT5Config:
    """Configuration for ByT5 Glyph encoder."""

    byt5_name: str = "google/byt5-small"
    max_length: int = 256
    hidden_dim: int = 1472  # ByT5-small output dimension
    # Paths for special tokens
    color_ann_path: Optional[str] = None
    font_ann_path: Optional[str] = None
    # Checkpoint path
    checkpoint_path: Optional[str] = None


class MultilingualPromptFormat:
    """Format prompts with multilingual font and color tokens."""

    def __init__(
        self,
        font_path: Optional[str] = None,
        color_path: Optional[str] = None,
    ):
        """Initialize the prompt formatter.

        Args:
            font_path: Path to font index JSON
            color_path: Path to color index JSON
        """
        self.idx_font_dict = {}
        self.idx_color_dict = {}

        if font_path and Path(font_path).exists():
            with open(font_path, "r") as f:
                self.idx_font_dict = json.load(f)

        if color_path and Path(color_path).exists():
            with open(color_path, "r") as f:
                self.idx_color_dict = json.load(f)

    def format_prompt(
        self,
        glyph_texts: list[str],
        text_styles: list[dict],
    ) -> str:
        """Format glyph texts with style tokens.

        Args:
            glyph_texts: List of text strings to render
            text_styles: List of style dicts with 'color' and 'font-family' keys

        Returns:
            Formatted text string with special tokens
        """
        formatted_parts = []

        for text, style in zip(glyph_texts, text_styles):
            # Add color token if specified
            color_token = ""
            if style.get("color") is not None and self.idx_color_dict:
                color_idx = style["color"]
                color_token = f"<color-{color_idx}>"

            # Add font token if specified
            font_token = ""
            if style.get("font-family") is not None and self.idx_font_dict:
                font_family = style["font-family"]
                if font_family in self.idx_font_dict:
                    font_idx = self.idx_font_dict[font_family]
                    # Use first 2 chars for language code
                    lang_code = font_family[:2] if len(font_family) >= 2 else "en"
                    font_token = f"<{lang_code}-font-{font_idx}>"

            formatted_parts.append(f"{color_token}{font_token}{text}")

        return " ".join(formatted_parts)


class HunyuanVideoByT5Encoder(nn.Module):
    """ByT5 encoder for glyph text rendering in videos.

    This module handles:
    1. Loading ByT5 model and tokenizer
    2. Adding special tokens for colors and fonts
    3. Extracting and encoding text from prompts
    """

    def __init__(
        self,
        config: ByT5Config,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        """Initialize the ByT5 encoder.

        Args:
            config: ByT5 configuration
            torch_dtype: Model precision (default: bfloat16)
            device: Device to load the model on
        """
        super().__init__()
        self.config = config
        self.max_length = config.max_length
        self.torch_dtype = torch_dtype
        self.device = device or torch.device("cpu")

        # Load tokenizer and model
        self._load_model(config, torch_dtype, device)

        # Initialize prompt formatter
        self.prompt_format = MultilingualPromptFormat(
            font_path=config.font_ann_path,
            color_path=config.color_ann_path,
        )

        logger.info(f"Loaded HunyuanVideoByT5Encoder from {config.byt5_name}")

    def _load_model(self, config: ByT5Config, torch_dtype: torch.dtype, device: Optional[torch.device]):
        """Load ByT5 model and tokenizer.

        Args:
            config: ByT5 configuration
            torch_dtype: Model precision
            device: Device to load the model on
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.byt5_name)

        # Load model (encoder only)
        model = T5ForConditionalGeneration.from_pretrained(config.byt5_name, torch_dtype=torch_dtype)
        self.model = model.get_encoder()

        # Add special tokens
        self._add_special_tokens(config)

        # Load custom checkpoint if provided
        if config.checkpoint_path and Path(config.checkpoint_path).exists():
            self._load_checkpoint(config.checkpoint_path)

        self.model.requires_grad_(False)
        self.model.eval()

        if device is not None:
            self.model = self.model.to(device)

    def _add_special_tokens(self, config: ByT5Config):
        """Add special tokens for colors and fonts.

        Args:
            config: ByT5 configuration
        """
        additional_special_tokens = []

        # Add color tokens
        if config.color_ann_path and Path(config.color_ann_path).exists():
            with open(config.color_ann_path, "r") as f:
                idx_color_dict = json.load(f)
            color_tokens = [f"<color-{i}>" for i in range(len(idx_color_dict))]
            additional_special_tokens.extend(color_tokens)

        # Add font tokens (multilingual)
        if config.font_ann_path and Path(config.font_ann_path).exists():
            with open(config.font_ann_path, "r") as f:
                idx_font_dict = json.load(f)
            font_tokens = [f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>" for font_code in idx_font_dict]
            additional_special_tokens.extend(font_tokens)

        if additional_special_tokens:
            self.tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            logger.info(f"Added {len(additional_special_tokens)} special tokens")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load custom checkpoint weights.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if "state_dict" in state_dict:
            sd = state_dict["state_dict"]
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("module.text_tower.encoder."):
                    new_sd[k[len("module.text_tower.encoder.") :]] = v
            state_dict = new_sd

        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded ByT5 checkpoint from {checkpoint_path}")

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.model.parameters()).dtype

    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text for ByT5.

        Args:
            text: Text to tokenize
            max_length: Maximum sequence length

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        max_length = max_length or self.max_length

        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return inputs["input_ids"], inputs["attention_mask"]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokens to embeddings.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            ByT5 embeddings tensor
        """
        outputs = self.model(
            input_ids.to(self.device),
            attention_mask=attention_mask.float().to(self.device),
        )
        return outputs.last_hidden_state

    def process_prompt(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single prompt to get ByT5 embeddings.

        Args:
            prompt: Input prompt string
            device: Target device for tensors

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        device = device or self.device

        # Default: zero embeddings if no glyph text
        embeddings = torch.zeros((1, self.max_length, self.config.hidden_dim), device=device)
        mask = torch.zeros((1, self.max_length), device=device, dtype=torch.int64)

        # Extract glyph texts
        glyph_texts = extract_glyph_texts(prompt)

        if glyph_texts:
            # Format with style tokens (default: no color/font)
            text_styles = [{"color": None, "font-family": None} for _ in glyph_texts]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)

            # Tokenize and encode
            input_ids, attention_mask = self.tokenize(formatted_text)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            embeddings = self.encode(input_ids, attention_mask)
            mask = attention_mask

        return embeddings, mask

    def forward(
        self,
        prompt: str | list[str],
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to process prompts.

        Args:
            prompt: Single prompt or list of prompts
            device: Target device for tensors

        Returns:
            Dictionary with 'byt5_text_states' and 'byt5_text_mask'
        """
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        embeddings_list = []
        masks_list = []

        for p in prompts:
            emb, mask = self.process_prompt(p, device)
            embeddings_list.append(emb)
            masks_list.append(mask)

        embeddings = torch.cat(embeddings_list, dim=0)
        masks = torch.cat(masks_list, dim=0)

        return {
            "byt5_text_states": embeddings,
            "byt5_text_mask": masks,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 256,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "HunyuanVideoByT5Encoder":
        """Load HunyuanVideoByT5Encoder from model root directory.

        Args:
            pretrained_model_name_or_path: Root directory containing text_encoder/Glyph-SDXL-v2
            torch_dtype: Model precision (default: bfloat16)
            max_length: Maximum sequence length (default: 256)
            device: Device to load the model on
            **kwargs: Ignored for compatibility

        Returns:
            Loaded HunyuanVideoByT5Encoder instance
        """
        model_root = pretrained_model_name_or_path
        text_encoder_dir = Path(model_root) / "text_encoder"

        config = ByT5Config(
            byt5_name=str(text_encoder_dir / "byt5-small"),
            max_length=max_length,
            hidden_dim=1472,
            color_ann_path=str(text_encoder_dir / "Glyph-SDXL-v2" / "assets" / "color_idx.json"),
            font_ann_path=str(text_encoder_dir / "Glyph-SDXL-v2" / "assets" / "multilingual_10-lang_idx.json"),
            checkpoint_path=str(text_encoder_dir / "Glyph-SDXL-v2" / "checkpoints" / "byt5_model.pt"),
        )

        # Fall back to HuggingFace if local path doesn't exist
        if not Path(config.byt5_name).exists():
            config.byt5_name = "google/byt5-small"
            logger.warning(f"ByT5 not found locally, using {config.byt5_name}")

        return cls(config, torch_dtype=torch_dtype, device=device)


def load_glyph_byT5_v2(args: dict, device) -> dict:
    """Load ByT5 tokenizer and encoder model for glyph encoding.

    This function is compatible with the original HunyuanVideo API.

    Args:
        args: Configuration dictionary containing:
            - byT5_google_path: Path to ByT5 model (e.g., "google/byt5-small" or local path)
            - byT5_ckpt_path: Path to ByT5 checkpoint
            - byt5_max_length: Maximum sequence length
            - multilingual_prompt_format_color_path: Path to color annotation JSON
            - multilingual_prompt_format_font_path: Path to font annotation JSON
        device: Device to load the model onto

    Returns:
        Dictionary with keys:
            - byt5_tokenizer: The ByT5 tokenizer
            - byt5_model: The ByT5 encoder model
            - byt5_max_length: Maximum sequence length
    """
    byt5_max_length = args.get("byt5_max_length", 256)

    config = ByT5Config(
        byt5_name=args.get("byT5_google_path", "google/byt5-small"),
        max_length=byt5_max_length,
        hidden_dim=1472,
        color_ann_path=args.get("multilingual_prompt_format_color_path"),
        font_ann_path=args.get("multilingual_prompt_format_font_path"),
        checkpoint_path=args.get("byT5_ckpt_path"),
    )

    encoder = HunyuanVideoByT5Encoder(config, torch_dtype=torch.bfloat16, device=device)

    return {
        "byt5_tokenizer": encoder.tokenizer,
        "byt5_model": encoder.model,
        "byt5_max_length": byt5_max_length,
    }
