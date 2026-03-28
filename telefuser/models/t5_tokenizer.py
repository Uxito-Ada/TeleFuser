from __future__ import annotations

import html
import string

import ftfy
import regex as re
import torch
from transformers import AutoTokenizer


def basic_clean(text: str) -> str:
    """Clean text using ftfy and HTML unescape."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    """Collapse multiple whitespaces into single space."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text: str, keep_punctuation_exact_string: str | None = None) -> str:
    """Normalize text: lowercase, remove punctuation, collapse whitespace."""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    """Wrapper for HuggingFace tokenizers with optional text cleaning."""

    def __init__(self, name: str, seq_len: int | None = None, clean: str | None = None, **kwargs):
        """
        Args:
            name: Model name or path for AutoTokenizer.
            seq_len: Max sequence length for padding/truncation. None for dynamic.
            clean: Text cleaning mode: None, 'whitespace', 'lower', or 'canonicalize'.
        """
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence: str | list[str], **kwargs) -> torch.Tensor:
        """Tokenize input sequence(s).

        Args:
            sequence: String or list of strings to tokenize.
            return_mask: If True, returns attention mask as well.
        """
        return_mask = kwargs.pop("return_mask", False)

        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update(
                {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self.seq_len,
                }
            )
        _kwargs.update(**kwargs)

        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids

    def _clean(self, text: str) -> str:
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text
