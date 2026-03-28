"""Switchable LoRA network for dynamic enable/disable during inference.

Provides LoRA modules that can be loaded, enabled, and disabled at runtime
without permanently modifying the base model weights. Used for refinement
LoRA in LongCat video pipeline.

References:
- https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
- https://github.com/meituan-longcat/LongCat-Video/blob/main/longcat_video/modules/lora_utils.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from telefuser.utils.logging import logger

LORA_PREFIX = "lora"
LORA_HYPHEN = "___lorahyphen___"
REFINE_LORA_KEY = "refinement_lora"


def decode_lora_module_name(lora_name: str) -> str:
    """Convert encoded LoRA name back to dotted module path.

    LoRA names use '___lorahyphen___' as separator to avoid dots in state dict keys.
    """
    return lora_name.replace(f"{LORA_PREFIX}{LORA_HYPHEN}", "").replace(LORA_HYPHEN, ".")


class LoRAUPParallel(nn.Module):
    """Parallel LoRA up-projection for n_separate > 1."""

    def __init__(self, blocks: list[nn.Linear]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % len(self.blocks) == 0
        xs = torch.chunk(x, len(self.blocks), dim=-1)
        return torch.cat([self.blocks[i](xs[i]) for i in range(len(self.blocks))], dim=-1)


class LoRAModule(nn.Module):
    """A single LoRA adapter wrapping a Linear module.

    Does NOT replace the original module — instead, the LoRA output is added
    externally via enable_loras() which wraps the module forward.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        n_separate: int = 1,
    ):
        super().__init__()
        self.lora_name = lora_name

        assert org_module.__class__.__name__ == "Linear"
        in_dim = org_module.in_features
        out_dim = org_module.out_features

        if n_separate > 1:
            assert out_dim % n_separate == 0

        self.lora_dim = lora_dim
        if n_separate > 1:
            self.lora_down = nn.Linear(in_dim, n_separate * self.lora_dim, bias=False)
            self.lora_up = LoRAUPParallel(
                [nn.Linear(self.lora_dim, out_dim // n_separate, bias=False) for _ in range(n_separate)]
            )
        else:
            self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        alpha_scale = alpha / self.lora_dim
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if n_separate > 1:
            for block in self.lora_up.blocks:
                nn.init.zeros_(block.weight)
        else:
            nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.use_lora = True

    def set_use_lora(self, use_lora: bool):
        self.use_lora = use_lora


class LoRANetwork(nn.Module):
    """Collection of LoRA modules for a model.

    Each LoRA module wraps a Linear layer. The network can be loaded from
    a state dict and applied to a model via enable_loras()/disable_all_loras().
    """

    def __init__(
        self,
        model: nn.Module,
        lora_network_state_dict_loaded: dict[str, torch.Tensor],
        multiplier: float = 1.0,
        lora_dim: int = 128,
        alpha: float = 64,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.use_lora = True
        self.lora_dim = lora_dim
        self.alpha = alpha

        logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")

        lora_module_names = set()
        n_blocks_map: dict[str, int] = {}  # pre-compute n_separate for each lora module
        for key in lora_network_state_dict_loaded:
            if key.endswith("lora_down.weight"):
                lora_name = key.split(".lora_down.weight")[0]
                lora_module_names.add(lora_name)
            elif ".lora_up.blocks." in key:
                # Count distinct block indices per lora module
                lora_name = key.split(".lora_up.blocks.")[0]
                n_blocks_map[lora_name] = n_blocks_map.get(lora_name, 0) + 1

        # Each block has one weight key, so n_blocks_map already has correct counts
        loras = []
        for lora_name in lora_module_names:
            module_name = decode_lora_module_name(lora_name)
            try:
                module = model
                for part in module_name.split("."):
                    module = getattr(module, part)
            except Exception as e:
                logger.warning(f"Cannot find module: {module_name}, error: {e}")
                continue
            if module.__class__.__name__ != "Linear":
                continue

            # Infer n_separate from pre-computed block counts
            n_separate = n_blocks_map.get(lora_name, 1)

            lora = LoRAModule(
                lora_name,
                module,
                self.multiplier,
                self.lora_dim,
                self.alpha,
                n_separate=n_separate,
            )
            loras.append(lora)

        self.loras = loras
        for lora in self.loras:
            self.add_module(lora.lora_name, lora)
        logger.info(f"create LoRA for model: {len(self.loras)} modules.")

        # Verify no duplicate names
        names = set()
        for lora in self.loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for lora in self.loras:
            lora.multiplier = self.multiplier

    def set_use_lora(self, use_lora: bool):
        self.use_lora = use_lora
        for lora in self.loras:
            lora.set_use_lora(use_lora)


def create_lora_network(
    transformer: nn.Module,
    lora_network_state_dict_loaded: dict[str, torch.Tensor],
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
) -> LoRANetwork:
    """Create a LoRA network from a state dict.

    Args:
        transformer: The base model to create LoRA adapters for.
        lora_network_state_dict_loaded: State dict from safetensors file.
        multiplier: LoRA output multiplier.
        network_dim: LoRA rank dimension.
        network_alpha: LoRA alpha scaling factor.

    Returns:
        LoRANetwork with modules matching the state dict.
    """
    return LoRANetwork(
        transformer,
        lora_network_state_dict_loaded,
        multiplier=multiplier,
        lora_dim=network_dim or 128,
        alpha=network_alpha or 64,
    )
