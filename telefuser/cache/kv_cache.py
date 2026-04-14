"""KVCache module for LiveAct - List-based structure for torch.compile optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class KVCacheConfig:
    """Configuration for KV cache management.

    Attributes:
        fp8_kv_cache: Enable FP8 quantization for memory efficiency
        offload_cache: Offload cache to CPU memory
        cache_frames: Number of frames to cache after compression (default 6)
    """

    fp8_kv_cache: bool = False
    offload_cache: bool = False
    cache_frames: int = 6


class KVCache:
    """Minimal KV cache for a single (timestep, layer) entry.

    Preserves exact original behavior:
    - Direct dict access (k, v, k_scale, v_scale)
    - FP8 quantization support
    - CPU offload support
    """

    def __init__(
        self,
        fp8_kv_cache: bool = False,
        offload_cache: bool = False,
    ):
        """Initialize KV cache.

        Args:
            fp8_kv_cache: Enable FP8 quantization
            offload_cache: Enable CPU offload
        """
        self.fp8_kv_cache = fp8_kv_cache
        self.offload_cache = offload_cache

        # Storage tensors
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.k_scale: torch.Tensor | None = None
        self.v_scale: torch.Tensor | None = None

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype, device: str | torch.device) -> None:
        """Allocate cache tensors.

        Args:
            shape: Shape of K/V tensor [batch, seq, heads, head_dim]
            dtype: Storage dtype (bf16 or fp8)
            device: Device for storage
        """
        storage_dtype = torch.float8_e4m3fn if self.fp8_kv_cache else dtype
        self.k = torch.zeros(shape, dtype=storage_dtype, device=device)
        self.v = torch.zeros(shape, dtype=storage_dtype, device=device)

        if self.fp8_kv_cache:
            # Scale shape: [batch, seq, heads, 1]
            scale_shape = (shape[0], shape[1], shape[2], 1)
            self.k_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
            self.v_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)

    def clear(self) -> None:
        """Clear cache."""
        self.k = None
        self.v = None
        self.k_scale = None
        self.v_scale = None

    def load(self, device: str | torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Load K/V tensors to compute device.

        Args:
            device: Target device
            dtype: Target dtype

        Returns:
            (k, v) tensors on device
        """
        # Move to device if offloaded
        if self.offload_cache:
            self._move_to_device(device)

        # Dequantize if FP8
        if self.fp8_kv_cache:
            k = self._dequantize(self.k, self.k_scale, dtype)
            v = self._dequantize(self.v, self.v_scale, dtype)
        else:
            if self.k.dtype != dtype:
                self.k = self.k.to(dtype=dtype)
            if self.v.dtype != dtype:
                self.v = self.v.to(dtype=dtype)
            k = self.k
            v = self.v

        return k, v

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store K/V tensors.

        Args:
            k: Key tensor
            v: Value tensor
        """
        if self.fp8_kv_cache:
            self.k, self.k_scale = self._quantize(k)
            self.v, self.v_scale = self._quantize(v)
        else:
            self.k = k
            self.v = v

        if self.offload_cache:
            self._move_to_device("cpu")

    def _move_to_device(self, device: str | torch.device) -> None:
        """Move cache tensors to device."""
        self.k = self.k.to(device=device, non_blocking=True)
        self.v = self.v.to(device=device, non_blocking=True)
        if self.k_scale is not None:
            self.k_scale = self.k_scale.to(device=device, non_blocking=True)
        if self.v_scale is not None:
            self.v_scale = self.v_scale.to(device=device, non_blocking=True)

    def _quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP8."""
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = tensor.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = torch.clamp(scale / fp8_max, min=1e-12)
        q_tensor = (tensor / scale.to(dtype=tensor.dtype)).to(torch.float8_e4m3fn)
        return q_tensor.contiguous(), scale.contiguous()

    def _dequantize(self, q_tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize FP8 tensor."""
        return q_tensor.to(dtype=dtype) * scale.to(device=q_tensor.device, dtype=dtype)

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility."""
        return {
            "k": self.k,
            "v": self.v,
            "k_scale": self.k_scale,
            "v_scale": self.v_scale,
            "fp8_kv_cache": self.fp8_kv_cache,
            "offload_cache": self.offload_cache,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KVCache":
        """Create from dict."""
        cache = cls(
            fp8_kv_cache=d.get("fp8_kv_cache", False),
            offload_cache=d.get("offload_cache", False),
        )
        cache.k = d.get("k")
        cache.v = d.get("v")
        cache.k_scale = d.get("k_scale")
        cache.v_scale = d.get("v_scale")
        return cache


class KVCacheManager:
    """Manager for nested KV cache structure (timestep -> layer -> KVCache).

    Uses list structure for optimal torch.compile performance:
    - List indexing is faster than dict hashing
    - No graph breaks from dynamic dict keys
    - Memory contiguous for compiler optimization

    Usage:
        config = KVCacheConfig(fp8_kv_cache=False, offload_cache=True, cache_frames=6)
        manager = KVCacheManager.from_dit_model(
            dit_model,
            config=config,
            tokens_per_frame=520,
            num_timesteps=3,
        )
        manager.allocate(device="cuda", dtype=torch.bfloat16)

        # Access cache for specific (t_idx, layer_idx)
        cache = manager.get_cache(t_idx=0, layer_idx=5)
        k, v = cache.load(device, dtype)

        # Get all layer caches for a timestep (returns list)
        kv_list = manager.get_timestep_caches(0)
    """

    def __init__(
        self,
        config: KVCacheConfig,
        num_timesteps: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        sp_size: int = 1,
    ):
        """Initialize KV cache manager.

        Args:
            config: KVCacheConfig instance
            num_timesteps: Number of denoising timesteps
            num_layers: Number of transformer layers
            num_heads: Total number of attention heads
            head_dim: Dimension per head
            sp_size: Sequence parallel world size (heads are sharded)
        """
        self.config = config
        self.num_timesteps = num_timesteps
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sp_size = sp_size

        # Local heads after SP sharding
        self.local_heads = num_heads // sp_size

        # Cache storage: list[timestep][layer] = KVCache
        self._caches: list[list[KVCache]] = []

        # Pre-allocated shape (set during allocate)
        self._shape: tuple[int, ...] | None = None
        self._tokens_per_frame: int | None = None

    @classmethod
    def from_dit_model(
        cls,
        dit_model: Any,
        config: KVCacheConfig,
        tokens_per_frame: int,
        num_timesteps: int = 3,
        sp_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KVCacheManager":
        """Create KV cache manager from DiT model.

        Args:
            dit_model: LiveActDiT model with blocks, num_heads, dim attributes
            config: KVCacheConfig instance
            tokens_per_frame: Number of tokens per frame (h * w)
            num_timesteps: Number of denoising timesteps
            sp_size: Sequence parallel size (None: auto-detect from dit_model)
            device: Target device (None: cuda)
            dtype: Target dtype (None: bf16)

        Returns:
            KVCacheManager instance
        """
        num_layers = len(dit_model.blocks)
        num_heads = dit_model.num_heads
        head_dim = dit_model.dim // dit_model.num_heads

        # Auto-detect sp_size from dit_model if not provided
        if sp_size is None:
            device_mesh = getattr(dit_model, "device_mesh", None)
            if device_mesh is not None:
                from telefuser.distributed.ulysses_comm import get_ulysses_world_size

                sp_size = get_ulysses_world_size(device_mesh) or 1
            else:
                sp_size = 1

        manager = cls(
            config=config,
            num_timesteps=num_timesteps,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            sp_size=sp_size,
        )

        # Allocate immediately if device/dtype provided
        if device is not None and dtype is not None:
            manager.allocate(tokens_per_frame, device, dtype)
        else:
            manager._tokens_per_frame = tokens_per_frame

        return manager

    def allocate(
        self,
        tokens_per_frame: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Allocate all cache tensors.

        Args:
            tokens_per_frame: Number of tokens per frame
            device: Target device (may be offloaded to CPU)
            dtype: Storage dtype
        """
        self._tokens_per_frame = tokens_per_frame

        # Storage device (CPU if offload enabled)
        storage_device = "cpu" if self.config.offload_cache else device
        storage_dtype = torch.float8_e4m3fn if self.config.fp8_kv_cache else dtype

        # Shape: [batch, cache_tokens, local_heads, head_dim]
        cache_tokens = tokens_per_frame * self.config.cache_frames
        self._shape = (1, cache_tokens, self.local_heads, self.head_dim)

        # Create KVCache for each (t_idx, layer_idx) as list structure
        self._caches = []
        for t_idx in range(self.num_timesteps):
            layer_caches = []
            for layer_idx in range(self.num_layers):
                cache = KVCache(
                    fp8_kv_cache=self.config.fp8_kv_cache,
                    offload_cache=self.config.offload_cache,
                )
                cache.allocate(self._shape, storage_dtype, storage_device)
                layer_caches.append(cache)
            self._caches.append(layer_caches)

    def clear(self) -> None:
        """Clear all caches."""
        for layer_caches in self._caches:
            for cache in layer_caches:
                cache.clear()
        self._caches = []
        self._shape = None
        self._tokens_per_frame = None

    def get_cache(self, t_idx: int, layer_idx: int) -> KVCache:
        """Get KVCache for specific timestep and layer.

        Args:
            t_idx: Timestep index
            layer_idx: Layer index

        Returns:
            KVCache instance
        """
        if t_idx >= len(self._caches):
            raise IndexError(f"Timestep {t_idx} out of range (max: {len(self._caches) - 1})")
        if layer_idx >= len(self._caches[t_idx]):
            raise IndexError(
                f"Layer {layer_idx} out of range for timestep {t_idx} (max: {len(self._caches[t_idx]) - 1})"
            )
        return self._caches[t_idx][layer_idx]

    def get_timestep_caches(self, t_idx: int) -> list[KVCache]:
        """Get all layer caches for a timestep.

        Args:
            t_idx: Timestep index

        Returns:
            List of KVCache for each layer
        """
        if t_idx >= len(self._caches):
            raise IndexError(f"Timestep {t_idx} out of range (max: {len(self._caches) - 1})")
        return self._caches[t_idx]

    def to_dict(self) -> dict[int, dict[int, dict]]:
        """Convert to nested dict for serialization/debugging.

        Returns:
            Dict: {t_idx: {layer_idx: {k, v, k_scale, v_scale, ...}}}
        """
        result = {}
        for t_idx, layer_caches in enumerate(self._caches):
            result[t_idx] = {}
            for layer_idx, cache in enumerate(layer_caches):
                result[t_idx][layer_idx] = cache.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "KVCacheManager":
        """Create from nested dict (for deserialization).

        Args:
            d: Nested dict {t_idx: {layer_idx: {k, v, ...}}}

        Returns:
            KVCacheManager instance
        """
        # Extract structure info from dict
        num_timesteps = len(d)
        num_layers = len(d[0]) if num_timesteps > 0 else 0

        # Get config from first entry
        first_entry = d[0][0] if num_timesteps > 0 and num_layers > 0 else {}
        config = KVCacheConfig(
            fp8_kv_cache=first_entry.get("fp8_kv_cache", False),
            offload_cache=first_entry.get("offload_cache", False),
            cache_frames=6,
        )

        # Infer shape from k tensor
        k_tensor = first_entry.get("k")
        if k_tensor is not None:
            shape = k_tensor.shape
            local_heads = shape[2]
            head_dim = shape[3]
            # We don't know num_heads without sp_size, assume sp_size=1
            num_heads = local_heads
        else:
            raise ValueError("Cannot infer shape from empty cache dict")

        manager = cls(
            config=config,
            num_timesteps=num_timesteps,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            sp_size=1,
        )

        # Restore caches as list structure
        for t_idx in range(num_timesteps):
            layer_caches = []
            for layer_idx in range(num_layers):
                layer_caches.append(KVCache.from_dict(d[t_idx][layer_idx]))
            manager._caches.append(layer_caches)

        manager._shape = shape
        return manager

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Cache tensor shape."""
        return self._shape

    @property
    def is_allocated(self) -> bool:
        """Check if caches are allocated."""
        return len(self._caches) > 0
