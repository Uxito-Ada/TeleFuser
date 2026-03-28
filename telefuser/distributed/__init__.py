"""
TeleFuser Distributed Module

This module provides distributed computing utilities for:
- Ulysses: All-to-All based sequence parallelism
- Ring: P2P based sequence parallelism for long sequences
- USP: Combined Ulysses + Ring for large scale
- PP: Pipeline parallelism for model splitting across stages
"""

from __future__ import annotations

from .device_mesh import (
    create_device_mesh_from_config,
    get_attention_strategy,
    get_cfg_group,
    get_cfg_rank,
    get_cfg_world_size,
    get_coordinates,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
    get_group,
    get_mesh_dim_names,
    get_pp_group,
    get_pp_rank,
    get_pp_world_size,
    get_rank,
    get_ranks,
    get_ring_group,
    get_ring_rank,
    get_ring_world_size,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    get_ulysses_group,
    get_ulysses_rank,
    get_ulysses_world_size,
    get_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from .pp_comm import PipelineP2PComm
from .ring import (
    RingP2PComm,
    merge_attn_states,
    ring_attention_allgather_forward,
    ring_attention_forward,
)
from .ulysses_comm import (
    ulysses_gather_heads,
    ulysses_scatter_heads,
)

__all__ = [
    # Device mesh
    "create_device_mesh_from_config",
    "get_attention_strategy",
    "get_cfg_group",
    "get_cfg_rank",
    "get_cfg_world_size",
    "get_coordinates",
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    "get_group",
    "get_mesh_dim_names",
    "get_ring_group",
    "get_ring_rank",
    "get_ring_world_size",
    "get_ranks",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "get_ulysses_group",
    "get_ulysses_rank",
    "get_ulysses_world_size",
    "get_world_size",
    # PP (Pipeline Parallel)
    "get_pp_group",
    "get_pp_rank",
    "get_pp_world_size",
    "is_pipeline_first_stage",
    "is_pipeline_last_stage",
    # Ulysses All-to-All communication
    "ulysses_scatter_heads",
    "ulysses_gather_heads",
    # Ring communication
    "RingP2PComm",
    "merge_attn_states",
    "ring_attention_allgather_forward",
    "ring_attention_forward",
    # PP communication
    "PipelineP2PComm",
]
