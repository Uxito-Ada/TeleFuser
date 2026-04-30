from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CacheMode(Enum):
    READ_WRITE = "read_write"  # 读取和写入缓存（默认）
    READ_ONLY = "read_only"  # 仅读取缓存
    WRITE_ONLY = "write_only"  # 仅写入缓存


@dataclass
class CacheConfig:
    """Cache configuration shared across stages/pipelines."""

    # 基础缓存 (Basic cache)
    enable_latent_cache: bool = False
    cache_mode: CacheMode = CacheMode.READ_WRITE  # read_write | read_only | write_only
    latent_cache_dir: str = "./latent_cache"
    max_cache_size_gb: int = 10
    cache_log_enabled: bool = True
    cache_log_dir: Optional[str] = None  # 默认: {latent_cache_dir}/logs
    cache_log_level: str = "DEBUG"
    cache_log_rotation: str = "100 MB"
    cache_log_retention: str = "7 days"

    # KV 存储 (KV store，用于 latent 等键值缓存)
    kv_store_type: str = "local_file"  # "local_file" | "fluxon"
    fluxon_config_path: Optional[str] = ""

    # 向量存储 (Vector store，用于 embedding 检索)
    vector_store_type: str = "faiss"  # "qdrant" | "faiss"
    qdrant_url: Optional[str] = ""
    qdrant_api_key: Optional[str] = None
    faiss_index_dir: Optional[str] = None
    vector_dim: int = 2048  # 向量维度（FAISS 初始化需要，应与 embedding 模型输出维度一致）
    cache_strategy_type: str = "video_approximate"  # 策略类型，对应 STRATEGY_REGISTRY 中的 key

    # 相似度与检索策略 (Similarity & lookup strategy)
    key_steps: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  # 参与缓存复用的 step
    lookup_mode: str = "video"  # 检索模式，如 "video"

    # 文本嵌入 (Prompt/text embedding 模型)
    text_embedding_model_path: str = ""
    text_embedding_instruction: str = "Represent the user's input"
    text_embedding_device_id: Optional[int] = None
    text_embedding_torch_dtype: Optional[str] = None
    text_embedding_attn_impl: Optional[str] = None

    # 视频嵌入 (Video embedding 模型)
    video_embedding_enabled: bool = True
    video_embedding_model_path: str = "Qwen/Qwen3-VL-Embedding-2B"
    video_embedding_instruction: str = "Represent the user's input"
    video_embedding_fps: float = 1.0
    video_embedding_max_frames: int = 16
    video_embedding_max_length: int = 8192
    video_embedding_min_pixels: int = 4096
    video_embedding_max_pixels: int = 1843200
    video_embedding_total_pixels: int = 7864320
    video_embedding_device_id: Optional[int] = None
    video_embedding_torch_dtype: Optional[str] = None
    video_embedding_attn_impl: Optional[str] = None

    # 视频向量检索与重排 (Video vector search & rerank)
    video_similarity_threshold: Optional[float] = 0.10
    video_vector_collection: str = "video"
    rerank_enabled: bool = False
    rerank_model_path: str = "Qwen/Qwen3-VL-Reranker-2B"
    rerank_top_k: int = 5
    rerank_batch_size: int = 2
    rerank_device_id: Optional[int] = None
    rerank_torch_dtype: Optional[str] = None
    rerank_score_threshold: float = 0.90

    # 异步保存 (Async save / write-behind)
    save_async_enabled: bool = True
    save_queue_size: int = 2
    save_on_full: str = "drop"  # drop | sync | downgrade
    save_queue_warn_threshold: int = 8
    vector_wait_warn_s: float = 2.0
    vector_wait_poll_s: float = 0.05
    vector_wait_timeout_s: float = 120.0
    flush_on_shutdown: bool = True
