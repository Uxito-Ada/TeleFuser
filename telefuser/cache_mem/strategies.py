from __future__ import annotations

import io
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from .cache_types import CacheResult, VectorSearchResult
from .config import CacheConfig
from .encoders import Qwen3VLEncoder, Qwen3VLReranker
from .encoding.interfaces import PromptEncoder, VideoEncoder
from .state.interfaces import CacheMetadataManager
from .storage.interfaces import KVStore
from .vector_store.interfaces import VectorStore


class BaseCacheStrategy(ABC):
    """缓存策略抽象基类。"""

    def __init__(
        self,
        config: CacheConfig,
        kv_store: KVStore,
        metadata_manager: CacheMetadataManager,
    ):
        self.config = config
        self.kv_store = kv_store
        self.metadata_manager = metadata_manager

    @abstractmethod
    async def lookup(self, **kwargs) -> CacheResult:
        pass

    async def apply(self, result: CacheResult) -> CacheResult:
        return result

    @abstractmethod
    async def save(self, **kwargs) -> None:
        pass

    def _load_latent(self, cache_id: str, step: int) -> Optional[torch.Tensor]:
        key = f"{cache_id}_step{int(step)}"
        data = self.kv_store.get(key)
        if data is None and "-" in (cache_id or ""):
            normalized = self._normalize_cache_id(cache_id)
            if normalized != cache_id:
                key = f"{normalized}_step{int(step)}"
                data = self.kv_store.get(key)
        if data is None:
            return None
        try:
            # weights_only=True blocks arbitrary code execution from untrusted
            # KV bytes; we only persist tensors here so this is safe.
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.exception(
                "Cache load failed cache_id={} step={} err={}",
                cache_id,
                int(step),
                exc,
            )
            raise RuntimeError(
                f"Cache load failed cache_id={cache_id} step={int(step)} type={type(exc).__name__} err={exc}"
            ) from exc

    def _save_latent(self, cache_id: str, step: int, latent: torch.Tensor) -> None:
        key = f"{cache_id}_step{int(step)}"
        buffer = io.BytesIO()
        try:
            torch.save(latent, buffer)
            self.kv_store.put(key, buffer.getvalue())
        except Exception as exc:
            logger.exception(
                "Cache save failed cache_id={} step={} err={}",
                cache_id,
                int(step),
                exc,
            )
            raise RuntimeError(
                f"Cache save failed cache_id={cache_id} step={int(step)} type={type(exc).__name__} err={exc}"
            ) from exc

    def _latent_size_bytes(self, cache_id: str, step: int, latent: torch.Tensor) -> int:
        nelement = getattr(latent, "nelement", None)
        element_size = getattr(latent, "element_size", None)
        if not callable(nelement) or not callable(element_size):
            raise TypeError(
                "Latent tensor does not expose size methods "
                f"cache_id={cache_id} step={int(step)} type={type(latent).__name__}"
            )
        return int(nelement()) * int(element_size())

    def _generate_cache_id(self) -> str:
        return uuid.uuid4().hex

    def _normalize_cache_id(self, cache_id: str) -> str:
        return (cache_id or "").replace("-", "")

    def _normalize_search_results(self, results: List[VectorSearchResult]) -> None:
        for r in results:
            r.cache_id = self._normalize_cache_id(r.cache_id)

    def _candidate_text(self, result: VectorSearchResult) -> str:
        text = result.prompt or ""
        if not text and isinstance(result.payload, dict):
            text = result.payload.get("prompt") or ""
        return text


class VideoBasedApproximateCache(BaseCacheStrategy):
    def __init__(
        self,
        config,
        kv_store: KVStore,
        vector_store: Optional[VectorStore],
        metadata_manager: CacheMetadataManager,
        *,
        prompt_encoder: Optional[PromptEncoder] = None,
        video_encoder: Optional["VideoEncoder"] = None,
        reranker: Optional[object] = None,
    ):
        super().__init__(config, kv_store, metadata_manager)
        self.vector_store = vector_store

        # Build text / video encoder
        enable_video_embedding = bool(getattr(self.config, "video_embedding_enabled", False))
        text_model_path = getattr(self.config, "text_embedding_model_path", None) or None
        use_text_embedding = bool(text_model_path) or enable_video_embedding

        def _build_prompt_encoder() -> Qwen3VLEncoder:
            model_path = (
                text_model_path
                or getattr(
                    self.config,
                    "video_embedding_model_path",
                    None,
                )
                or "Qwen/Qwen3-VL-Embedding-2B"
            )
            device_id = getattr(self.config, "text_embedding_device_id", None)
            encoder = Qwen3VLEncoder(
                model_path=model_path,
                instruction=getattr(
                    self.config,
                    "text_embedding_instruction",
                    "Represent the user's input",
                ),
                max_frames=int(getattr(self.config, "video_embedding_max_frames", 16)),
                fps=float(getattr(self.config, "video_embedding_fps", 1.0)),
                device_id=device_id,
                torch_dtype=getattr(self.config, "text_embedding_torch_dtype", None),
                attn_implementation=getattr(self.config, "text_embedding_attn_impl", None),
            )
            logger.info(
                "VideoBasedApproximateCache prompt encoder enabled model_path={} device_id={}",
                model_path,
                device_id,
            )
            return encoder

        def _build_video_encoder() -> Qwen3VLEncoder:
            model_path = (
                getattr(
                    self.config,
                    "video_embedding_model_path",
                    None,
                )
                or text_model_path
                or "Qwen/Qwen3-VL-Embedding-2B"
            )
            device_id = getattr(self.config, "video_embedding_device_id", None)
            encoder = Qwen3VLEncoder(
                model_path=model_path,
                instruction=getattr(
                    self.config,
                    "video_embedding_instruction",
                    "Represent the user's input",
                ),
                max_frames=int(getattr(self.config, "video_embedding_max_frames", 16)),
                fps=float(getattr(self.config, "video_embedding_fps", 1.0)),
                device_id=device_id,
                torch_dtype=getattr(self.config, "video_embedding_torch_dtype", None),
                attn_implementation=getattr(self.config, "video_embedding_attn_impl", None),
            )
            logger.info(
                "VideoBasedApproximateCache video encoder enabled model_path={} device_id={}",
                model_path,
                device_id,
            )
            return encoder

        self.prompt_encoder = prompt_encoder
        self.video_encoder = video_encoder

        if use_text_embedding and self.prompt_encoder is None:
            self.prompt_encoder = _build_prompt_encoder()
        if enable_video_embedding and self.video_encoder is None:
            # Qwen3VLEncoder exposes both encode(text) and encode_video(frames)
            # on a single backend embedder. When text and video configs would
            # load the identical model onto the identical device, reuse the
            # prompt_encoder instance to save ~5GB GPU mem and one cold load.
            video_model_path = (
                getattr(self.config, "video_embedding_model_path", None)
                or getattr(self.config, "text_embedding_model_path", None)
                or "Qwen/Qwen3-VL-Embedding-2B"
            )
            video_device_id = getattr(self.config, "video_embedding_device_id", None)
            if (
                self.prompt_encoder is not None
                and getattr(self.prompt_encoder, "model_path", None) == video_model_path
                and getattr(self.prompt_encoder, "device_id", None) == video_device_id
            ):
                self.video_encoder = self.prompt_encoder
                logger.info(
                    "VideoBasedApproximateCache video_encoder shares prompt_encoder "
                    "instance (same model_path={} device_id={}, save ~5GB)",
                    video_model_path,
                    video_device_id,
                )
            else:
                self.video_encoder = _build_video_encoder()

        if use_text_embedding and self.prompt_encoder is None:
            logger.warning(
                "VideoBasedApproximateCache prompt encoder unavailable;"
                " configure text_embedding_model_path or provide prompt_encoder"
            )
        if enable_video_embedding and self.video_encoder is None:
            logger.warning(
                "VideoBasedApproximateCache video encoder unavailable;"
                " configure video embedding or provide video_encoder"
            )

        # Build reranker
        if reranker is not None:
            self.reranker = reranker
        elif getattr(self.config, "rerank_enabled", False):
            self.reranker = Qwen3VLReranker(
                model_path=getattr(self.config, "rerank_model_path", None) or "Qwen/Qwen3-VL-Reranker-2B",
                device_id=getattr(self.config, "rerank_device_id", None),
                batch_size=int(getattr(self.config, "rerank_batch_size", 2) or 2),
                torch_dtype=getattr(self.config, "rerank_torch_dtype", None),
            )
            backend_reranker = getattr(self.reranker, "_reranker", None)
            actual_reranker_device = getattr(getattr(backend_reranker, "model", None), "device", None)
            if actual_reranker_device is None:
                actual_reranker_device = getattr(backend_reranker, "device", "unknown")
            logger.debug(
                "VideoBasedApproximateCache reranker enabled model_path={} device_id={} actual_device={}",
                getattr(self.config, "rerank_model_path", ""),
                getattr(self.config, "rerank_device_id", None),
                actual_reranker_device,
            )
        else:
            self.reranker = None

    async def lookup(self, prompt: str, task_type: str) -> CacheResult:
        prompt = prompt or ""
        logger.debug(f"VideoBasedApproximateCache.lookup start task_type={task_type} prompt_len={len(prompt)}")
        if not prompt:
            logger.debug("VideoBasedApproximateCache.lookup miss: empty prompt")
            return CacheResult(hit=False)
        if self.vector_store is None:
            logger.debug("VideoBasedApproximateCache.lookup miss: vector_store unavailable")
            return CacheResult(hit=False)

        if self.prompt_encoder is None:
            logger.warning("VideoBasedApproximateCache.lookup miss: prompt encoder unavailable")
            return CacheResult(hit=False)

        query_vec = self.prompt_encoder.encode(prompt)
        if not query_vec:
            logger.debug("VideoBasedApproximateCache.lookup miss: prompt embedding unavailable")
            return CacheResult(hit=False)

        hit_score = None
        if getattr(self.config, "rerank_enabled", False):
            top_k = int(getattr(self.config, "rerank_top_k", 1) or 1)
            results = self._vector_search(query_vec, top_k=top_k)
            self._normalize_search_results(results)
            if not results:
                logger.debug("VideoBasedApproximateCache.lookup miss: no vector result")
                return CacheResult(hit=False)
            try:
                self.metadata_manager.record_similarity_scores(
                    request_prompt=prompt,
                    task_type=task_type,
                    cache_type="video_approximate_cache",
                    stage="vector_search",
                    candidates=[
                        {
                            "cache_id": item.cache_id,
                            "similarity": float(item.similarity),
                            "prompt": item.prompt,
                            "saved_steps": item.saved_steps,
                        }
                        for item in results
                    ],
                )
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache record_similarity_scores failed stage=vector_search err_type={} err={}",
                    type(exc).__name__,
                    exc,
                )
            scores = self._rerank_scores(prompt, results, "VideoBasedApproximateCache")
            if scores is None:
                logger.debug("VideoBasedApproximateCache.lookup rerank skip: fallback to vector similarity")
                result = results[0]
                threshold = getattr(self.config, "video_similarity_threshold", 0.10)
                if result.similarity < threshold:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: similarity below threshold "
                        f"sim={result.similarity:.4f} threshold={threshold:.4f}"
                    )
                    return CacheResult(hit=False)
                hit_score = result.similarity
                skip_step = self._determine_skip_step(hit_score, result.saved_steps)
                if skip_step <= 0:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: skip_step=0 "
                        f"sim={result.similarity:.4f} saved_steps={result.saved_steps}"
                    )
                    return CacheResult(hit=False)
            else:
                if len(scores) != len(results):
                    logger.warning(
                        "VideoBasedApproximateCache.lookup rerank invalid scores size={}",
                        len(scores or []),
                    )
                    return CacheResult(hit=False)
                try:
                    self.metadata_manager.record_similarity_scores(
                        request_prompt=prompt,
                        task_type=task_type,
                        cache_type="video_approximate_cache",
                        stage="rerank",
                        candidates=[
                            {
                                "cache_id": item.cache_id,
                                "similarity": float(item.similarity),
                                "rerank_score": float(scores[idx]),
                                "prompt": item.prompt,
                                "saved_steps": item.saved_steps,
                            }
                            for idx, item in enumerate(results)
                        ],
                    )
                except Exception as exc:
                    logger.exception(
                        "VideoBasedApproximateCache record_similarity_scores failed stage=rerank err_type={} err={}",
                        type(exc).__name__,
                        exc,
                    )
                best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
                rerank_score = float(scores[best_idx])
                result = results[best_idx]
                logger.debug(
                    "VideoBasedApproximateCache.lookup rerank select cache_id={} score={:.4f} sim={:.4f}",
                    result.cache_id,
                    rerank_score,
                    result.similarity,
                )
                rerank_threshold = float(getattr(self.config, "rerank_score_threshold", 0.95) or 0.95)
                if rerank_score <= rerank_threshold:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: rerank score below threshold "
                        f"score={rerank_score:.4f} threshold={rerank_threshold:.4f}"
                    )
                    return CacheResult(hit=False)
                hit_score = rerank_score
                skip_step = self._determine_skip_step(hit_score, result.saved_steps)
                if skip_step <= 0:
                    logger.debug(
                        "VideoBasedApproximateCache.lookup miss: skip_step=0 "
                        f"score={rerank_score:.4f} saved_steps={result.saved_steps}"
                    )
                    return CacheResult(hit=False)
        else:
            results = self._vector_search(query_vec, top_k=1)
            if not results:
                logger.debug("VideoBasedApproximateCache.lookup miss: no vector result")
                return CacheResult(hit=False)
            try:
                self.metadata_manager.record_similarity_scores(
                    request_prompt=prompt,
                    task_type=task_type,
                    cache_type="video_approximate_cache",
                    stage="vector_search",
                    candidates=[
                        {
                            "cache_id": item.cache_id,
                            "similarity": float(item.similarity),
                            "prompt": item.prompt,
                            "saved_steps": item.saved_steps,
                        }
                        for item in results
                    ],
                )
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache record_similarity_scores failed stage=vector_search err_type={} err={}",
                    type(exc).__name__,
                    exc,
                )
            result = results[0]

            threshold = getattr(self.config, "video_similarity_threshold", 0.10)
            if result.similarity < threshold:
                logger.debug(
                    "VideoBasedApproximateCache.lookup miss: similarity below threshold "
                    f"sim={result.similarity:.4f} threshold={threshold:.4f}"
                )
                return CacheResult(hit=False)

            hit_score = result.similarity
            skip_step = self._determine_skip_step(hit_score, result.saved_steps)
            if skip_step <= 0:
                logger.debug(
                    "VideoBasedApproximateCache.lookup miss: skip_step=0 "
                    f"sim={result.similarity:.4f} saved_steps={result.saved_steps}"
                )
                return CacheResult(hit=False)

        latent = self._load_latent(result.cache_id, skip_step)
        if latent is None:
            meta = None
            try:
                meta = self.metadata_manager.get_cache_meta(result.cache_id)
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache lookup meta check failed cache_id={} err_type={} err={}",
                    result.cache_id,
                    type(exc).__name__,
                    exc,
                )
            meta_hint = ""
            if meta:
                meta_hint = (
                    f" meta_prompt={meta.get('prompt')} "
                    f"meta_steps={meta.get('saved_steps')} "
                    f"meta_type={meta.get('cache_type')}"
                )
            logger.warning(
                "VideoBasedApproximateCache.lookup miss: hit by threshold but KV missing "
                f"cache_id={result.cache_id} step={skip_step} sim={result.similarity:.4f} "
                f"meta_exists={bool(meta)}{meta_hint}"
            )
            return CacheResult(hit=False)
        self.metadata_manager.record_access(result.cache_id)
        try:
            self.metadata_manager.record_hit_pair(
                request_prompt=prompt,
                cache_id=result.cache_id,
                cached_prompt=result.prompt,
                similarity=float(hit_score if hit_score is not None else result.similarity),
                task_type=task_type,
                cache_type="video_approximate_cache",
                skip_step=skip_step,
            )
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache record_hit_pair failed cache_id={} err_type={} err={}",
                result.cache_id,
                type(exc).__name__,
                exc,
            )
        logger.debug(
            "VideoBasedApproximateCache.lookup hit "
            f"cache_id={result.cache_id} step={skip_step} sim={result.similarity:.4f}"
        )
        return CacheResult(
            hit=True,
            skip_step=skip_step,
            cache_type="video_approximate_cache",
            similarity=result.similarity,
            latent_state=latent,
            cached_prompt=result.prompt,
        )

    async def save(
        self,
        prompt: str,
        latent_states_dict: Dict[int, torch.Tensor],
        num_frames: int,
        task_type: str,
        saved_steps: List[int],
        embedding_video_frames: Optional[List[Any]] = None,
    ) -> None:
        prompt = prompt or ""
        logger.debug(
            "VideoBasedApproximateCache.save start "
            f"task_type={task_type} prompt_len={len(prompt)} saved_steps={saved_steps}"
        )
        if not prompt:
            logger.debug("VideoBasedApproximateCache.save skip: empty prompt")
            return
        if not latent_states_dict or not saved_steps:
            logger.debug("VideoBasedApproximateCache.save skip: no latent_states or saved_steps")
            return

        cache_id = self._generate_cache_id()
        requested_steps = sorted(set(int(s) for s in saved_steps))
        saved_steps = []
        total_bytes = 0
        collection = getattr(self.config, "video_vector_collection", "video")
        vector_written = False
        metadata_attempted = False

        try:
            for step in requested_steps:
                latent = latent_states_dict.get(step)
                if latent is None:
                    continue
                self._save_latent(cache_id, step, latent)
                saved_steps.append(step)
                total_bytes += self._latent_size_bytes(cache_id, step, latent)
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache.save latent persistence failed cache_id={} err={}",
                cache_id,
                exc,
            )
            if saved_steps:
                try:
                    self._remove_saved_latents(cache_id, saved_steps)
                except Exception as cleanup_exc:
                    raise RuntimeError(
                        "VideoBasedApproximateCache.save failed during latent persistence "
                        f"cache_id={cache_id} err={exc}; cleanup_err={cleanup_exc}"
                    ) from exc
            raise RuntimeError(
                f"VideoBasedApproximateCache.save failed during latent persistence cache_id={cache_id} err={exc}"
            ) from exc

        if not saved_steps:
            logger.debug("VideoBasedApproximateCache.save skip: no latent saved")
            return

        size_mb = float(total_bytes) / (1024 * 1024) if total_bytes > 0 else 0.0
        logger.debug(
            "VideoBasedApproximateCache.save stored "
            f"cache_id={cache_id} steps={saved_steps} size_mb={size_mb:.4f} frames={num_frames}"
        )

        if self.vector_store is None:
            logger.warning("VideoBasedApproximateCache.save skip: vector_store unavailable")
            self._remove_saved_latents(cache_id, saved_steps)
            return

        if not embedding_video_frames:
            logger.debug("VideoBasedApproximateCache.save skip: no video frames provided")
            self._remove_saved_latents(cache_id, saved_steps)
            return

        if self.video_encoder is None:
            logger.warning("VideoBasedApproximateCache.save skip: video encoder unavailable")
            self._remove_saved_latents(cache_id, saved_steps)
            return

        try:
            frames = self._load_frames_for_embedding(
                embedding_video_frames=embedding_video_frames,
            )
            if not frames:
                logger.debug("VideoBasedApproximateCache.save skip: sampled frames empty")
                self._remove_saved_latents(cache_id, saved_steps)
                return
            logger.debug(
                "VideoBasedApproximateCache.save frames decoded "
                f"count={len(frames)} size={getattr(frames[0], 'size', None)}"
            )
            video_vec = self.video_encoder.encode_video(frames, prompt=prompt)
            if not video_vec:
                logger.debug("VideoBasedApproximateCache.save skip: video embedding unavailable")
                self._remove_saved_latents(cache_id, saved_steps)
                return

            vector_dim = len(video_vec)
            self.vector_store.ensure_collection(collection, vector_dim)
            logger.debug(f"VideoBasedApproximateCache.save ensure collection={collection} dim={vector_dim}")

            payload = {
                "prompt": prompt,
                "saved_steps": saved_steps,
                "task_type": task_type,
            }
            self.vector_store.upsert(
                collection,
                cache_id,
                video_vec,
                payload,
            )
            vector_written = True
            metadata_attempted = True
            self.metadata_manager.register_cache(
                cache_id,
                prompt,
                saved_steps,
                size_mb,
                num_frames,
                cache_type="video_approximate_cache",
            )
        except Exception as exc:
            logger.exception(
                "VideoBasedApproximateCache.save failed cache_id={} collection={} err={}",
                cache_id,
                collection,
                exc,
            )
            try:
                self._rollback_cache_entry(
                    cache_id=cache_id,
                    saved_steps=saved_steps,
                    collection=collection,
                    remove_vector=vector_written,
                    remove_metadata=metadata_attempted,
                )
            except Exception as rollback_exc:
                raise RuntimeError(
                    "VideoBasedApproximateCache.save failed "
                    f"cache_id={cache_id} collection={collection} err={exc}; "
                    f"rollback_err={rollback_exc}"
                ) from exc
            raise RuntimeError(
                f"VideoBasedApproximateCache.save failed cache_id={cache_id} collection={collection} err={exc}"
            ) from exc
        logger.debug(f"VideoBasedApproximateCache.vector_store upsert collection={collection} cache_id={cache_id}")

    def _remove_saved_latents(self, cache_id: str, saved_steps: List[int]) -> None:
        errors: List[str] = []
        for step in saved_steps:
            try:
                self.kv_store.remove(f"{cache_id}_step{int(step)}")
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache latent cleanup failed cache_id={} step={} err={}",
                    cache_id,
                    int(step),
                    exc,
                )
                errors.append(
                    f"kv remove failed cache_id={cache_id} step={int(step)} type={type(exc).__name__} err={exc}"
                )
        if errors:
            raise RuntimeError("VideoBasedApproximateCache latent cleanup failed: " + "; ".join(errors))

    def _rollback_cache_entry(
        self,
        cache_id: str,
        saved_steps: List[int],
        collection: str,
        remove_vector: bool,
        remove_metadata: bool,
    ) -> None:
        errors: List[str] = []
        if remove_vector and self.vector_store is not None:
            try:
                self.vector_store.delete(collection, [cache_id])
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache vector rollback failed collection={} cache_id={} err={}",
                    collection,
                    cache_id,
                    exc,
                )
                errors.append(
                    "vector rollback failed "
                    f"collection={collection} cache_id={cache_id} "
                    f"type={type(exc).__name__} err={exc}"
                )
        if remove_metadata:
            try:
                self.metadata_manager.remove_cache(cache_id)
            except Exception as exc:
                logger.exception(
                    "VideoBasedApproximateCache metadata rollback failed cache_id={} err={}",
                    cache_id,
                    exc,
                )
                errors.append(f"metadata rollback failed cache_id={cache_id} type={type(exc).__name__} err={exc}")
        try:
            self._remove_saved_latents(cache_id, saved_steps)
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            raise RuntimeError(f"VideoBasedApproximateCache rollback failed cache_id={cache_id}: {'; '.join(errors)}")

    def _vector_search(self, query_vec: List[float], top_k: int = 1) -> List[VectorSearchResult]:
        if self.vector_store is None:
            return []
        collection = getattr(self.config, "video_vector_collection", "video")
        top_k = max(1, int(top_k or 1))
        res = self.vector_store.search(collection, query_vec, limit=top_k)
        if not res:
            return []
        res.sort(key=lambda item: item.similarity, reverse=True)
        return res[:top_k]

    def _load_frames_for_embedding(
        self,
        *,
        embedding_video_frames: Optional[List[Any]],
    ) -> List[Any]:
        if embedding_video_frames:
            return list(embedding_video_frames)
        return []

    def _sample_indices(self, total: int, max_frames: int) -> List[int]:
        if total <= 0:
            return []
        max_frames = max(1, int(max_frames or 1))
        if total <= max_frames:
            return list(range(total))
        step = float(total) / float(max_frames)
        return [min(int(i * step), total - 1) for i in range(max_frames)]

    def _determine_skip_step(self, similarity: float, saved_steps: List[int]) -> int:
        steps = set(int(s) for s in saved_steps)
        rerank_threshold = float(getattr(self.config, "rerank_score_threshold", 0.90) or 0.90)
        if similarity > rerank_threshold and 5 in steps:
            return 5
        return 0

    def _build_rerank_documents(
        self,
        results: List[VectorSearchResult],
    ) -> List[Dict[str, object]]:
        documents: List[Dict[str, object]] = []
        for item in results:
            text = self._candidate_text(item)
            doc: Dict[str, object] = {}
            if text:
                doc["text"] = text
            documents.append(doc)
        return documents

    # def _build_rerank_documents(
    #     self,
    #     results: List[VectorSearchResult],
    # ) -> List[Dict[str, object]]:
    #     documents: List[Dict[str, object]] = []
    #     for item in results:
    #         text = self._candidate_text(item)
    #         doc: Dict[str, object] = {}
    #         if text:
    #             doc["text"] = text
    #         documents.append(doc)
    #     return documents

    def _rerank_scores(
        self,
        query: str,
        results: List[VectorSearchResult],
        source: str,
    ) -> Optional[List[float]]:
        reranker = getattr(self, "reranker", None)
        if reranker is None:
            logger.warning(f"{source} rerank skip: reranker unavailable")
            return None
        if not hasattr(reranker, "score_mm"):
            logger.warning(f"{source} rerank skip: text reranker unavailable")
            return None
        documents = self._build_rerank_documents(results)
        has_text_docs = any("text" in doc and doc["text"] for doc in documents)
        if not has_text_docs:
            logger.debug(f"{source} rerank skip: no text candidates available")
            return None

        try:
            logger.debug(f"{source} rerank mode=text candidates={len(results)}")
            scores = reranker.score_mm({"text": query}, documents)
        except Exception as exc:
            logger.exception(f"{source} text rerank failed: {exc}")
            raise RuntimeError(f"{source} text rerank failed err_type={type(exc).__name__} err={exc}") from exc
        if not scores or len(scores) != len(results):
            raise ValueError(f"{source} rerank invalid scores size={len(scores or [])} expected={len(results)}")
        score_pairs = []
        for idx, item in enumerate(results):
            try:
                score_value = float(scores[idx])
                score_pairs.append(f"{item.cache_id}:{score_value:.4f}/{item.similarity:.4f}")
            except (IndexError, TypeError, ValueError) as exc:
                logger.exception(
                    "{} rerank score formatting failed cache_id={} idx={} err_type={} err={}",
                    source,
                    item.cache_id,
                    idx,
                    type(exc).__name__,
                    exc,
                )
                raise RuntimeError(
                    f"{source} rerank score formatting failed "
                    f"cache_id={item.cache_id} idx={idx} "
                    f"err_type={type(exc).__name__} err={exc}"
                ) from exc
        logger.debug(f"{source} rerank scores={score_pairs}")
        return [float(value) for value in scores]


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(name: str, cls: type) -> None:
    _STRATEGY_REGISTRY[name] = cls


def get_strategy_class(name: str) -> Optional[type]:
    return _STRATEGY_REGISTRY.get(name)


register_strategy("video_approximate", VideoBasedApproximateCache)
