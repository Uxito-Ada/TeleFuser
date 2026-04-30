from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

# 被拦截的模块名前缀列表
_CACHE_MODULE_PREFIXES = (
    "telefuser.cache_mem",
    "telefuser.service.cache.cache_service",
    "telefuser.service.cache.cache_factory",
)

_sink_id: Optional[int] = None


def _cache_module_filter(record: dict) -> bool:
    name = record.get("name", "")
    return any(name.startswith(prefix) for prefix in _CACHE_MODULE_PREFIXES)


def setup_cache_log_sink(
    log_dir: str | Path,
    *,
    level: str = "DEBUG",
    rotation: str = "100 MB",
    retention: str = "7 days",
    fmt: str = ("[CACHE] {time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"),
) -> Path:
    global _sink_id

    # 如果已有旧 sink，先清除
    if _sink_id is not None:
        try:
            logger.remove(_sink_id)
        except ValueError:
            pass  # sink 已被其他逻辑移除
        _sink_id = None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cache_service.log"

    _sink_id = logger.add(
        str(log_path),
        filter=_cache_module_filter,
        format=fmt,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,  # 异步写入，不阻塞业务线程
    )

    logger.info(
        "Cache log sink configured: path={} level={} rotation={} retention={}",
        log_path,
        level,
        rotation,
        retention,
    )
    return log_path


def remove_cache_log_sink() -> None:
    global _sink_id
    if _sink_id is not None:
        try:
            logger.remove(_sink_id)
        except ValueError:
            pass
        _sink_id = None


def is_cache_log_sink_active() -> bool:
    return _sink_id is not None
