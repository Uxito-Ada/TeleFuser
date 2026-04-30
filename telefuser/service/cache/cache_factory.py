import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from telefuser.utils.utils import import_function_from_file

from .cache_service import CacheService

try:
    from telefuser.cache_mem.log_monitor import setup_cache_log_sink
except Exception:
    setup_cache_log_sink = None

try:
    from telefuser.cache_mem.config import CacheConfig, CacheMode
    from telefuser.cache_mem.latent_cache import LatentCache
except Exception as exc:  # optional dependency for cache service
    _cache_dep_import_error = exc
    _cache_dep_import_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    LatentCache = Any
    CacheConfig = Any
    CacheMode = Any
else:
    _cache_dep_import_error = None
    _cache_dep_import_traceback = None


class CacheServiceFactory:
    """Create CacheService with config parsing and dependency wiring."""

    @staticmethod
    def create_cache_service(
        ppl_file: Optional[str],
        enable_latent_cache: Optional[bool],
        cache_mode: Optional[str] = None,
    ) -> Optional[CacheService]:
        try:
            if CacheConfig is Any or CacheMode is Any or LatentCache is Any:
                unavailable_symbols = (
                    ", ".join(
                        symbol_name
                        for symbol_name, symbol_value in (
                            ("CacheConfig", CacheConfig),
                            ("CacheMode", CacheMode),
                            ("LatentCache", LatentCache),
                        )
                        if symbol_value is Any
                    )
                    or "unknown"
                )
                if _cache_dep_import_error is not None:
                    logger.warning(
                        "Cache config not available, cache service disabled. "
                        "unavailable_symbols={}, import_error_type={}, import_error={}, "
                        "traceback:\n{}",
                        unavailable_symbols,
                        type(_cache_dep_import_error).__name__,
                        _cache_dep_import_error,
                        _cache_dep_import_traceback,
                    )
                else:
                    logger.warning(
                        "Cache config not available, cache service disabled. unavailable_symbols={}",
                        unavailable_symbols,
                    )
                return None

            # ppl_file 必须提供且包含 build_latent_data，否则抛出错误、不初始化 cache_service
            if ppl_file is None:
                raise ValueError(
                    "enable_latent_cache is enabled but no ppl_file provided. "
                    "Please provide a pipeline file that contains the build_latent_data function."
                )
            # 尝试从 ppl_file 读取 CACHE_CONFIG 覆盖项
            ppl_cache_config = None
            ppl_cache_config_load_error = None
            try:
                ppl_cache_config = import_function_from_file(ppl_file, "CACHE_CONFIG")
                logger.info(f"Found CACHE_CONFIG in {ppl_file}")
            except AttributeError:
                ppl_cache_config = None
            except Exception as exc:
                ppl_cache_config_load_error = exc
                logger.warning(f"Failed to load CACHE_CONFIG from {ppl_file}: {exc}")
                ppl_cache_config = None

            # 构建 app_cache_config（默认值 + ppl 覆盖）
            cache_config_source = "CacheConfig"
            if isinstance(ppl_cache_config, CacheConfig):
                app_cache_config = ppl_cache_config
                cache_config_source = "ppl CACHE_CONFIG"
            elif isinstance(ppl_cache_config, dict):
                valid_keys = {field.name for field in fields(CacheConfig)}
                overrides = {k: v for k, v in ppl_cache_config.items() if k in valid_keys}
                unknown_keys = sorted(set(ppl_cache_config.keys()) - valid_keys)
                if unknown_keys:
                    logger.warning(f"Ignore unknown CACHE_CONFIG keys: {', '.join(unknown_keys)}")
                app_cache_config = CacheConfig(**overrides)
                cache_config_source = "ppl CACHE_CONFIG"
            else:
                app_cache_config = CacheConfig()

            # 兼容 cache_mode 为字符串的写法
            if isinstance(app_cache_config.cache_mode, str):
                try:
                    app_cache_config.cache_mode = CacheMode(app_cache_config.cache_mode)
                except ValueError:
                    logger.warning(
                        f"Invalid cache_mode '{app_cache_config.cache_mode}' in CACHE_CONFIG, "
                        "fallback to default READ_WRITE"
                    )
                    app_cache_config.cache_mode = CacheConfig().cache_mode

            # 命令行传入的 enable_latent_cache 写入配置（调用方已保证为 True 才进入本函数）
            if enable_latent_cache is not None:
                app_cache_config.enable_latent_cache = enable_latent_cache
                cache_config_source = "command line"

            if cache_mode is not None:
                try:
                    app_cache_config.cache_mode = CacheMode(cache_mode)
                    cache_config_source = "command line"
                except ValueError:
                    logger.warning(f"Invalid cache_mode '{cache_mode}', using {app_cache_config.cache_mode}")

            # 尽量提前初始化 cache 日志 sink，覆盖后续 build/latent cache 初始化错误
            if getattr(app_cache_config, "cache_log_enabled", False) and setup_cache_log_sink:
                cache_log_dir = getattr(app_cache_config, "cache_log_dir", None)
                if not cache_log_dir:
                    cache_log_dir = str(Path(app_cache_config.latent_cache_dir) / "logs")
                setup_cache_log_sink(
                    log_dir=cache_log_dir,
                    level=getattr(app_cache_config, "cache_log_level", "DEBUG"),
                    rotation=getattr(app_cache_config, "cache_log_rotation", "100 MB"),
                    retention=getattr(app_cache_config, "cache_log_retention", "7 days"),
                )
                if ppl_cache_config_load_error is not None:
                    logger.warning(
                        "CACHE_CONFIG load failed during cache init, using defaults. Original error: {}",
                        ppl_cache_config_load_error,
                    )

            try:
                build_latent_data_func = import_function_from_file(ppl_file, "build_latent_data")
                logger.info(f"Found build_latent_data function in {ppl_file}")
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"ppl_file must define 'build_latent_data' for cache service. "
                    f"Missing or invalid in {ppl_file}. Error: {e}"
                ) from e

            # 初始化 LatentCache
            latent_cache = LatentCache(
                Path(app_cache_config.latent_cache_dir),
                app_cache_config,
            )

            # 初始化 CacheService
            cache_service = CacheService(
                latent_cache=latent_cache,
                build_latent_data_func=build_latent_data_func,
                cache_mode=app_cache_config.cache_mode,
                app_cache_config=app_cache_config,
            )

            mode_value = getattr(app_cache_config.cache_mode, "value", app_cache_config.cache_mode)
            logger.info(f"Cache service enabled (mode: {mode_value}, source: {cache_config_source})")
            return cache_service
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Failed to initialize cache service: {e}")
            return None
