# 日志系统

TeleFuser 提供了一个基于 [loguru](https://github.com/Delgan/loguru) 的工业级日志系统，专为生产部署场景设计。

## 特性

- **结构化 JSON 日志** - 机器可解析的输出，便于日志聚合系统处理
- **日志轮转** - 按大小/时间自动轮转，支持压缩
- **多输出目标** - 同时输出到控制台、文件、Syslog 或组合
- **环境变量配置** - 通过环境变量快速配置
- **模块级过滤** - 不同模块可设置不同日志级别
- **异步日志** - 高性能场景下非阻塞日志
- **上下文绑定** - 附加 request_id、trace_id 等字段到日志
- **分布式追踪** - 内置 trace_id/span_id 支持，适用于微服务架构
- **多进程支持** - PID 和 Rank 信息动态注入
- **Syslog 支持** - RFC 5424 兼容的 UDP 日志发送

## 快速开始

### 基础用法

将 `from loguru import logger` 替换为：

```python
from telefuser.utils.logging import logger

logger.info("应用启动")
logger.debug("调试信息: {}", some_value)
logger.error("发生错误")
```

### 通过环境变量配置

```bash
# 设置日志级别
export TELEFUSER_LOG_LEVEL=INFO

# 使用 JSON 格式进行结构化日志记录
export TELEFUSER_LOG_FORMAT=json

# 记录到文件并启用轮转
export TELEFUSER_LOG_FILE=/var/log/telefuser/app.log
export TELEFUSER_LOG_ROTATION="1 day"
export TELEFUSER_LOG_RETENTION="30 days"

# 同时输出到控制台和文件
export TELEFUSER_LOG_SINK=both

# 多进程模式
export TELEFUSER_LOG_MULTIPROCESS=per_rank

# Syslog 配置
export TELEFUSER_LOG_SINK=syslog
export TELEFUSER_LOG_SYSLOG_HOST=syslog.example.com
export TELEFUSER_LOG_SYSLOG_PORT=514
```

### 通过代码配置

```python
from telefuser.utils.logging import configure_logging, LogConfig, LogFormat, LogSink

# 简单配置
configure_logging(LogConfig(level="DEBUG"))

# 高级配置
configure_logging(LogConfig(
    level="INFO",
    format=LogFormat.JSON,
    sink=LogSink.BOTH,
    log_file="/var/log/telefuser/app.log",
    rotation="100 MB",
    retention="30 days",
    compression="gz",
))

# Syslog 配置
configure_logging(LogConfig(
    sink=LogSink.SYSLOG,
    syslog_host="syslog.example.com",
    syslog_port=514,
    syslog_facility="user",
))
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `TELEFUSER_LOG_LEVEL` | 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `TELEFUSER_LOG_FORMAT` | 输出格式 (simple, detailed, json, structured) | detailed |
| `TELEFUSER_LOG_SINK` | 输出目标 (console, file, syslog, both) | console |
| `TELEFUSER_LOG_FILE` | 日志文件路径 | None |
| `TELEFUSER_LOG_ROTATION` | 轮转条件 (如 "100 MB", "1 day") | 100 MB |
| `TELEFUSER_LOG_RETENTION` | 保留期限 (如 "30 days", "10 files") | 30 days |
| `TELEFUSER_LOG_COMPRESSION` | 压缩格式 (gz, bz2, xz) | gz |
| `TELEFUSER_LOG_ENQUEUE` | 使用异步日志 | true |
| `TELEFUSER_LOG_BACKTRACE` | 在错误中包含堆栈跟踪 | true |
| `TELEFUSER_LOG_DIAGNOSE` | 包含诊断信息 | true |
| `TELEFUSER_LOG_SERIALIZE` | 序列化为 JSON | false |
| `TELEFUSER_LOG_MULTIPROCESS` | 多进程模式 (single, per_process, per_rank) | single |
| `TELEFUSER_LOG_SYSLOG_HOST` | Syslog 服务器主机名 | localhost |
| `TELEFUSER_LOG_SYSLOG_PORT` | Syslog 服务器端口 | 514 |
| `TELEFUSER_LOG_SYSLOG_FACILITY` | Syslog 设施 (user, daemon, local0) | user |
| `TELEFUSER_LOG_EXTRA_FIELDS` | JSON 字符串，额外字段 | {} |
| `TELEFUSER_LOG_FILTER_MODULES` | JSON 字符串，模块过滤 | {} |

## 高级功能

### 分布式追踪

使用 `TracingContext` 添加 trace_id 和 span_id 用于分布式追踪：

```python
from telefuser.utils.logging import logger, TracingContext, get_trace_id, set_trace_id

# 自动生成 trace_id 和 span_id
with TracingContext():
    logger.info("处理请求")  # 包含自动生成的 trace_id
    trace_id = get_trace_id()  # 获取当前 trace_id

# 使用指定的 trace_id（如从请求头获取）
with TracingContext(trace_id="abc123", span_id="def456"):
    logger.info("使用自定义 trace")  # 使用提供的 trace_id

# 手动设置 trace_id
set_trace_id("xyz789")
logger.info("带有 trace_id 的消息")
```

trace_id 和 span_id 会自动包含在：
- JSON 输出格式
- Structured 格式
- 所有日志记录的 `extra` 字段中

### 上下文绑定

在作用域内将上下文信息附加到所有日志：

```python
from telefuser.utils.logging import logger, LoggingContext

with LoggingContext(request_id="req-123", user_id="user-456"):
    logger.info("处理请求")  # 日志包含 request_id 和 user_id
    process_data()
    logger.info("请求完成")   # 仍然包含上下文
```

### 性能监控

支持同步和异步函数：

```python
from telefuser.utils.logging import log_execution_time

@log_execution_time
def heavy_computation():
    # 这将在 DEBUG 级别记录执行时间
    pass

@log_execution_time
async def async_computation():
    # 也支持异步函数
    pass
```

### 异常日志

支持同步和异步函数：

```python
from telefuser.utils.logging import log_exceptions

@log_exceptions
def risky_operation():
    # 异常会自动记录完整上下文
    raise ValueError("出错了")

@log_exceptions
async def async_risky_operation():
    # 也支持异步函数
    raise ValueError("异步错误")
```

### 模块级过滤

```python
from telefuser.utils.logging import configure_logging, LogConfig

configure_logging(LogConfig(
    level="INFO",
    filter_modules={
        "telefuser.models": "WARNING",  # models 模块只记录 WARNING 及以上
        "telefuser.distributed": "DEBUG",  # distributed 模块记录 DEBUG 及以上
    }
))
```

## 输出格式

### Simple（简单）
```
2024-01-15 10:30:00 | INFO     | 应用启动
```

### Detailed（详细，默认）
```
2024-01-15 10:30:00.123 | INFO     | [pid:12345 rank:0] mymodule:myfunc:42 | 应用启动
```

`pid` 和 `rank` 在日志记录时动态注入，即使进程环境发生变化也能保证正确的值。

### JSON
```json
{
  "text": "应用启动\n",
  "record": {
    "time": {"repr": "2024-01-15 10:30:00.123456"},
    "level": {"name": "INFO"},
    "extra": {
      "pid": "12345",
      "rank": "0",
      "trace_id": "abc123def456",
      "span_id": "789ghi"
    }
  }
}
```

### Structured（结构化）
```
time=2024-01-15 10:30:00.123 level=INFO pid=12345 rank=0 trace_id=abc123 span_id=789ghi module=mymodule func=myfunc line=42 msg=应用启动
```

## 生产部署

### Docker 示例

```dockerfile
ENV TELEFUSER_LOG_LEVEL=INFO
ENV TELEFUSER_LOG_FORMAT=json
ENV TELEFUSER_LOG_FILE=/var/log/telefuser/app.log
ENV TELEFUSER_LOG_ROTATION="1 day"
ENV TELEFUSER_LOG_RETENTION="7 days"
```

### Kubernetes 示例

```yaml
env:
  - name: TELEFUSER_LOG_LEVEL
    value: "INFO"
  - name: TELEFUSER_LOG_FORMAT
    value: "json"
  - name: TELEFUSER_LOG_FILE
    value: "/var/log/telefuser/app.log"
volumeMounts:
  - name: logs
    mountPath: /var/log/telefuser
```

### Syslog 集成

用于集中式日志管理：

```python
from telefuser.utils.logging import configure_logging, LogConfig, LogSink

configure_logging(LogConfig(
    sink=LogSink.SYSLOG,
    syslog_host="logs.example.com",
    syslog_port=514,
    syslog_facility="user",
    level="INFO",
))
```

消息以 RFC 5424 格式通过 UDP 发送。

## 多进程日志

支持三种多进程日志模式：

### 1. single（默认）- 统一日志文件
所有进程写入同一个日志文件，通过 `enqueue=True` 保证线程/进程安全。

```python
from telefuser.utils.logging import configure_logging, LogConfig

configure_logging(LogConfig(
    log_file="/var/log/telefuser/app.log",
    multiprocess="single",  # 默认
))
```

### 2. per_process - 每个进程独立文件
每个进程写入独立的日志文件，文件名包含进程 ID。

```python
configure_logging(LogConfig(
    log_file="/var/log/telefuser/app.log",
    multiprocess="per_process",
))
# 生成: app_12345.log, app_12346.log 等
```

### 3. per_rank - 按分布式 rank 分文件
适用于分布式训练场景，按 rank 号分文件。

```python
configure_logging(LogConfig(
    log_file="/var/log/telefuser/app.log",
    multiprocess="per_rank",
))
# 生成: app_rank0.log, app_rank1.log 等
```

### 手动指定占位符
也可以在路径中手动指定占位符：

```python
configure_logging(LogConfig(
    log_file="/var/log/telefuser/app_{pid}_{rank}.log",
))
# 生成: app_12345_0.log 等
```

支持的占位符：
- `{pid}` - 进程 ID（日志记录时动态解析）
- `{rank}` - 分布式训练 rank（从 RANK/LOCAL_RANK 等环境变量读取）
- `{time}` - 时间戳

## 启动信息

当日志系统初始化时，会自动打印配置信息到 stderr：

```
╔══════════════════════════════════════════════════════════════════╗
║              TeleFuser Logging System Initialized               ║
╠══════════════════════════════════════════════════════════════════╣
║  Log Level:      INFO                                             ║
║  Format:         DETAILED                                         ║
║  Sink:           BOTH                                             ║
║  Log File:       /var/log/telefuser/app.log                       ║
║  Rotation:       1 day                                            ║
║  Retention:      30 days                                          ║
║  Compression:    gz                                               ║
║  Async Mode:     True                                             ║
║  Backtrace:      True                                             ║
║  Colorize:       True                                             ║
║  Multiprocess:   per_rank                                         ║
║  PID:            12345                                            ║
║  Rank:           0                                                ║
╚══════════════════════════════════════════════════════════════════╝
```

这样可以在应用启动时立即确认日志配置是否正确，以及日志文件保存位置。

## 从 loguru 迁移

日志系统是 loguru 的直接替代品：

```python
# 之前
from loguru import logger

# 之后
from telefuser.utils.logging import logger

# 其他代码完全不变！
logger.info("用法完全一样")
```

## 最佳实践

### 1. 容器化部署

在容器环境中，推荐使用 JSON 格式输出到 stdout，由外部日志收集器处理：

```bash
export TELEFUSER_LOG_FORMAT=json
export TELEFUSER_LOG_SINK=console
```

### 2. 本地开发

开发时使用详细格式便于调试：

```bash
export TELEFUSER_LOG_LEVEL=DEBUG
export TELEFUSER_LOG_FORMAT=detailed
```

### 3. 生产环境文件日志

生产环境同时输出到控制台和文件：

```bash
export TELEFUSER_LOG_SINK=both
export TELEFUSER_LOG_FILE=/var/log/telefuser/app.log
export TELEFUSER_LOG_ROTATION="1 day"
export TELEFUSER_LOG_RETENTION="30 days"
```

### 4. 微服务分布式追踪

在微服务间传递 trace_id：

```python
from telefuser.utils.logging import TracingContext, set_trace_id

# 在 API 处理器中，从请求头提取 trace_id
@app.route("/api/process")
def process_request():
    trace_id = request.headers.get("X-Trace-ID")
    with TracingContext(trace_id=trace_id):
        # 此请求中的所有日志共享同一个 trace_id
        logger.info("处理请求")
        result = process_data()
        logger.info("请求完成")
    return result
```

## 完整示例

以下是演示所有日志功能的完整可运行示例：

```python
from __future__ import annotations

import time

from telefuser.utils.logging import (
    LogConfig,
    LogFormat,
    LogSink,
    LoggingContext,
    configure_logging,
    log_exceptions,
    log_execution_time,
    logger,
)


def demo_basic_logging():
    """演示基本日志用法。"""
    logger.debug("这是一条调试消息")
    logger.info("这是一条信息消息")
    logger.warning("这是一条警告消息")
    logger.error("这是一条错误消息")


def demo_context_binding():
    """演示请求追踪的上下文绑定。"""
    # 模拟请求上下文
    with LoggingContext(request_id="req-12345", user_id="user-789"):
        logger.info("处理请求")
        logger.debug("请求详情: method=POST, path=/api/v1/generate")

        # 嵌套上下文
        with LoggingContext(stage="inference"):
            logger.info("运行推理")
            time.sleep(0.1)
            logger.info("推理完成")

        logger.info("请求完成")

    # 上下文外部 - 无 request_id
    logger.info("此日志没有请求上下文")


def demo_performance_monitoring():
    """演示性能监控装饰器。"""
    @log_execution_time
    def heavy_computation():
        """模拟繁重计算。"""
        time.sleep(0.5)
        return sum(range(10000))

    # 配置 DEBUG 级别以查看计时日志
    configure_logging(LogConfig(level="DEBUG"))

    result = heavy_computation()
    logger.info(f"计算结果: {result}")


def demo_exception_logging():
    """演示异常日志。"""
    @log_exceptions
    def risky_operation():
        """可能抛出异常的函数。"""
        raise ValueError("出错了!")

    try:
        risky_operation()
    except ValueError:
        logger.info("异常已被捕获并记录")


def demo_json_format():
    """演示 JSON 结构化日志。"""
    configure_logging(
        LogConfig(
            level="INFO",
            format=LogFormat.JSON,
            serialize=True,
        )
    )

    logger.info(
        "应用事件",
        extra={
            "event_type": "user_action",
            "action": "generate_image",
            "model": "wan21",
        },
    )


def demo_production_config():
    """演示生产环境配置。"""
    config = LogConfig(
        level="INFO",
        format=LogFormat.JSON,
        sink=LogSink.BOTH,
        log_file="/var/log/telefuser/app.log",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        enqueue=True,  # 异步日志提升性能
        backtrace=True,
        diagnose=True,
    )

    print("生产配置:")
    print(f"  级别: {config.level}")
    print(f"  格式: {config.format.name}")
    print(f"  输出: {config.sink.name}")
    print(f"  文件: {config.log_file}")


if __name__ == "__main__":
    print("TeleFuser 日志系统演示")
    print("=" * 50)

    demo_basic_logging()
    demo_context_binding()
    demo_performance_monitoring()
    demo_exception_logging()
    demo_json_format()
    demo_production_config()

    print("\n" + "=" * 50)
    print("演示完成!")
```

使用不同配置运行：

```bash
TELEFUSER_LOG_LEVEL=DEBUG python your_script.py
TELEFUSER_LOG_FORMAT=json python your_script.py
```
