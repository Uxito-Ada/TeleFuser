# Service Metadata 消费指南

本文档说明前端、网关和自动化层应该如何消费 `GET /v1/service/metadata`。

## 为什么需要这个端点

TeleFuser 允许每个 pipeline example 通过 pipeline contract 声明自己的服务能力。
`/v1/service/metadata` 就是服务端加载完 pipeline 之后，对这份 contract 的运行时视图。

消费方应把这个端点视为以下信息的事实来源：

- 当前运行中的 pipeline 支持哪些 task
- 支持哪些 media type
- 每个 task 需要哪些文件输入
- 每个 task 对外暴露哪些用户参数
- 动态生成表单时应该展示哪些默认值

不要只根据 example 文件名去硬编码任务能力。

## 返回结构

一个典型响应大致如下：

```json
{
  "pipeline_file": "./examples/wan_video/wan22_14b_image_to_video_distill_h100.py",
  "parallelism": 1,
  "task": "i2v",
  "security_level": "STRICT",
  "runner": "PipelineRunner",
  "declared_pipeline_contract": true,
  "contract_version": "v1",
  "pipeline_name": "wan22_A14B_i2v_h100_distill",
  "supported_tasks": ["i2v", "fl2v"],
  "supported_media_types": ["video"],
  "execution_mode": "serial_single_pipeline",
  "effective_max_concurrent_tasks": 1,
  "entrypoints": {
    "get_pipeline": "get_pipeline",
    "run_with_file": "run_with_file"
  },
  "task_contracts": {
    "i2v": {
      "media_type": "video",
      "required_inputs": ["first_image_path"],
      "optional_inputs": ["last_image_path"],
      "parameters": {
        "prompt": {
          "type": "string",
          "required": true,
          "default": "",
          "description": "正向提示词。",
          "enum": [],
          "exposed": true
        },
        "resolution": {
          "type": "string",
          "required": false,
          "default": "720p",
          "description": "对用户暴露的输出分辨率。",
          "enum": ["480p", "720p"],
          "exposed": true
        }
      }
    }
  },
  "service_effective_max_concurrent_tasks": 1,
  "service_configured_max_concurrent_tasks": 4,
  "max_queue_size": 32
}
```

## 关键顶层字段

| 字段 | 含义 |
|------|------|
| `declared_pipeline_contract` | 为 `true` 表示 pipeline 显式提供了 contract；为 `false` 表示服务端使用的是 legacy fallback。 |
| `supported_tasks` | 当前运行中的 pipeline 实际支持的 task 列表。 |
| `supported_media_types` | 高层输出媒体类型，通常是 `video` 和/或 `image`。 |
| `task_contracts` | 每个 task 的输入和参数元数据，可用于 UI 生成和请求校验。 |
| `effective_max_concurrent_tasks` | contract 声明的 pipeline 有效并发。对于当前单 pipeline 运行时，一般是 `1`。 |
| `service_effective_max_concurrent_tasks` | service 层的实际有效并发。在当前模型里通常也为 `1`。 |
| `service_configured_max_concurrent_tasks` | 用户配置值，在运行时收敛前的数字。更适合观测，不适合拿来做乐观并发控制。 |
| `max_queue_size` | 排队上限，适合用于仪表盘展示和背压策略。 |

## 如何使用 `task_contracts`

每个 task contract 都由四部分组成：

- `media_type`：该 task 的输出类别
- `required_inputs`：必须提供的文件类输入
- `optional_inputs`：可选文件类输入
- `parameters`：对外暴露的用户参数

只有用户可见参数才会出现在 `parameters` 中。内部 pipeline 设置会被有意过滤掉。

### 动态表单生成

客户端可以按以下流程生成表单：

1. 读取 `supported_tasks`。
2. 让用户选择 task，或者根据上传内容推断 task。
3. 读取 `task_contracts[task]`。
4. 根据 `required_inputs` 和 `optional_inputs` 渲染上传控件。
5. 根据 `parameters` 渲染参数控件。
6. 用 `default` 作为表单初始值。
7. 当存在 `enum` 时，优先渲染下拉选择。
8. 用 `required` 在发请求前做前端校验。

### 任务推断

对于上传驱动的交互，可以用 contract 帮助推断最可能的 task：

- 没有文件输入：优先考虑 `t2v`、`t2i` 这类纯文本任务
- `first_image_path`：优先考虑 `i2v` 或 `i2i`
- `first_image_path` + `last_image_path`：优先考虑 `fl2v`
- `ref_video_path`：优先考虑 `vc` 或 `vsr`

最终 task 仍由服务端校验。前端推断只是提升体验，不是权限来源。

### 参数语义

服务端会先应用 task contract 默认值，再校验必填参数。因此：

- contract 默认值应当直接作为 UI 默认值展示
- contract 标记为必填的字段应在 UI 中视为必填
- 通用请求模型里的默认值不应当替代 task-specific 的用户体验来源

## 网关路由策略

如果你在实现一个决定该走 TeleFuser 原生路由还是 OpenAI 兼容路由的网关，可以这样使用 metadata：

1. 用 `supported_tasks` 和 `task_contracts` 判断当前 pipeline 真实支持什么。
2. 用 `media_type` 判断请求属于 image 还是 video 流程。
3. 用 `required_inputs` 判断请求是纯文本、图像条件还是视频条件。
4. 在转发前先拒绝不支持的 task 组合。

示例：

- `media_type=image` 且没有必需输入：适合 `/v1/images/generations`
- `media_type=image` 且包含 `first_image_path`：适合 `/v1/images/edits`
- `media_type=video` 且没有必需输入：文生视频流程
- `media_type=video` 且包含 `first_image_path`：图生视频流程
- `media_type=video` 且包含 `ref_video_path`：视频条件流程，如续写或超分

## Legacy Pipeline

当 `declared_pipeline_contract` 为 `false` 时，服务端会根据 CLI task 合成一份兼容契约。

在这种模式下：

- `supported_tasks` 可能比现代 manifest 模式更窄
- `task_contracts` 可能只有默认输入要求
- `parameters` 可能为空

客户端仍然可以工作，但应该预期拿到的元数据更少。

## 推荐的客户端行为

- 按服务实例缓存 metadata，但在启动或切换 pipeline 时刷新。
- 不要假设每台服务都支持同一批 task。
- 优先使用 `task_contracts`，而不是手写一套长期维护的前端 schema。
- 对背压敏感的 UI 可以结合 `max_queue_size` 和队列相关端点。
- 把 `/v1/service/metadata` 当作描述性元数据，而不是替代服务端校验。

## 相关文档

- [Service 指南](./service.md)
- [添加新 Example](./adding_new_example.md)