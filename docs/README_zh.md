# TeleFuser 文档

欢迎使用 TeleFuser 文档。本目录包含使用 TeleFuser 进行开发和使用的完整指南。

## 可用语言

- [English Documentation](./en/)
- [中文文档](./zh/)

## 文档索引

### 用户指南

| 文档 | 描述 |
|------|------|
| [Service 指南](./zh/service.md) | CLI、API 和 SDK 的完整使用指南 |
| [模型加载指南](./zh/model_loading.md) | 使用 ModuleManager 加载模型的指南 |
| [CPU 卸载指南](./zh/offload.md) | 通过 CPU 卸载进行内存优化 |

### 并行推理

| 文档 | 描述 |
|------|------|
| [并行推理指南](./zh/parallel.md) | 分布式并行推理架构和使用方法 |
| [Attention 指南](./zh/attention.md) | 注意力实现和长上下文注意力 |
| [Feature Cache](./zh/feature_cache.md) | 特征缓存加速推理 |

### 开发指南

| 文档 | 描述 |
|------|------|
| [添加新模型](./zh/adding_new_model.md) | 集成新模型的分步指南 |
| [Hash 配置管理](./zh/hash_config_management.md) | 管理模型哈希配置 |
| [torch.compile 兼容性](./zh/torch_compile_compatibility.md) | 编写兼容 torch.compile 的推理代码 |

## 快速链接

### 对于用户

- **快速开始**: 参见 [Service 指南 - 快速开始](./zh/service.md#快速开始)
- **CLI 参考**: 参见 [Service 指南 - CLI 使用](./zh/service.md#cli-命令行工具)
- **API 参考**: 参见 [Service 指南 - HTTP API](./zh/service.md#http-api-参考)

### 对于开发者

- **添加新模型**: 参见 [添加新模型](./zh/adding_new_model.md)
- **模型配置**: 参见 [Hash 配置管理](./zh/hash_config_management.md)

## 语言切换

- English: [Click here](./en/)
- 简体中文: [点击这里](./zh/)
