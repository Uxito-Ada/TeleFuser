# TeleFuser Documentation

Welcome to the TeleFuser documentation. This directory contains comprehensive guides for using and developing with TeleFuser.

## Available Languages

- [English Documentation](./en/)
- [中文文档](./zh/)

## Documentation Index

### User Guides

| Document | Description |
|----------|-------------|
| [Service Guide](./en/service.md) | Complete guide for CLI, API, and SDK usage |
| [Service Metadata Guide](./en/service_metadata.md) | Consume `/v1/service/metadata` for dynamic forms, routing, and gateways |
| [Adding New Example](./en/adding_new_example.md) | Build runnable examples and define server-facing pipeline contracts |
| [Model Loading](./en/model_loading.md) | Guide for loading models with ModuleManager |
| [CPU Offloading](./en/offload.md) | Memory optimization via CPU offloading |

### Parallel Inference

| Document | Description |
|----------|-------------|
| [Parallel Inference Guide](./en/parallel.md) | Distributed parallel inference architecture and usage |
| [Attention Guide](./en/attention.md) | Attention implementation and long-context attention |
| [Feature Cache](./en/feature_cache.md) | Feature caching for inference acceleration |

### Developer Guides

| Document | Description |
|----------|-------------|
| [Adding New Model](./en/adding_new_model.md) | Step-by-step guide for integrating new models |
| [Hash Config Management](./en/hash_config_management.md) | Managing model hash configurations |
| [torch.compile Compatibility](./en/torch_compile_compatibility.md) | Writing code compatible with torch.compile for inference |

## Quick Links

### For Users

- **Getting Started**: See [Service Guide - Quick Start](./en/service.md#quick-start)
- **CLI Reference**: See [Service Guide - CLI Usage](./en/service.md#cli-usage)
- **API Reference**: See [Service Guide - HTTP API](./en/service.md#http-api-reference)
- **Service Contract**: See [Service Guide - Pipeline Contract and Parameter Definitions](./en/service.md#pipeline-contract-and-parameter-definitions)
- **Metadata Consumption**: See [Service Metadata Guide](./en/service_metadata.md)

### For Developers

- **Add a New Model**: See [Adding New Model](./en/adding_new_model.md)
- **Add a New Example**: See [Adding New Example](./en/adding_new_example.md)
- **Model Configuration**: See [Hash Config Management](./en/hash_config_management.md)

## Language Switch

- English: [Click here](./en/)
- 简体中文: [点击这里](./zh/)
