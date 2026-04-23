# Service Metadata Consumption Guide

This guide explains how frontend applications, gateways, and automation layers should consume
`GET /v1/service/metadata`.

## Why This Endpoint Exists

TeleFuser allows each pipeline example to declare its service-facing capabilities through a pipeline contract.
`/v1/service/metadata` is the runtime view of that contract after the server loads the pipeline.

Consumers should treat this endpoint as the source of truth for:

- which tasks are supported by the running pipeline
- which media types are supported
- which file inputs are required for each task
- which user-facing parameters are available for each task
- which defaults should be shown in a generated form

Do not hardcode task support from the example filename alone.

## Response Shape

A typical response looks like this:

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
          "description": "Positive guidance text prompt.",
          "enum": [],
          "exposed": true
        },
        "resolution": {
          "type": "string",
          "required": false,
          "default": "720p",
          "description": "User-facing output resolution.",
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

## Important Top-Level Fields

| Field | Meaning |
|------|---------|
| `declared_pipeline_contract` | `true` when the pipeline explicitly provided a contract; `false` for legacy fallback behavior. |
| `supported_tasks` | Tasks that this running pipeline can serve right now. |
| `supported_media_types` | High-level output media categories, usually `video` and/or `image`. |
| `task_contracts` | Per-task input and parameter metadata for UI generation and validation. |
| `effective_max_concurrent_tasks` | Pipeline-level effective concurrency declared by the contract. For the current single-pipeline runtime this is usually `1`. |
| `service_effective_max_concurrent_tasks` | Effective runtime concurrency of the service layer. Also usually `1` in the current model. |
| `service_configured_max_concurrent_tasks` | User-configured value before runtime clamping. Useful for observability, not for optimistic client-side concurrency. |
| `max_queue_size` | Admission-control limit for queued tasks. Useful for dashboards and backpressure strategies. |

## How To Use `task_contracts`

Each task contract has four parts:

- `media_type`: expected output category for the task
- `required_inputs`: file-like inputs that must be present
- `optional_inputs`: additional file-like inputs the task can consume
- `parameters`: exposed user-facing request fields

Only exposed user-facing parameters appear in `parameters`. Internal pipeline settings are intentionally filtered out.

### Form Generation

A client can generate a task form using this process:

1. Read `supported_tasks`.
2. Let the user choose one task, or infer one from uploaded inputs.
3. Read `task_contracts[task]`.
4. Render file-upload controls from `required_inputs` and `optional_inputs`.
5. Render parameter controls from `parameters`.
6. Use `default` values as initial form values.
7. Use `enum` to render a select component when present.
8. Use `required` to block submission before making the request.

### Task Inference

For upload-driven UX, the contract can help infer the most likely task:

- no file inputs: prefer text-only tasks such as `t2v` or `t2i`
- `first_image_path`: prefer `i2v` or `i2i`
- `first_image_path` + `last_image_path`: prefer `fl2v`
- `ref_video_path`: prefer `vc` or `vsr`

The server still validates the final task choice. Client-side inference is only a UX optimization.

### Parameter Semantics

The server applies task-contract defaults before validating required parameters. That means:

- contract defaults should be shown as the UI defaults
- contract-required fields should be treated as required in the UI
- generic API-model defaults are not the best source for task-specific UX

## Routing Strategy For Gateways

If you are building a gateway that decides whether to call native TeleFuser routes or OpenAI-compatible routes,
use the metadata like this:

1. Use `supported_tasks` and `task_contracts` to know what the current pipeline actually supports.
2. Use `media_type` to decide whether the task belongs to image or video flows.
3. Use `required_inputs` to decide whether a request is text-only, image-conditioned, or video-conditioned.
4. Reject unsupported task combinations before forwarding the request.

Examples:

- `media_type=image` with no required inputs: good fit for `/v1/images/generations`
- `media_type=image` with `first_image_path`: good fit for `/v1/images/edits`
- `media_type=video` with no required inputs: text-to-video flow
- `media_type=video` with `first_image_path`: image-to-video flow
- `media_type=video` with `ref_video_path`: video-conditioned flow such as continue or super-resolution

## Legacy Pipelines

When `declared_pipeline_contract` is `false`, the server synthesizes a compatibility contract from the CLI task.

In that mode:

- `supported_tasks` may be narrower than a modern manifest-based pipeline
- `task_contracts` may only contain default input requirements
- `parameters` may be empty

Clients should still work, but should expect less rich metadata.

## Recommended Client Behavior

- Cache metadata per server instance, but refresh on startup or pipeline switch.
- Do not assume every task exists on every server.
- Prefer `task_contracts` over hand-maintained UI schemas.
- Use `max_queue_size` and queue endpoints for backpressure-aware UX.
- Treat `/v1/service/metadata` as descriptive metadata, not as a replacement for server-side validation.

## Related Documents

- [Service Guide](./service.md)
- [Adding New Example](./adding_new_example.md)