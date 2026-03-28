---
name: add-new-pipeline
description: Guide for integrating external project pipelines into TeleFuser. Six-phase workflow with interactive checkpoints.
---

# Add New Pipeline Integration Guide

## Trigger Conditions

- User requests to integrate a new model/pipeline from external project
- User mentions "integrate xxx into telefuser"

---

## Workflow Overview

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
 Analyze    Refactor   Integrate   Cleanup    Review
    ↓          ↓          ↓          ↓          ↓
Checkpoint  Checkpoint  Checkpoint  Checkpoint  Done
```

**Each phase ends with AskUserQuestion checkpoint - wait for approval before proceeding.**

---

## Phase 1: Analyze Original Pipeline

### Goals
1. Understand model architecture by reading source code
2. Document pipeline logic and inference flow
3. Create analysis reports

### Key Tasks

1. **Read pipeline entry point** - Trace `__call__` method execution flow
2. **Read model definitions** - Go deep to actual class implementations (DiT, VAE, Text Encoder)
3. **Create analysis reports** in `examples/<model_name>/analysis/`:
   - `PIPELINE_LOGIC.md` - Entry point, execution steps, key functions
   - `MODEL_DEFINITION.md` - Architecture, configuration, class hierarchy
   - `INFERENCE_LOGIC.md` - Forward flow, data transformations

### Progress Tracking

Create `examples/<model_name>/PROGRESS.md`:

```markdown
# [Model Name] Integration Progress

## Overview
- **Model**: [Name]
- **Type**: [T2V/I2V/T2I/SR]
- **Started**: [Date]

## Phase Status
| Phase | Status | Notes |
|-------|--------|-------|
| 1. Analyze | 🔄 In Progress | |
| 2. Refactor | ⏳ Pending | |
| 3. Integrate | ⏳ Pending | |
| 4. Cleanup | ⏳ Pending | |
| 5. Review | ⏳ Pending | |

## Key Findings
- Architecture patterns: ...
- Special handling required: ...
- Implementation challenges: ...
```

### Model Source Rules

| Component | Integration Method |
|-----------|-------------------|
| **DiT/Transformer** | Source-level (`telefuser/models/<model>_dit.py`, inherit `BaseModel`) |
| VAE | `module_manager.load_from_huggingface()` |
| Text Encoder | `module_manager.load_from_huggingface()` |
| Scheduler | Use existing or HuggingFace |

**CRITICAL for DiT implementation:**
- Use ONLY standard PyTorch ops (no TeleFuser optimizations)
- Goal: Output matches original exactly
- Optimizations come later via `/optimize-pipeline`

### 🛑 Phase 1 Checkpoint

After completion:
1. Show analysis report summaries
2. Highlight critical findings (unique patterns, challenges)
3. **AskUserQuestion**: "Phase 1 complete. Ready for Phase 2?"

---

## Phase 2: Refactor to Stage/Pipeline

### Goals
1. Create Pipeline class (inherit `BasePipeline`)
2. Create Stage classes (inherit `BaseStage`)
3. Verify output matches Phase 1

### Files to Create

```
telefuser/pipelines/<model_name>/
├── __init__.py
├── pipeline.py          # Pipeline class
├── dit_denoising.py     # Denoising stage
├── vae.py               # VAE stage
└── text_encoding.py     # Text encoding stage
```

### Example File

Create `examples/<model_name>/<model>_<task>_<hardware>.py`:

```python
PPL_CONFIG = dict(
    name="model_task_hardware",
    negative_prompt="...",
    num_inference_steps=50,
    cfg_scale=4.0,
)

def get_pipeline(parallelism=1, model_root="..."):
    ...

def run(pipeline, prompt, ...):
    ...

@click.command()
@click.option("--gpu_num", default=1)
def main(gpu_num, prompt, ...):
    ...
```

### Key Patterns

- Use `@with_model_offload(["model_name"])` for CPU offloading
- Use `@torch.inference_mode()` in stage process methods
- Use `auto_async_call()` for overlapping stages

**Reference:** `telefuser/pipelines/wan_video/`, `telefuser/pipelines/z_image/`

### 🛑 Phase 2 Checkpoint

After completion:
1. Show pipeline.py and stage files
2. Highlight key implementation decisions
3. **AskUserQuestion**: "Phase 2 complete. Ready for Phase 3?"

---

## Phase 3: Integrate Internal Models

### Goals
1. Implement DiT model at source-level (inherit `BaseModel`)
2. Implement `state_dict_converter()` for loading pretrained weights
3. Verify model loads correctly

### DiT Model Requirements

```python
# telefuser/models/<model>_dit.py
from telefuser.core.base_model import BaseModel

class MyModelDiT(BaseModel):
    def __init__(self, config: MyModelDiTConfig):
        super().__init__()
        # Implement all sub-modules
        self.x_embedder = nn.Linear(...)
        self.transformer_blocks = nn.ModuleList([...])

    def forward(self, hidden_states, timestep, encoder_hidden_states, ...):
        ...

    def get_fsdp_module_names(self) -> list[str]:
        return ["TransformerBlock", "SingleTransformerBlock"]

    @staticmethod
    def state_dict_converter():
        return MyModelDiTStateDictConverter()


class MyModelDiTStateDictConverter:
    def from_diffusers(self, state_dict: dict) -> dict:
        return state_dict  # or key remapping

    def from_official(self, state_dict: dict) -> dict:
        # Key remapping from official/BFL format
        ...
```

**IMPORTANT:** Use only standard PyTorch ops. No TeleFuser optimizations.

### Loading in Pipeline

```python
# DiT - source-level
transformer = Flux2DiT.from_pretrained(transformer_path, torch_dtype)
mm.add_module(transformer, "transformer")

# VAE/TextEncoder - HuggingFace loading
mm.load_from_huggingface(vae_path, module_source="diffusers",
                         module_class=AutoencoderKLFlux2, module_name="vae")
mm.load_from_huggingface(text_encoder_path, module_source="transformers",
                         module_class=Qwen3ForCausalLM, module_name="text_encoder")
```

### 🛑 Phase 3 Checkpoint

After completion:
1. Show `<model>_dit.py` implementation
2. Show state_dict_converter
3. **AskUserQuestion**: "Phase 3 complete. Ready for Phase 4?"

---

## Phase 4: Code Cleanup

### Remove
- `gradient_checkpointing` attributes
- `self.training` conditionals
- Duplicate definitions (RMSNorm, swish, etc.)
- Unused code

### Standardize
- Consistent `from_pretrained` parameter names
- Encoders return dataclass (not dict)
- Shared utilities in single location

### 🛑 Phase 4 Checkpoint

After completion:
1. Run `pre-commit run --all-files`
2. Show cleanup summary
3. Update PROGRESS.md status
4. **AskUserQuestion**: "Phase 4 complete. Ready for Phase 5?"

---

## Phase 5: Review & Compare

### Goals
1. Compare pipeline logic with original
2. Verify edge case handling
3. Ensure numerical output matches

### Comparison Checklist

| Aspect | Check |
|--------|-------|
| Pipeline flow | Steps match original? |
| Edge cases | CFG=1, batch>1, custom sizes? |
| Model config | Parameters match? |
| Numerical | Output matches original? |

Create `examples/<model_name>/COMPARISON_REPORT.md` with findings.

### 🛑 Phase 5 Checkpoint

After completion:
1. Show comparison summary
2. Highlight any mismatches
3. If issues: provide fix suggestions
4. **AskUserQuestion**: "Integration complete. What next?"

---

## Skip Handling

When user says "skip X":
- Skip the work, NOT the checkpoint
- Still analyze from code if skipping execution
- Still use AskUserQuestion for approval

---

## Context Management

Phase 3-4 are critical and need precision. If context is exhausted after Phases 1-2, recommend starting a fresh session before Phase 3.

---

## Related Documentation

| Topic | Document |
|-------|----------|
| Model Implementation | [docs/en/adding_new_model.md](../../docs/en/adding_new_model.md) |
| Stage Implementation | [docs/en/adding_new_stage.md](../../docs/en/adding_new_stage.md) |
| Attention Config | [docs/en/attention.md](../../docs/en/attention.md) |
| Parallel Inference | [docs/en/parallel.md](../../docs/en/parallel.md) |
| Optimization | Use `/optimize-pipeline` skill after integration |