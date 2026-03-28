# Model Weight Information Viewer Tool

A powerful model weight file viewer tool that supports multiple formats and advanced features.

## Features

- ✅ **Multi-format support**: pt, pth, ckpt, bin, safetensors
- ✅ **Wildcard loading**: Supports `*` and `?` wildcards to match multiple files
- ✅ **Split model support**: Automatically identifies and merges split model files
- ✅ **Weight information display**: shape, dtype, numel, requires_grad, device, etc.
- ✅ **Nested structure building**: Automatically builds hierarchical structure based on prefix relationships
- ✅ **Parameter count calculation**: Automatically calculates total parameters and layer parameters
- ✅ **Similar module merging**: Automatically identifies and merges structurally identical modules
- ✅ **Weight hash calculation**: Computes unique hash values based on weight names and shapes
- ✅ **Data export**: Supports exporting to JSON format
- ✅ **Command-line interface**: Easy-to-use CLI tool

## Installation Dependencies

The project already includes required dependencies, ensure they are installed:
```bash
pip install -r requirements/runtime.txt
```

## Usage

### 1. Command Line Usage

```bash
# View a single file
python tool/viewer/weight_viewer.py model.safetensors

# View split models
python tool/viewer/weight_viewer.py "model_part*.safetensors"

# Limit display depth
python tool/viewer/weight_viewer.py model.pt --max-depth 2

# Show complete structure
python tool/viewer/weight_viewer.py model.pt --show-all

# Disable similar module merging
python tool/viewer/weight_viewer.py model.pt --no-merge

# Show only summary
python tool/viewer/weight_viewer.py model.pt --quiet

# Export to JSON
python tool/viewer/weight_viewer.py model.pt --export weight_info.json
```

### 2. Python API Usage

**Note**: According to the project structure, the import path is `tool.viewer.weight_viewer`

```python
from tool.viewer.weight_viewer import WeightViewer

# Create viewer
viewer = WeightViewer()

# Load weights
weights_info = viewer.load_weights("model.safetensors")

# Get summary
summary = viewer.get_summary()
print(f"Total parameters: {summary['total_parameters_formatted']}")
print(f"Weight hash value: {summary['hash_with_shape']}")

# Print detailed information (with similar module merging enabled)
viewer.print_detailed_info(max_depth=3, merge_similar=True)

# Print complete structure (with module merging disabled)
viewer.print_detailed_info(max_depth=3, show_all=True, merge_similar=False)

# Export to JSON
viewer.export_to_json("weight_info.json")
```

## Output Example

```
================================================================================
Model Weight Information Overview
================================================================================
Total parameters: 1.23B (1,234,567,890)
hash with shape: abc123def456...
Loaded files count: 1
File list: ['model.safetensors']

Data type distribution:
  torch.float32: 1.23B (100.00%)

Detailed weight structure:
transformer
transformer.encoder
transformer.encoder.layers
transformer.encoder.layers.0.self_attn.q_proj.weight | (768, 768)        | torch.float32 |    589.82K x24
...
```

## Supported Model Formats

### PyTorch Format
- `.pt`, `.pth`, `.ckpt`, `.bin`
- Supports state_dict and model objects

### SafeTensors Format
- `.safetensors`
- Safe and efficient serialization format

### Split Models
- Automatically identifies files like `model-00001-of-00002.safetensors`
- Loads and merges in numerical order

## Command Line Arguments

### Available Parameters

- `file_pattern`: Weight file path pattern (supports wildcards)
- `--max-depth`: Maximum nesting depth to display (default: 5)
- `--show-all`: Show complete weight structure (disable depth limit)
- `--no-merge`: Disable merging of structurally identical modules
- `--export`: Export weight information to JSON file
- `--quiet`: Show only summary information

## API Reference

### WeightViewer Class

#### Main Methods

**`load_weights(file_pattern: str) -> Dict[str, Any]`**
- Loads weight files, supports wildcards
- Returns nested structure weight information

**`get_summary() -> Dict[str, Any]`**
- Returns weight information summary
- Includes total parameters, weight hash value, file information, data type distribution, etc.

**`print_detailed_info(max_depth: int = 3, show_all: bool = False, merge_similar: bool = True)`**
- Prints detailed weight structure information
- `merge_similar`: Whether to merge structurally identical modules

**`export_to_json(output_path: str)`**
- Exports weight information to JSON file

#### Properties
- `weights_info`: Nested structure weight information
- `total_params`: Total parameter count
- `file_paths`: List of loaded file paths
- `weight_hash`: Weight hash value

## Advanced Features

### Nested Structure Building
The tool automatically builds hierarchical structure based on weight name prefix relationships:
- `transformer.encoder.layers.0.self_attn.q_proj.weight`
- → `transformer` → `encoder` → `layers` → `0` → `self_attn` → `q_proj` → `weight`

### Automatic Similar Module Merging
The tool automatically identifies and merges structurally identical modules for display:
- For example: `transformer.encoder.layers.0` to `transformer.encoder.layers.23` will be merged and displayed as `transformer.encoder.layers.0 x24`
- This feature can be disabled using the `--no-merge` parameter

### Weight Hash Calculation
Computes unique hash values based on weight names and shapes, used for model version identification and comparison.

### Parameter Count Statistics
- Automatically calculates the number of elements for each weight
- Summarizes total parameter count
- Formats display (K/M/B)

### Data Type Analysis
- Statistics on parameter distribution across different data types
- Displays percentage distribution

## Troubleshooting

### Common Issues

**Q: File not found error**
A: Check if the file path is correct and ensure read permissions

**Q: Unsupported file format**
A: Currently supports pt/pth/ckpt/bin/safetensors formats

**Q: Insufficient memory**
A: For large models, try using `--quiet` mode or limit display depth

**Q: Split model loading order error**
A: Ensure filenames follow standard naming conventions (e.g., model-00001-of-00002.safetensors)

**Q: Inaccurate similar module merging display**
A: Use `--no-merge` parameter to disable merging function and view complete structure

**Q: Inconsistent weight hash values**
A: Hash values are calculated based on weight names and shapes, ensure compared models have the same weight structure