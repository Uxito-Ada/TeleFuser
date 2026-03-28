"""HuggingFace utilities for loading models and components."""

from __future__ import annotations

import os
from typing import Any

import torch

from telefuser.utils.logging import logger

try:
    import importlib

    from diffusers import __version__ as diffusers_version
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoConfig = None
    AutoProcessor = None
    AutoTokenizer = None


def load_module_from_huggingface(
    module_path: str,
    module_source: str = "transformers",
    module_name: str | None = None,
    module_class: type | None = None,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> Any:
    """Load models from HuggingFace model repository or local folder.

    Args:
        module_path (str): Path to the HuggingFace model repository or local folder
        module_source (str): Source of the model - "transformers" or "diffusers"
        module_name (str, optional): Name to assign to the loaded model. If None, uses folder name
        module_class (class, optional): Specific model class to use for loading. If None, auto-detects
        device (torch.device, optional): Target device. If None, uses self.device
        torch_dtype (torch.dtype, optional): Data type. If None, uses self.torch_dtype
        **kwargs: Additional arguments passed to the model loading function
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and diffusers are required for HuggingFace model loading. "
            "Please install them with: pip install transformers diffusers"
        )

    logger.info(f"Loading HuggingFace model from {module_path} (source: {module_source})")

    # Determine model name
    if module_name is None:
        module_name = os.path.basename(os.path.normpath(module_path))

    try:
        if module_source == "transformers":
            # Load transformers components (Model, Processor, Tokenizer)
            if module_class is None:
                # Try to auto-detect based on model structure or config
                config_path = os.path.join(module_path, "config.json")
                if os.path.exists(config_path):
                    import json

                    with open(config_path, "r") as f:
                        config = json.load(f)

                    # Check if it's a processor or tokenizer by looking for specific files
                    processor_files = ["preprocessor_config.json", "processor_config.json"]
                    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json"]

                    has_processor = any(os.path.exists(os.path.join(module_path, f)) for f in processor_files)
                    has_tokenizer = any(os.path.exists(os.path.join(module_path, f)) for f in tokenizer_files)

                    if has_processor:
                        module_class = AutoProcessor
                        logger.info("Detected processor - using AutoProcessor")
                    elif has_tokenizer:
                        module_class = AutoTokenizer
                        logger.info("Detected tokenizer - using AutoTokenizer")
                    else:
                        # Default to AutoModel for model-like structures
                        config = AutoConfig.from_pretrained(module_path, **kwargs)
                        module_class = AutoModel
                        logger.info("Detected model - using AutoModel")
                else:
                    # If no config.json, try to determine based on folder structure
                    # Look for common processor/tokenizer files
                    processor_files = ["preprocessor_config.json", "processor_config.json"]
                    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json"]

                    has_processor = any(os.path.exists(os.path.join(module_path, f)) for f in processor_files)
                    has_tokenizer = any(os.path.exists(os.path.join(module_path, f)) for f in tokenizer_files)

                    if has_processor:
                        module_class = AutoProcessor
                        logger.info("Detected processor - using AutoProcessor")
                    elif has_tokenizer:
                        module_class = AutoTokenizer
                        logger.info("Detected tokenizer - using AutoTokenizer")
                    else:
                        # Default to AutoModel
                        module_class = AutoModel
                        logger.info("Using AutoModel as default")

            # Load the specific component
            if module_class in [AutoProcessor, AutoTokenizer]:
                # Processors and tokenizers don't have device/dtype parameters
                module = module_class.from_pretrained(module_path, **kwargs)
            else:
                # Models support device and dtype
                module = module_class.from_pretrained(
                    module_path,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )

                # Handle device mapping for non-string devices
                if not isinstance(device, str):
                    module = module.to(device)

        elif module_source == "diffusers":
            # Load diffusers components using dynamic import based on _class_name
            if module_class is None:
                # Try different config files for diffusers components
                config_files = ["config.json", "scheduler_config.json"]
                config_path = None
                config = None

                for config_file in config_files:
                    candidate_path = os.path.join(module_path, config_file)
                    if os.path.exists(candidate_path):
                        config_path = candidate_path
                        break

                if config_path is not None:
                    import json

                    with open(config_path, "r") as f:
                        config = json.load(f)

                    if "_class_name" in config:
                        class_name = config["_class_name"]

                        # Try different import strategies for the class
                        if "." in class_name:
                            # Standard class with module path
                            module_path, class_name_only = class_name.rsplit(".", 1)

                            try:
                                # Import the module
                                module = importlib.import_module(module_path)
                                # Get the class
                                module_class = getattr(module, class_name_only)
                                logger.info(f"Using dynamically imported class: {class_name}")
                            except (ImportError, AttributeError) as e:
                                # If direct import fails, try common diffusers patterns
                                # For classes like QwenImageTransformer2DModel that might be in diffusers
                                if "diffusers" not in module_path:
                                    # Try to find it in diffusers modules
                                    diffusers_modules = [
                                        "diffusers.models",
                                        "diffusers.models.transformers",
                                        "diffusers.models.autoencoders",
                                        "diffusers.models.unets",
                                        "diffusers",
                                    ]

                                    for diffusers_module in diffusers_modules:
                                        try:
                                            module = importlib.import_module(diffusers_module)
                                            if hasattr(module, class_name_only):
                                                module_class = getattr(module, class_name_only)
                                                logger.info(f"Found class {class_name_only} in {diffusers_module}")
                                                break
                                        except (ImportError, AttributeError):
                                            continue
                                    else:
                                        raise ImportError(f"Failed to import class {class_name}: {e}")
                                else:
                                    raise ImportError(f"Failed to import class {class_name}: {e}")
                        else:
                            # Class name without module path - search in diffusers modules
                            diffusers_modules = [
                                "diffusers.models",
                                "diffusers.models.transformers",
                                "diffusers.models.autoencoders",
                                "diffusers.models.unets",
                                "diffusers",
                            ]

                            for diffusers_module in diffusers_modules:
                                try:
                                    module = importlib.import_module(diffusers_module)
                                    if hasattr(module, class_name):
                                        module_class = getattr(module, class_name)
                                        logger.info(f"Found class {class_name} in {diffusers_module}")
                                        break
                                except (ImportError, AttributeError):
                                    continue
                            else:
                                # If not found in diffusers, try local search as fallback
                                try:
                                    # Search for the class in telefuser.models
                                    from telefuser.models import __dict__ as models_dict

                                    if class_name in models_dict:
                                        module_class = models_dict[class_name]
                                        logger.info(f"Using local telefuser model class: {class_name}")
                                    else:
                                        # Try to find it in the global namespace
                                        import sys

                                        current_module = sys.modules[__name__]
                                        if hasattr(current_module, class_name):
                                            module_class = getattr(current_module, class_name)
                                            logger.info(f"Using local model class: {class_name}")
                                        else:
                                            raise AttributeError(
                                                f"Class {class_name} not found in diffusers, telefuser.models, or local scope"  # noqa
                                            )
                                except (ImportError, AttributeError) as e:
                                    raise ImportError(f"Failed to find class {class_name}: {e}")
                    else:
                        raise ValueError("config.json does not contain '_class_name' field")
                else:
                    raise FileNotFoundError(f"config.json not found in {module_path}")

            # Load the specific component
            if not os.path.isdir(module_path):
                module_path = os.path.dirname(module_path)
            print(module_path)
            module = module_class.from_pretrained(
                module_path, torch_dtype=torch_dtype if hasattr(module_class, "from_pretrained") else None, **kwargs
            )

        else:
            raise ValueError(
                f"Unsupported model source: {module_source}. Supported sources are 'transformers' and 'diffusers'"
            )

        # Set model to evaluation mode if it has eval method
        if hasattr(module, "eval"):
            module.eval()

        # Disable gradients for trainable models
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
        return module, module_path, module_name

    except Exception as e:
        logger.error(f"Failed to load HuggingFace model from {module_path}: {e}")
        raise
