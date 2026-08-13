from unittest.mock import MagicMock, patch

import pytest

from examples.wan_video import wan21_1_3b_text_to_video_h100 as example
from telefuser.core.config import AttnImplType, QuantKernelBackend, QuantType


@pytest.mark.parametrize(
    ("name", "expected"),
    [("dense", AttnImplType.TORCH_SDPA), ("sage", AttnImplType.SAGE_ATTN_2_8_8_SM90)],
)
def test_attention_name_resolves_to_dense_backend(name: str, expected: AttnImplType) -> None:
    config = example._make_attention_config(name)
    assert config.attn_impl is expected
    assert config.sparse_config is None


@pytest.mark.parametrize(
    ("name", "quant_type", "backend"),
    [
        ("none", QuantType.FP8, QuantKernelBackend.AUTO),
        ("tf-kernel-fp8", QuantType.FP8, QuantKernelBackend.TF_KERNEL),
        ("torchao-fp8", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO),
    ],
)
def test_quantization_name_resolves_to_runtime_config(name: str, quant_type, backend) -> None:
    config = example._make_quant_config(name)
    assert config.enabled is (name != "none")
    assert config.quant_type is quant_type
    assert config.kernel_backend is backend


def test_example_get_pipeline_forwards_attention_and_quantization() -> None:
    module_manager = MagicMock()
    pipeline = MagicMock()
    with (
        patch.object(example, "ModuleManager", return_value=module_manager),
        patch.object(example, "Wan21VideoPipeline", return_value=pipeline),
    ):
        result = example.get_pipeline(
            model_root="/models/Wan2.1-T2V-1.3B",
            attention="sage",
            quantization="tf-kernel-fp8",
        )

    assert result is pipeline
    dit_load = module_manager.load_models.call_args_list[1]
    assert dit_load.kwargs["quant_config"].quant_type is QuantType.FP8
    assert dit_load.kwargs["quant_config"].kernel_backend is QuantKernelBackend.TF_KERNEL
    config = pipeline.init.call_args.args[1]
    assert config.dit_config.attention_config.attn_impl is AttnImplType.SAGE_ATTN_2_8_8_SM90
    assert config.dit_config.quant_config.quant_type is QuantType.FP8
