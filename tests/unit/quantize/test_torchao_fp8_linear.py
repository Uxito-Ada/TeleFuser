import sys
from types import ModuleType
from unittest.mock import Mock

import torch.nn as nn

from telefuser.ops import torchao_fp8_linear


def test_torchao_fp8_uses_weight_only_config(monkeypatch) -> None:
    class Float8WeightOnlyConfig:
        pass

    quantize = Mock()
    torchao_module = ModuleType("torchao")
    quantization_module = ModuleType("torchao.quantization")
    quantization_module.quantize_ = quantize
    monkeypatch.setitem(sys.modules, "torchao", torchao_module)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization_module)
    monkeypatch.setattr(torchao_fp8_linear.metadata, "version", lambda _: "test")

    requested_attrs = None

    def import_config(module_names, attr_names):
        nonlocal requested_attrs
        requested_attrs = attr_names
        return Float8WeightOnlyConfig

    monkeypatch.setattr(torchao_fp8_linear, "_import_first_attr", import_config)

    model = nn.Sequential(nn.Linear(4, 4))
    replaced = torchao_fp8_linear.replace_linear_layers_with_torchao_fp8(model)

    assert replaced == 1
    assert requested_attrs == ("Float8WeightOnlyConfig", "float8_weight_only")
    assert isinstance(quantize.call_args.args[1], Float8WeightOnlyConfig)
