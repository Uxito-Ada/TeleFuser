import sys
import types

import torch.nn as nn

from telefuser.ops import torchao_fp8_linear


class TinyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.GELU())


def test_replace_linear_layers_with_torchao_fp8_filters_linear_layers(monkeypatch):
    """TorchAO FP8 helper should count and pass only selected Linear layers."""
    model = TinyLinearModel()
    selected_names = []

    def fake_quantize_(module, quant_config, filter_fn=None):
        for name, child in module.named_modules():
            if filter_fn is not None and filter_fn(child, name):
                selected_names.append(name)

    fake_quant_module = types.SimpleNamespace(
        quantize_=fake_quantize_,
        Float8DynamicActivationFloat8WeightConfig=type("FakeFloat8Config", (), {}),
    )
    fake_torchao = types.SimpleNamespace(quantization=fake_quant_module)
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_quant_module)
    monkeypatch.setattr(torchao_fp8_linear, "_check_torchao_fp8_available", lambda: None)

    replaced = torchao_fp8_linear.replace_linear_layers_with_torchao_fp8(model)

    assert replaced == 2
    assert selected_names == ["keep", "block.0"]
