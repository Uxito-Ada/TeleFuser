from unittest.mock import patch

import pytest
import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
    SparseAttentionConfig,
)
from telefuser.models.wan_video_dit import SelfAttention, WanModel, precompute_freqs_cis_3d
from telefuser.ops.attention import SparseAttentionState, attention_impl
from telefuser.ops.fp8_attention import (
    dequantize_fp8_per_token,
    quantize_fp8_per_block,
    quantize_fp8_per_token,
)


def test_fp8_attention_quantization_round_trip() -> None:
    torch.manual_seed(0)
    value = torch.randn(1, 8, 2, 128, dtype=torch.bfloat16)
    quantized, scale = quantize_fp8_per_token(value)
    restored = dequantize_fp8_per_token(quantized, scale, torch.bfloat16)

    assert quantized.dtype is torch.float8_e4m3fn
    assert scale.shape == (1, 8, 2)
    assert restored.dtype is torch.bfloat16
    assert torch.isfinite(restored).all()
    assert torch.mean((value.float() - restored.float()).abs()) < 0.03


def test_fp8_attention_block_quantization_round_trip() -> None:
    torch.manual_seed(0)
    value = torch.randn(1, 65, 2, 128, dtype=torch.bfloat16)
    quantized, scale = quantize_fp8_per_block(value)
    restored = dequantize_fp8_per_token(
        quantized,
        scale.repeat_interleave(64, dim=1)[:, : value.shape[1]],
        torch.bfloat16,
    )

    assert quantized.dtype is torch.float8_e4m3fn
    assert scale.shape == (1, 2, 2)
    assert restored.dtype is torch.bfloat16
    assert torch.isfinite(restored).all()
    assert torch.mean((value.float() - restored.float()).abs()) < 0.04


def test_wan_tf_kernel_fp8_quantization_uses_filtered_linear_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList()
    calls: list[tuple[str, object]] = []

    def fake_count(module: torch.nn.Module, **kwargs: object) -> int:
        calls.append(("count", kwargs["module_filter"]))
        return 12

    def fake_enable(module: torch.nn.Module, **kwargs: object) -> None:
        calls.append(("enable", kwargs))

    monkeypatch.setattr("telefuser.ops.fp8_gemm.count_linear_layers", fake_count)
    monkeypatch.setattr("telefuser.ops.fp8_gemm.enable_fp8_gemm", fake_enable)

    model.enable_quant(
        QuantConfig(
            enabled=True,
            quant_type=QuantType.FP8,
            kernel_backend=QuantKernelBackend.TF_KERNEL,
        )
    )

    assert calls[0][0] == "count"
    assert calls[1][0] == "enable"
    options = calls[1][1]["options"]
    assert options.fp16_weight_storage == "discard"
    assert model.tf_kernel_fp8_replaced_linear == 12
    assert model.quant_type is QuantType.FP8


def test_wan_model_enables_sol_attention_state() -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)

    sparse_config = SparseAttentionConfig(
        sparse_impl="sol",
        dense_layers=2,
        dense_timesteps=3,
        sol_tau=0.75,
        sol_threshold_type="exact",
        sol_kv_splits=2,
    )
    model.enable_sol_attention(height=64, width=64, num_frames=5, sparse_config=sparse_config)

    state = model.create_sparse_state(numeral_timestep=4, layer_idx=5)
    assert state is not None
    assert state.mask_map is None
    assert state.numeral_timestep == 4
    assert state.layer_idx == 5
    assert state.config is sparse_config
    perm = model.sol_morton_perm
    inverse = model.sol_morton_inverse
    torch.testing.assert_close(perm.index_select(0, inverse), torch.arange(perm.numel()))


def test_wan_sol_attention_uses_official_dense_guards() -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    config = AttentionConfig.sol_attention()
    assert config.sparse_config is not None
    model.enable_sol_attention(height=480, width=832, num_frames=81, sparse_config=config.sparse_config)

    state = model.create_sparse_state(numeral_timestep=0, layer_idx=1)
    assert state is not None and state.should_use_dense()
    state.update(numeral_timestep=10, layer_idx=0)
    assert state.should_use_dense()
    state.update(numeral_timestep=10, layer_idx=1)
    assert not state.should_use_dense()
    assert model.sol_morton_perm.numel() == 21 * 30 * 52


def test_wan_sol_token_order_round_trip() -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    config = AttentionConfig.sol_attention()
    assert config.sparse_config is not None
    model.enable_sol_attention(height=64, width=64, num_frames=5, sparse_config=config.sparse_config)
    state = model.create_sparse_state()

    tokens = model.sol_morton_perm.numel()
    x = torch.arange(tokens).reshape(1, tokens, 1)
    t_mod = x.clone()
    freqs_cos = torch.arange(tokens).reshape(tokens, 1)
    freqs_sin = -freqs_cos

    ordered = model._apply_sol_token_order(x, t_mod, freqs_cos, freqs_sin, state, reorder_tokens=True)
    perm = model.sol_morton_perm
    torch.testing.assert_close(ordered[0], x.index_select(1, perm))
    torch.testing.assert_close(ordered[1], t_mod.index_select(1, perm))
    torch.testing.assert_close(ordered[2], freqs_cos.index_select(0, perm))
    torch.testing.assert_close(ordered[3], freqs_sin.index_select(0, perm))
    torch.testing.assert_close(model._restore_sol_token_order(ordered[0], state), x)


def test_wan_self_attention_dispatches_sol_through_public_ops() -> None:
    module = SelfAttention(dim=128, num_heads=1)
    sparse_config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=0)
    module.attention_config = AttentionConfig(attn_impl=AttnImplType.SOL_ATTN, sparse_config=sparse_config)
    state = SparseAttentionState(sparse_config, mask_map=None)
    captured = {}

    def fake_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        captured.update(kwargs)
        return q

    x = torch.randn(1, 4, 128)
    with (
        patch("telefuser.models.wan_video_dit.rope_apply", side_effect=lambda tensor, *_args: tensor),
        patch("telefuser.models.wan_video_dit.attn_func", side_effect=fake_attention),
    ):
        output = module.default_forward(x, torch.empty(0), torch.empty(0), sparse_state=state)

    assert output.shape == x.shape
    assert captured["attention_config"].attn_impl is AttnImplType.SOL_ATTN
    assert captured["attention_config"].sparse_config is sparse_config
    assert captured["sparse_state"] is state


def test_wan_self_attention_only_quantizes_qkv_after_sol_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SelfAttention(dim=128, num_heads=1).to(torch.bfloat16)
    sparse_config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=2, sol_fp8=True)
    state = SparseAttentionState(sparse_config, mask_map=None)
    captured: list[torch.dtype] = []

    def fake_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.append(q.dtype)
        return q.to(torch.bfloat16)

    monkeypatch.setattr("telefuser.models.wan_video_dit.attn_func", fake_attention)
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    freqs = torch.zeros(4, 64, dtype=torch.bfloat16)

    module.default_forward(x, freqs, freqs, sparse_state=state)
    state.update(numeral_timestep=2, layer_idx=1)
    module.default_forward(x, freqs, freqs, sparse_state=state)

    assert captured == [torch.bfloat16, torch.float8_e4m3fn]


@pytest.mark.gpu
def test_wan_self_attention_executes_sol_on_h100(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Wan Sol-Attn execution test requires H100")

    assert attention_impl.SOL_ATTN_AVAILABLE
    assert attention_impl.sol_attn is not None
    kernel_calls = 0
    sol_attn = attention_impl.sol_attn

    def tracked_sol_attn(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return sol_attn(*args, **kwargs)

    monkeypatch.setattr(attention_impl, "sol_attn", tracked_sol_attn)

    module = SelfAttention(dim=128, num_heads=1).eval().cuda().to(torch.bfloat16)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    freqs = precompute_freqs_cis_3d(128)
    freqs_cos = torch.cat([freq.real for freq in freqs], dim=-1)[:256].cuda()
    freqs_sin = torch.cat([freq.imag for freq in freqs], dim=-1)[:256].cuda()
    sparse_config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=0, sol_tau=-1000.0)
    module.attention_config = AttentionConfig(attn_impl=AttnImplType.SOL_ATTN, sparse_config=sparse_config)
    state = SparseAttentionState(sparse_config, mask_map=None)

    output = module(x, freqs_cos, freqs_sin, sparse_state=state)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert kernel_calls == 1
