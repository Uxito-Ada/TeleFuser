# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_norm.py

import pytest
import torch

import tf_kernel
from tf_kernel.utils import is_arch_support_pdl


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    enable_pdl = is_arch_support_pdl()
    if specify_out:
        y = torch.empty_like(x)
        tf_kernel.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = tf_kernel.rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    enable_pdl = is_arch_support_pdl()
    tf_kernel.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
