/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <optional>
#include <vector>

namespace tf_kernel {

// Forward pass for block sparse attention
std::vector<at::Tensor> block_sparse_attn_fwd(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& head_mask_type,
    std::optional<at::Tensor> streaming_info_,
    std::optional<at::Tensor> row_blockmask_,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double p_dropout,
    double softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t m_block_dim,
    int64_t n_block_dim,
    bool exact_streaming,
    bool return_softmax,
    std::optional<at::Generator> gen_);

// Backward pass for block sparse attention
std::vector<at::Tensor> block_sparse_attn_bwd(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor> dq_,
    std::optional<at::Tensor> dk_,
    std::optional<at::Tensor> dv_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& head_mask_type,
    std::optional<at::Tensor> streaming_info_,
    std::optional<at::Tensor> col_blockmask_,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double p_dropout,
    double softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t m_block_dim,
    int64_t n_block_dim,
    bool deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor> rng_state);

}  // namespace tf_kernel
