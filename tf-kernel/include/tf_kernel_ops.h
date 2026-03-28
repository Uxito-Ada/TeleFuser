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
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

#include "scalar_type.hpp"

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

using fptr_t = int64_t;

/*
 * From csrc/elementwise
 */
void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl);
void tf_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl);
void silu_and_mul(at::Tensor& out, at::Tensor& input);
void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input);
void gelu_and_mul(at::Tensor& out, at::Tensor& input);

void apply_rope_pos_ids_cos_sin_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    bool enable_pdl,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc);

void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox);

void downcast_fp8(
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& k_out,
    at::Tensor& v_out,
    at::Tensor& k_scale,
    at::Tensor& v_scale,
    at::Tensor& loc,
    int64_t mult,
    int64_t offset);

void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);

/*
 * From FlashInfer (norm operations)
 */
void gemma_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl);

/*
 * From csrc/gemm
 */
#ifdef ENABLE_NVFP4
void cutlass_scaled_fp4_mm(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha);
#endif
torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias);
torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias);
torch::Tensor fp8_blockwise_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype);
#ifdef ENABLE_NVFP4
void scaled_fp4_quant(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_scale, torch::Tensor const& input_scale);

#endif
void tf_per_token_group_quant_8bit(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0);
void tf_per_token_group_quant_8bit_v2(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const std::optional<torch::Tensor>& masked_m);
void tf_per_tensor_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s, bool is_static);
void tf_per_token_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s);
void bmm_fp8(
    at::Tensor A,
    at::Tensor B,
    at::Tensor D,
    at::Tensor A_scale,
    at::Tensor B_scale,
    at::Tensor workspace_buffer,
    int64_t cublas_handle);

#ifdef ENABLE_NVFP4
void scaled_fp4_experts_quant(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale,
    torch::Tensor const& mask,
    bool use_silu_and_mul);
#endif

/*
 * From csrc/memory
 */
at::Tensor weak_ref_tensor(const at::Tensor& tensor);
void store_kv_cache(at::Tensor k_cache, at::Tensor v_cache, at::Tensor out_loc, at::Tensor k, at::Tensor v);

/*
 * From csrc/block_sparse_attn
 */
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
