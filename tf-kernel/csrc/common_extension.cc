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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "block_sparse_attn/block_sparse_attn_api.h"
#include "sageattn2/fused/fused.h"
#include "sageattn2/qattn/attn_cuda_sm89.h"
#include "tf_kernel_ops.h"
#ifdef ENABLE_NVFP4
#include "sageattn3/api.h"
#endif

TORCH_LIBRARY_FRAGMENT(tf_kernel, m) {
  /*
   * From csrc/elementwise
   */
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &tf_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def(
      "apply_rope_pos_ids_cos_sin_cache(Tensor q, Tensor k, Tensor! q_rope, Tensor! k_rope, Tensor cos_sin_cache, "
      "Tensor pos_ids, bool interleave, bool enable_pdl, "
      "Tensor? v, Tensor!? k_buffer, Tensor!? v_buffer, Tensor? kv_cache_loc) -> ()");
  m.impl("apply_rope_pos_ids_cos_sin_cache", torch::kCUDA, &apply_rope_pos_ids_cos_sin_cache);

  m.def("copy_to_gpu_no_ce(Tensor input, Tensor! output) -> ()");
  m.impl("copy_to_gpu_no_ce", torch::kCUDA, &copy_to_gpu_no_ce);

  /*
   * Additional elementwise operators
   */
  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_rmsnorm", torch::kCUDA, &gemma_rmsnorm);

  m.def(
      "rotary_embedding(Tensor positions, Tensor query, Tensor? key, int head_size, Tensor cos_sin_cache, bool "
      "is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  m.def(
      "downcast_fp8(Tensor k, Tensor v, Tensor! k_out, Tensor! v_out, Tensor k_scale, Tensor v_scale, Tensor loc, int "
      "mult, int offset) -> ()");
  m.impl("downcast_fp8", torch::kCUDA, &downcast_fp8);

  /*
   * From csrc/gemm
   */
  m.def(
      "int8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("int8_scaled_mm", torch::kCUDA, &int8_scaled_mm);

  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("fp8_scaled_mm", torch::kCUDA, &fp8_scaled_mm);

  m.def(
      "fp8_blockwise_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype) -> "
      "Tensor");
  m.impl("fp8_blockwise_scaled_mm", torch::kCUDA, &fp8_blockwise_scaled_mm);

  m.def(
      "tf_per_token_group_quant_8bit(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");
  m.impl("tf_per_token_group_quant_8bit", torch::kCUDA, &tf_per_token_group_quant_8bit);

  m.def("tf_per_tensor_quant_fp8(Tensor input, Tensor! output_q, Tensor! output_s, bool is_static) -> ()");
  m.impl("tf_per_tensor_quant_fp8", torch::kCUDA, &tf_per_tensor_quant_fp8);

  m.def("tf_per_token_quant_fp8(Tensor input, Tensor! output_q, Tensor! output_s) -> ()");
  m.impl("tf_per_token_quant_fp8", torch::kCUDA, &tf_per_token_quant_fp8);

  m.def(
      "tf_per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, Tensor! output_s, int group_size, float eps, "
      "float min_8bit, float max_8bit, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
  m.impl("tf_per_token_group_quant_8bit_v2", torch::kCUDA, &tf_per_token_group_quant_8bit_v2);
#ifdef ENABLE_NVFP4
  m.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");
  m.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  m.def(
      "scaled_fp4_quant(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale, Tensor! input_scale) -> ()");
  m.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  // Compute NVFP4 experts quantization.
  m.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  m.impl("scaled_fp4_experts_quant", torch::kCUDA, &scaled_fp4_experts_quant);

  m.def(
      "silu_and_mul_scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor mask, bool use_silu_and_mul) -> ()");
  m.impl("silu_and_mul_scaled_fp4_experts_quant", torch::kCUDA, &silu_and_mul_scaled_fp4_experts_quant);

  // SageAttention3 FP4 quantization functions
  m.def("sageattn3_scaled_fp4_quant(Tensor input, Tensor! output, Tensor! output_sf, int tensor_layout) -> ()");
  m.impl("sageattn3_scaled_fp4_quant", torch::kCUDA, &sageattn3_scaled_fp4_quant);

  m.def("sageattn3_scaled_fp4_quant_permute(Tensor input, Tensor! output, Tensor! output_sf, int tensor_layout) -> ()");
  m.impl("sageattn3_scaled_fp4_quant_permute", torch::kCUDA, &sageattn3_scaled_fp4_quant_permute);

  m.def("sageattn3_scaled_fp4_quant_trans(Tensor input, Tensor! output, Tensor! output_sf, int tensor_layout) -> ()");
  m.impl("sageattn3_scaled_fp4_quant_trans", torch::kCUDA, &sageattn3_scaled_fp4_quant_trans);

  // SageAttention3 FP4 attention function (Blackwell)
  // out: output tensor (batch_size x num_heads x seqlen_q x head_size), in-place output
  // return_lse: if > 0, return softmax_lse tensor
  m.def(
      "sageattn3_fp4_attn(Tensor q, Tensor k, Tensor v, Tensor sfq, Tensor sfk, Tensor sfv, Tensor delta_s, int "
      "unpadded_k, Tensor! out, float softmax_scale, bool is_causal, bool per_block_mean, bool is_bf16, int "
      "return_lse) -> Tensor");
  m.impl("sageattn3_fp4_attn", torch::kCUDA, &sageattn3_fp4_attn);
#endif

  /*
   * From csrc/memory
   */
  m.def("store_kv_cache(Tensor k_cache, Tensor v_cache, Tensor out_loc, Tensor k, Tensor v) -> ()");
  m.impl("store_kv_cache", &store_kv_cache);

  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");
  m.impl("weak_ref_tensor", &weak_ref_tensor);

  /*
   * From FlashInfer
   */
  m.def(
      "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
      "int cublas_handle) -> ()",
      {at::Tag::needs_fixed_stride_order});
  m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
  /*
   * From SageAttention fused operations
   */
  m.def(
      "quant_per_block_int8_fuse_sub_mean_cuda(Tensor input, Tensor mean, Tensor! output, Tensor! scale, int "
      "block_size, int tensor_layout) -> ()");
  m.impl("quant_per_block_int8_fuse_sub_mean_cuda", torch::kCUDA, &quant_per_block_int8_fuse_sub_mean_cuda);

  m.def(
      "quant_per_block_int8_cuda_with_sm_scale(Tensor input, Tensor! output, Tensor! scale, float sm_scale, int "
      "block_size, int tensor_layout) -> ()");
  m.impl(
      "quant_per_block_int8_cuda_with_sm_scale",
      torch::kCUDA,
      static_cast<void (*)(torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t)>(
          &quant_per_block_int8_cuda));

  m.def(
      "quant_per_block_int8_cuda(Tensor input, Tensor! output, Tensor! scale, int block_size, int tensor_layout) -> "
      "()");
  m.impl(
      "quant_per_block_int8_cuda",
      torch::kCUDA,
      static_cast<void (*)(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t)>(&quant_per_block_int8_cuda));

  m.def(
      "quant_per_warp_int8_cuda(Tensor input, Tensor! output, Tensor! scale, int block_size, int warp_block_size, int "
      "tensor_layout) -> ()");
  m.impl("quant_per_warp_int8_cuda", torch::kCUDA, &quant_per_warp_int8_cuda);

  m.def("sub_mean_cuda(Tensor input, Tensor mean, Tensor! output, int tensor_layout) -> ()");
  m.impl("sub_mean_cuda", torch::kCUDA, &sub_mean_cuda);

  m.def("transpose_pad_permute_cuda(Tensor input, Tensor! output, int tensor_layout) -> ()");
  m.impl("transpose_pad_permute_cuda", torch::kCUDA, &transpose_pad_permute_cuda);

  m.def(
      "scale_fuse_quant_cuda(Tensor input, Tensor! output, Tensor! scale, int num_tokens, float scale_max, int "
      "tensor_layout) -> ()");
  m.impl("scale_fuse_quant_cuda", torch::kCUDA, &scale_fuse_quant_cuda);

  m.def(
      "mean_scale_fuse_quant_cuda(Tensor input, Tensor! output, Tensor! mean, Tensor! scale, int num_tokens, float "
      "scale_max, int tensor_layout) -> ()");
  m.impl("mean_scale_fuse_quant_cuda", torch::kCUDA, &mean_scale_fuse_quant_cuda);

  /*
   * From SageAttention qattn operations (SM80)
   */
  m.def(
      "qk_int8_sv_f16_accum_f32_attn(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor query_scale, "
      "Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> "
      "Tensor");
  m.impl("qk_int8_sv_f16_accum_f32_attn", torch::kCUDA, &qk_int8_sv_f16_accum_f32_attn);

  m.def(
      "qk_int8_sv_f16_accum_f16_attn(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor query_scale, "
      "Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> "
      "Tensor");
  m.impl("qk_int8_sv_f16_accum_f16_attn", torch::kCUDA, &qk_int8_sv_f16_accum_f16_attn);

  m.def(
      "qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor "
      "query_scale, Tensor key_scale, Tensor value_mean, int tensor_layout, int is_causal, int qk_quant_gran, float "
      "sm_scale, int return_lse) -> Tensor");
  m.impl("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", torch::kCUDA, &qk_int8_sv_f16_accum_f16_fuse_v_mean_attn);

  m.def(
      "qk_int8_sv_f16_accum_f16_attn_inst_buf(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor "
      "query_scale, Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int "
      "return_lse) -> Tensor");
  m.impl("qk_int8_sv_f16_accum_f16_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f16_accum_f16_attn_inst_buf);

  /*
   * From SageAttention qattn operations (SM89)
   */
  m.def(
      "qk_int8_sv_f8_accum_f32_attn(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor query_scale, Tensor "
      "key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
  m.impl("qk_int8_sv_f8_accum_f32_attn", torch::kCUDA, &qk_int8_sv_f8_accum_f32_attn);

  m.def(
      "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor "
      "query_scale, Tensor key_scale, Tensor value_scale, int tensor_layout, int is_causal, int qk_quant_gran, float "
      "sm_scale, int return_lse) -> Tensor");
  m.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", torch::kCUDA, &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn);

  m.def(
      "qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(Tensor query, Tensor key, Tensor value, Tensor! output, "
      "Tensor query_scale, Tensor key_scale, Tensor value_scale, Tensor value_mean, int tensor_layout, int is_causal, "
      "int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
  m.impl(
      "qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn",
      torch::kCUDA,
      &qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn);

  m.def(
      "qk_int8_sv_f8_accum_f32_attn_inst_buf(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor "
      "query_scale, Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int "
      "return_lse) -> Tensor");
  m.impl("qk_int8_sv_f8_accum_f32_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f8_accum_f32_attn_inst_buf);

  m.def(
      "qk_int8_sv_f8_accum_f16_attn_inst_buf(Tensor query, Tensor key, Tensor value, Tensor! output, Tensor "
      "query_scale, Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int "
      "return_lse) -> Tensor");
  m.impl("qk_int8_sv_f8_accum_f16_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f8_accum_f16_attn_inst_buf);

  m.def(
      "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(Tensor query, Tensor key, Tensor value, Tensor! output, "
      "Tensor query_scale, Tensor key_scale, Tensor value_scale, int tensor_layout, int is_causal, int qk_quant_gran, "
      "float sm_scale, int return_lse) -> Tensor");
  m.impl(
      "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
      torch::kCUDA,
      &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf);

  m.def(
      "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(Tensor query, Tensor key, Tensor value, Tensor! output, "
      "Tensor query_scale, Tensor key_scale, Tensor value_scale, int tensor_layout, int is_causal, int qk_quant_gran, "
      "float sm_scale, int return_lse) -> Tensor");
  m.impl(
      "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
      torch::kCUDA,
      &qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf);

  /*
   * From block_sparse_attn
   */
  m.def(
      "block_sparse_attn_fwd(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor "
      "head_mask_type, Tensor? streaming_info, Tensor? row_blockmask, int max_seqlen_q, int max_seqlen_k, float "
      "p_dropout, float softmax_scale, bool is_causal, int window_size_left, int window_size_right, int m_block_dim, "
      "int n_block_dim, bool exact_streaming, bool return_softmax) -> Tensor[]");
  m.impl(
      "block_sparse_attn_fwd",
      torch::kCUDA,
      [](at::Tensor& q,
         const at::Tensor& k,
         const at::Tensor& v,
         const at::Tensor& cu_seqlens_q,
         const at::Tensor& cu_seqlens_k,
         const at::Tensor& head_mask_type,
         std::optional<at::Tensor> streaming_info,
         std::optional<at::Tensor> row_blockmask,
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
         bool return_softmax) -> std::vector<at::Tensor> {
        return tf_kernel::block_sparse_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            row_blockmask,
            max_seqlen_q,
            max_seqlen_k,
            p_dropout,
            softmax_scale,
            is_causal,
            window_size_left,
            window_size_right,
            m_block_dim,
            n_block_dim,
            exact_streaming,
            return_softmax,
            std::nullopt);
      });

  m.def(
      "block_sparse_attn_bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, Tensor? dq, "
      "Tensor? dk, Tensor? dv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor head_mask_type, Tensor? "
      "streaming_info, Tensor? col_blockmask, int max_seqlen_q, int max_seqlen_k, float p_dropout, float "
      "softmax_scale, bool zero_tensors, bool is_causal, int window_size_left, int window_size_right, int m_block_dim, "
      "int n_block_dim, bool deterministic, Tensor? rng_state) -> Tensor[]");
  m.impl(
      "block_sparse_attn_bwd",
      torch::kCUDA,
      [](const at::Tensor& dout,
         const at::Tensor& q,
         const at::Tensor& k,
         const at::Tensor& v,
         const at::Tensor& out,
         const at::Tensor& softmax_lse,
         std::optional<at::Tensor> dq,
         std::optional<at::Tensor> dk,
         std::optional<at::Tensor> dv,
         const at::Tensor& cu_seqlens_q,
         const at::Tensor& cu_seqlens_k,
         const at::Tensor& head_mask_type,
         std::optional<at::Tensor> streaming_info,
         std::optional<at::Tensor> col_blockmask,
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
         std::optional<at::Tensor> rng_state) -> std::vector<at::Tensor> {
        return tf_kernel::block_sparse_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            col_blockmask,
            max_seqlen_q,
            max_seqlen_k,
            p_dropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            window_size_left,
            window_size_right,
            m_block_dim,
            n_block_dim,
            deterministic,
            std::nullopt,
            rng_state);
      });
}

REGISTER_EXTENSION(common_ops)
