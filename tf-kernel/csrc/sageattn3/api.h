// FP4 quantization functions for SageAttention3
void sageattn3_scaled_fp4_quant(
    torch::Tensor const& input, torch::Tensor const& output, torch::Tensor const& output_sf, int64_t tensor_layout);

void sageattn3_scaled_fp4_quant_permute(
    torch::Tensor const& input, torch::Tensor const& output, torch::Tensor const& output_sf, int64_t tensor_layout);

void sageattn3_scaled_fp4_quant_trans(
    torch::Tensor const& input, torch::Tensor const& output, torch::Tensor const& output_sf, int64_t tensor_layout);

// FP4 attention function for SageAttention3 (Blackwell)
// out: output tensor (batch_size x num_heads x seqlen_q x head_size), in-place output
// return_lse: if > 0, return softmax_lse tensor
at::Tensor sageattn3_fp4_attn(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& sfq,
    const at::Tensor& sfk,
    const at::Tensor& sfv,
    const at::Tensor& delta_s,
    int64_t unpadded_k,
    at::Tensor& out,
    const double softmax_scale,
    bool is_causal,
    bool per_block_mean,
    bool is_bf16,
    int64_t return_lse);
