// attention_kernel.h
#pragma once

#include <torch/extension.h>

void lse_reduce_kernel_launcher(
    torch::Tensor lse,
    torch::Tensor m_ij,
    torch::Tensor l_ij_first,
    torch::Tensor l_ij_rest,
    torch::Tensor m_ij_last,
    torch::Tensor o,
    torch::Tensor o_tiles_first,
    torch::Tensor o_tiles_rest,
    torch::Tensor acc_o_scales_first,
    torch::Tensor acc_o_scales_rest,
    torch::Tensor t,
    torch::Tensor token_index_mapping,
    int start_head_id,
    int total_len,
    int topk
);


void o_reduce_kernel_launcher(
    torch::Tensor lse,
    torch::Tensor m_ij,
    torch::Tensor l_ij_first,
    torch::Tensor l_ij_rest,
    torch::Tensor m_ij_last,
    torch::Tensor o,
    torch::Tensor o_tiles_first,
    torch::Tensor o_tiles_rest,
    torch::Tensor acc_o_scales_first,
    torch::Tensor acc_o_scales_rest,
    torch::Tensor t,
    torch::Tensor token_index_mapping,
    int start_head_id,
    int total_len,
    int topk
);