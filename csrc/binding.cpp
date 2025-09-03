// csrc/binding.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attention_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lse_reduce_kernel_launcher", &lse_reduce_kernel_launcher, "Reduce kernel optimized (CUDA)",
          pybind11::arg("lse"),
          pybind11::arg("m_ij"),
          pybind11::arg("l_ij_first"),
          pybind11::arg("l_ij_rest"),
          pybind11::arg("m_ij_last"),
          pybind11::arg("o"),
          pybind11::arg("o_tiles_first"),
          pybind11::arg("o_tiles_rest"),
          pybind11::arg("acc_o_scales_first"),
          pybind11::arg("acc_o_scales_rest"),
          pybind11::arg("t"),
          pybind11::arg("token_index_mapping"),
          pybind11::arg("start_head_id"),
          pybind11::arg("total_len"),
          pybind11::arg("topk")
    );

    m.def("o_reduce_kernel_launcher", &o_reduce_kernel_launcher, "Reduce kernel optimized (CUDA)",
          pybind11::arg("lse"),
          pybind11::arg("m_ij"),
          pybind11::arg("l_ij_first"),
          pybind11::arg("l_ij_rest"),
          pybind11::arg("m_ij_last"),
          pybind11::arg("o"),
          pybind11::arg("o_tiles_first"),
          pybind11::arg("o_tiles_rest"),
          pybind11::arg("acc_o_scales_first"),
          pybind11::arg("acc_o_scales_rest"),
          pybind11::arg("t"),
          pybind11::arg("token_index_mapping"),
          pybind11::arg("start_head_id"),
          pybind11::arg("total_len"),
          pybind11::arg("topk")
    );
}

// #include <pybind11/pybind11.h>
// #include <torch/extension.h>

// // Simple test function
// torch::Tensor test_add(torch::Tensor a, torch::Tensor b) {
//     return a + b;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("test_add", &test_add, "Add two tensors");
// }
