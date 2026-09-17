<img width="6200" height="1294" alt="github_title" src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" />

---

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2508.18224-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18224)

</div>

This repository provides the official implementation of **<ins>F</ins>lash <ins>S</ins>parse <ins>A</ins>ttention (FSA)**, which includes a novel kernel design that enables efficient Native Sparse Attention (NSA) across a wide range of popular LLMs on modern GPUs.

- [News](#news)
- [Method](#method)
- [Advantages](#advantages)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Instantiate FSA Module](#instantiate-fsa-module)
  - [Train with FSA](#train-with-fsa)
- [Evaluation](#evaluation)
  - [Benchmark FSA Module](#benchmark-fsa-module)
  - [Benchmark FSA Selected Attention Module](#benchmark-fsa-selected-attention-module)
- [Performance](#performance)
  - [Kernel Performance](#kernel-performance)
  - [End-to-end Performance](#end-to-end-performance)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## News

- **$\texttt{[2025-09, upcoming]}$:** 🚀 Online profiling module, which seamlessly transitions between NSA and FSA, will be released soon.
- **$\texttt{[2025-08]}$:** 💥 Our [Arxiv paper](https://www.arxiv.org/abs/2508.18224) is released.
- **$\texttt{[2025-08]}$:** 🎈 Beta version of one-step decoding is released, check the code residing in [`fsa_preview`](fsa_preview).
- **$\texttt{[2025-08]}$:** 🎉 Open sourced `Flash-Sparse-Attention`, offering an optimized implementation for NSA, broadening the applicability of this novel natively trainable sparse attention technique.

## Method

For NSA selected attention module, the major system bottleneck, NSA loops over query tokens in the outer loop and loops over KV blocks in the inner loop. To optimize performance, NSA batches query heads that share the same key-value head for more efficient computation. However, when GQA group size is not sufficiently large, NSA selected attention kernel must pad data to satisfy hardware requirements on the matrix dimensions of matrix multiplication. Specifically, for NVIDIA GPUs, the [warp-level matrix multiply-accumulate instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions) require that each dimension of a matrix tile executed on a warp must be larger than the specified value (at least 8 on Hopper GPUs). In [Triton](https://triton-lang.org/main/), NSA selected attention kernel must also satisfy that each dimension of a matrix tile executed on a thread block must be at least 16.


In contrast, FSA exchanges the kernel loop order of original NSA kernel design, i.e., FSA loops over KV blocks in the outer loop and loops over query tokens in the inner loop. To optimize performance, FSA decouples the computation into three major kernels: (i) the main kernel batches query tokens that attend to the same KV block and stores the partial results to a buffer, (ii) the reduction kernel accumulates attention results for each query token, and (iii) the online softmax kernel that handles online softmax statistics computation. The key insight behind this arrangement is to effectively reduce unnecessary memory access and computations for the padded data, while avoiding `atomic` additions for accumulating attention results for each query token across KV blocks.

The concrete computation process comparison between NSA (left) and FSA main kernel (right) can be visualized as follows:

![combine2 (1)](https://github.com/user-attachments/assets/00ac87a6-1af3-4f78-b019-b2d4edf997e1)


## Advantages

🚀 The speedup of FSA originates from significantly lowered kernel-level memory access volume and computations.

Under varied GQA group sizes, NSA hyperparameters block size $B_K=64$ and topk-k value $T=16$, 64K sequence length, 4 KV heads, the execution latency comparisons between NSA and our method are as follows (execution latency of our method is normalized to 1):
<img width="4320" height="2592" alt="GQA_comp" src="https://github.com/user-attachments/assets/8cd7d3c2-4b8b-4e9b-bce9-ce290cb792fe" />



## Features

FSA provides an optimized kernel implementation for NSA selected attention module. Without modifying NSA algorithm, FSA provides an efficient Triton-based implementation for GQA group sizes smaller than 8, which is more popular on state-of-the-art large language models (LLMs), on modern high-performance NVIDIA GPUs. For GQA group sizes larger than or equal to 8, FSA usually chooses to fall back to the original NSA implementation for better performance.

FSA is currently well tested with:
- NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 PCIe, H100 NVL, H100 SXM, H200 SXM);
- Datatype of fp16 and bf16;
- The same head dimension (less than or equal to 256) across query, key, and value;
- Varied GQA group sizes, ranging from 1 to 16;
- Training and inference (prefill).

## Installation

The following requirements should be satisfied:
- [PyTorch](https://pytorch.org/) >= 2.4
- [Triton](https://github.com/openai/triton) >=3.0
- [transformers](https://github.com/huggingface/transformers) >=4.45.0
- [datasets](https://github.com/huggingface/datasets) >=3.3.0
- [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
- [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

You can install dependencies for FSA with:
```sh
pip install -r requirements.txt
```

## Usage

### Instantiate FSA Module

We provide [``FlashSparseAttention``](fsa/module/FSA.py) class for you to use, it can be used as the following example:
```Python
import torch
from fsa.module.fsa import FlashSparseAttention, RopeConfig

FSA = (
    FlashSparseAttention(
        hidden_size=4096,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=128,
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        topk=16,
        init_blocks=1,
        local_blocks=2,
        window_size=512,
        rope_config=RopeConfig(
            max_position_embeddings=131072,
            head_dim=128,
            rope_theta=500000,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        ),
    )
    .cuda()
    .to(torch.bfloat16)
)
# random input
seqlens = torch.LongTensor([65536, 32768]).int().cuda()

cu_seqlens = torch.cat(
    [
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(seqlens, dim=0),
    ],
    dim=0,
).to(torch.int32)
x = torch.randn(cu_seqlens[-1], 4096, device="cuda", dtype=torch.bfloat16)

y = FSA(x, cu_seqlens)
loss = (y * torch.randn_like(y)).sum(-1).mean()
loss.backward()
```

Under the hood, the [``FSATopkSparseAttention``](fsa/ops/FSA_topk_sparse_attention.py) class is called, provding the optimized kernels that accelerate the NSA selected attention module.

### Train with FSA

Training with FSA can be esaily achieved by replacing the attention module. The only thing you may need to handle is to instantiate the FSA module, and compute the ``cu_seqlens`` for FSA. We provide an example on how to insert FSA into a LLM in [``SparseLlamaAttention``](test/train.py).

## Evaluation

### Benchmark FSA Module

We provide detailed commands in [`scripts/run_unit_test.sh`](scripts/run_unit_test.sh) for convenient benchmarking of FSA module. The benchmarking provides correctness comparison of forward and backward outputs, performance comparison, and memory usage comparison.


### Benchmark FSA Selected Attention Module

The optimized NSA selected attention module, which is the major system bottleneck, can be benchmarked through the commands in [``scripts/run_unit_test_sel_attn.sh``](scripts/run_unit_test_sel_attn.sh).

> [!Tip]
Try varied ``gqa``, `seqlen`, `block_size`, `topk` argument in the provided scripts for more comprehensive benchmarking on your machine! Compared to benchmarking the FSA attention module, benchmarking the FSA selected attention module usually provides a higher speedup.

## Performance

### Kernel Performance

> Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $BK$ and top-k value $Topk$, respectively. For FSA and NSA, the execution latency is composed of compressed, selected, and sliding attention; for Full Attention, the execution latency is the Flash Attention kernel execution latency.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

> End-to-end training (right) and prefill (left) latency of state-of-the-art LLMs with FSA, NSA, or Full Attention.

<img width="6165" height="3093" alt="e2e_githubpic" src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" />


## Citation

Please cite our [arXiv paper](https://arxiv.org/abs/2508.18224):

```bibtex
@misc{yan2026fsaalternativeefficientimplementation,
  title={{FSA}: An Alternative Efficient Implementation of Native Sparse Attention Kernel},
  author={Ran Yan and Youhe Jiang and Zhuoming Chen and Haohui Mai and Beidi Chen and Binhang Yuan},
  year={2026},
  eprint={2508.18224},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2508.18224},
}
```

**Citation correction:** The [previous README citation](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/blob/7ff144fd7ff485dc4220d439f31cc1708b64fef3/README.md#L174-L184) used the title "Flash Sparse Attention: More Efficient Natively Trainable Sparse Attention" and listed only Ran Yan, Youhe Jiang, and Binhang Yuan. It referred to the same paper, arXiv:2508.18224, but its title and author list were incorrect. The entry above matches the current arXiv record. This note and [issue #15](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/issues/15) document the correction for readers who used the earlier citation.

## Acknowledgments

NSA paper:
[Native Sparse Attention](https://arxiv.org/abs/2502.11089)

NSA reference implementation:
[Native Sparse Attention Triton](https://github.com/XunhaoLai/native-sparse-attention-triton)
