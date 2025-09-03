import argparse
import math
from functools import partial

import torch
import torch.nn as nn

from fsa.ops.FSA_topk_sparse_attention import (_topk_sparse_attention_bwd_opt,
                                               _topk_sparse_attention_fwd_opt)
from nsa_ref.ops import compressed_attention, linear_compress
from nsa_ref.ops.topk_sparse_attention import (_topk_sparse_attention_bwd,
                                               _topk_sparse_attention_fwd)
from nsa_ref.ops.utils import is_hopper_gpu
from utils import cuda_timer

IS_HOPPER_GPU = is_hopper_gpu()


def create_cu_seqlens(seqlen: int) -> torch.Tensor:
    """Create cumulative sequence lengths tensor for batch processing."""
    return torch.arange(0, 2 * seqlen, seqlen, dtype=torch.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark topk sparse attention kernels")

    parser.add_argument("--seqlen", type=int, default=65536, help="Sequence length")
    parser.add_argument("--num-q-heads", type=int, default=64, help="Number of query heads")
    parser.add_argument("--num-k-heads", type=int, default=64, help="Number of key/value heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")

    # NSA specific configuration
    parser.add_argument("--block-size", type=int, default=64, help="Block size for sparse attention")
    parser.add_argument("--topk", type=int, default=16, help="Top-k blocks for each query")

    # Benchmark configuration
    parser.add_argument("--warm-up", type=int, default=5, help="Number of warm-up runs")
    parser.add_argument("--benchmark-iters", type=int, default=30, help="Number of benchmark runs")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    seqlen = args.seqlen
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    head_dim = args.head_dim
    block_size = args.block_size
    topk = args.topk
    kernel_size = 32
    kernel_stride = 16

    # Benchmark parameters
    warm_up = args.warm_up
    benchmark_iters = args.benchmark_iters

    # Create test data
    torch.random.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    assert num_q_heads % num_k_heads == 0, "num_k_heads must be divisible by num_q_heads"
    num_share_q_heads = num_q_heads // num_k_heads

    # Generate random q, k, v tensors
    q = torch.randn(seqlen, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)

    # generate nsa parameters
    compress_key = torch.empty(num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    nn.init.xavier_uniform_(compress_key)

    compress_value = torch.empty(num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    nn.init.xavier_uniform_(compress_value)

    intra_block_pe = torch.empty(num_k_heads, kernel_size, head_dim, device=device, dtype=dtype)
    nn.init.xavier_uniform_(intra_block_pe)

    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(seqlen).to(device)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"cu_seqlens: {cu_seqlens}")
    print(f"block_size: {block_size}, topk: {topk}")

    # Compute topk_idx using compressed_attention
    print("Computing topk_idx using compressed_attention...")
    compressed_k, compressed_cu_seqlens = linear_compress(
        k,
        compress_key,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        intra_block_pe,
    )
    compressed_v, _ = linear_compress(
        v,
        compress_value,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        None,
    )

    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    sm_scale = 1 / math.sqrt(head_dim)

    _, topk_idx = compressed_attention(
        q=q,
        k=compressed_k,
        v=compressed_v,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=compressed_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=compressed_seqlens.max().item(),
        sm_scale=None,
        init_blocks=1,
        local_blocks=2,
        parallel_topk_compute=False,
    )

    H, N, TopK = topk_idx.shape
    num_blocks = topk_idx.max().item() + 1
    causal = (topk_idx == -1).sum().item() != 0

    print(f"topk_idx shape: {topk_idx.shape}, dtype: {topk_idx.dtype}")
    print(f"topk_idx range: [{topk_idx.min().item()}, {topk_idx.max().item()}]")
    print(f"H, N, TopK: {H}, {N}, {TopK}")
    print(f"num_blocks: {num_blocks}")
    print(f"causal: {causal}")

    # Create partial function for optimized version
    func_opt = partial(
        _topk_sparse_attention_fwd_opt,
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        seqlen,
        seqlen,
        sm_scale,
        causal=causal,
    )
    func_ref = partial(
        _topk_sparse_attention_fwd,
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        seqlen,
        seqlen,
        sm_scale,
    )

    # Warm up optimized version
    print("\nWarming up FSA optimized fwd kernel...")
    for i in range(warm_up):
        o_opt, lse_opt, _ = func_opt()

    # Benchmark optimized version
    print("Benchmarking FSA optimized fwd kernel...")
    with cuda_timer("topk_sparse_attention_fwd_opt") as timer:
        for i in range(benchmark_iters):
            o_opt, lse_opt, permute_results = func_opt()

    opt_time = timer.elapsed_time

    # Warm up reference version
    print("\nWarming up reference fwd kernel...")
    for i in range(warm_up):
        o_ref, lse_ref = func_ref()

    # Benchmark reference version
    print("Benchmarking reference fwd kernel...")
    with cuda_timer("topk_sparse_attention_fwd_ref") as timer_ref:
        for i in range(benchmark_iters):
            o_ref, lse_ref = func_ref()

    ref_time = timer_ref.elapsed_time

    dq = torch.zeros_like(q)
    dq_ref = torch.zeros_like(q)
    o = torch.randn_like(q)
    do = torch.randn_like(q)
    delta = torch.randn(num_q_heads, seqlen, device=device, dtype=dtype)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    func_opt_bwd = partial(
        _topk_sparse_attention_bwd_opt,
        o,
        do,
        lse_opt,
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        sm_scale,
        permute_results,
    )

    # Create partial function for backward_dq kernel
    func_ref_bwd = partial(
        _topk_sparse_attention_bwd,
        o,
        do,
        lse_ref,
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        sm_scale,
    )

    print("\nWarming up reference bwd kernel...")
    for i in range(warm_up):
        func_ref_bwd()

    with cuda_timer("topk_sparse_attention_bwd_ref") as timer_ref:
        for i in range(benchmark_iters):
            dq_ref, dk_ref, dv_ref = func_ref_bwd()

    bwd_ref_time = timer_ref.elapsed_time

    print("\nWarming up FSA optimized bwd kernel...")
    for i in range(warm_up):
        func_opt_bwd()

    print("Benchmarking FSA optimized bwd kernel...")
    with cuda_timer("topk_sparse_attention_bwd_opt") as timer:
        for i in range(benchmark_iters):
            dq, dk, dv = func_opt_bwd()

    bwd_opt_time = timer.elapsed_time

    total_diff = (o_ref - o_opt).abs().max()
    relative_diff = total_diff / o_ref.abs().max()
    lse_diff = (lse_ref - lse_opt).abs().max()

    dq_diff = (dq_ref - dq).abs().max()
    dq_relative_diff = (dq_ref - dq).abs().max() / dq_ref.abs().max()

    dk_diff = (dk_ref - dk).abs().max()
    dk_relative_diff = (dk_ref - dk).abs().max() / dk_ref.abs().max()

    dv_diff = (dv_ref - dv).abs().max()
    dv_relative_diff = (dv_ref - dv).abs().max() / dv_ref.abs().max()
    print(o_ref[0], o_opt[0])
    print(o_ref[1], o_opt[1])
    print(o_ref[2], o_opt[2])
    print(lse_ref[:, :512], lse_opt[:, :512])
    # Assert accuracy
    try:
        torch.testing.assert_close(o_ref, o_opt.to(torch.bfloat16), atol=1e-2, rtol=1e-2)
        print("\n✅ [Fwd] Output accuracy test PASSED (atol=1e-2, rtol=1e-2)")
    except AssertionError as e:
        print(f"\n❌ [Fwd] Output accuracy test FAILED: {e}")

    try:
        torch.testing.assert_close(lse_ref, lse_opt, atol=1e-5, rtol=1e-5)
        print("✅ [Fwd] LSE accuracy test PASSED (atol=1e-5, rtol=1e-5)")
    except AssertionError as e:
        print(f"❌ [Fwd] LSE accuracy test FAILED: {e}")

    try:
        torch.testing.assert_close(dq_ref, dq.to(torch.bfloat16), atol=9e-1, rtol=1e-3)
        torch.testing.assert_close(dk_ref, dk.to(torch.bfloat16), atol=9e-1, rtol=1e-3)
        torch.testing.assert_close(dv_ref, dv.to(torch.bfloat16), atol=9e-1, rtol=1e-3)
        print("✅ [Bwd] Output accuracy test PASSED (atol=9e-1, rtol=1e-3)")
    except AssertionError as e:
        print(f"❌ [Bwd] Output accuracy test FAILED: {e}")

    avg_opt_time = opt_time / benchmark_iters
    avg_ref_time = ref_time / benchmark_iters
    avg_opt_time_bwd = bwd_opt_time / benchmark_iters
    avg_ref_time_bwd = bwd_ref_time / benchmark_iters

    # Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print("Configuration:")
    print(f"  - Sequence length: {seqlen}")
    print(f"  - Num q heads: {num_q_heads}")
    print(f"  - Num kv heads: {num_k_heads}")
    print(f"  - Head dim: {head_dim}")
    print(f"  - Block size: {block_size}")
    print(f"  - TopK: {topk}")
    print(f"  - Kernel size: {kernel_size}")
    print(f"  - Kernel stride: {kernel_stride}")
    print(f"  - Causal: {causal}")
    print("\nAccuracy:")
    print(f"  - [Fwd] Output diff: {total_diff:.2e} (relative: {relative_diff:.2e})")
    print(f"  - [Fwd] LSE diff: {lse_diff:.2e}")
    print(f"  - [Bwd] dQ diff: {dq_diff:.2e} (relative: {dq_relative_diff:.2e})")
    print(f"  - [Bwd] dK diff: {dk_diff:.2e} (relative: {dk_relative_diff:.2e})")
    print(f"  - [Bwd] dV diff: {dv_diff:.2e} (relative: {dv_relative_diff:.2e})")

    print("\nPerformance:")
    print(f"  - [Fwd] Reference: {avg_ref_time:.3f} ms")
    print(f"  - [Fwd] Optimized: {avg_opt_time:.3f} ms")
    print(f"  - [Fwd] Ratio (ref/opt): {(avg_ref_time / avg_opt_time):.3f}x")
    print(f"  - [Bwd] Reference: {avg_ref_time_bwd:.3f} ms")
    print(f"  - [Bwd] Optimized: {avg_opt_time_bwd:.3f} ms")
    print(f"  - [Bwd] Ratio (ref/opt): {(avg_ref_time_bwd / avg_opt_time_bwd):.3f}x")

    print("\n🎉 Benchmark completed successfully!")
