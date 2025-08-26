import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import einsum, rearrange

from nsa_ref.ops.utils import is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


def _linear_compress_decode(
    new_tokens,  # New tokens to append [1, num_heads, head_dim]
    compress_weight,  # [num_heads, head_dim * kernel_size, head_dim]
    kernel_size,
    kernel_stride,
    intra_block_pe=None,  # [num_heads, kernel_size, head_dim] or None
    prev_total_len=0,  # Previous total sequence length
    token_buffer=None,  # Buffer containing recent tokens
):
    """
    Decode version that properly handles absolute positions and windowing.
    """
    device = new_tokens.device

    # Update token buffer with new tokens
    if token_buffer is None:
        all_tokens = new_tokens
        buffer_start_pos = prev_total_len
    else:
        all_tokens = torch.cat([token_buffer, new_tokens], dim=0)
        buffer_start_pos = prev_total_len - token_buffer.shape[0]

    new_total_len = prev_total_len + 1

    # Calculate which output positions we had before and what we should have now
    prev_max_output_idx = math.floor((prev_total_len - kernel_size) / kernel_stride) if prev_total_len >= kernel_size else -1
    new_max_output_idx = math.floor((new_total_len - kernel_size) / kernel_stride) if new_total_len >= kernel_size else -1

    if new_max_output_idx <= prev_max_output_idx:
        # No new outputs to compute
        return None

    # Determine the input window
    windows_to_compute = []
    window_start_abs = new_max_output_idx * kernel_stride  # Absolute position in full sequence
    window_end_abs = window_start_abs + kernel_size

    # Convert to relative position in our buffer
    window_start_rel = window_start_abs - buffer_start_pos
    window_end_rel = window_end_abs - buffer_start_pos

    # Check if we have all tokens for this window
    if window_start_rel >= 0 and window_end_rel <= all_tokens.shape[0]:
        window_tokens = all_tokens[window_start_rel:window_end_rel]
        windows_to_compute.append(window_tokens)

    if not windows_to_compute:
        return None
    # Create cu_seqlens for the stacked windows
    cu_seqlens = torch.tensor([0, kernel_size], dtype=torch.int32, device=device)
    y_cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    # Compute compressed representation
    compressed_output = linear_compress_with_pe(
        window_tokens,
        compress_weight,
        cu_seqlens,
        kernel_size,
        kernel_size,  # stride = kernel_size for non-overlapping windows
        y_cu_seqlens,
        intra_block_pe,
    )

    return compressed_output


class LinearCompressDecode(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
        y_cu_seqlens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress key and value tensor with kernel_size and kernel_stride. Similar to conv_compress.

        Args:
            x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
            w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
            cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen
            kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
            kernel_stride (int): stride for each compress kernel

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
        """
        # dtype check
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        assert x.dtype == w.dtype, f"x dtype: {x.dtype}, w dtype: {w.dtype}"
        assert cu_seqlens.dtype == torch.int32

        # shape check
        total_len, num_heads, head_dim = x.shape
        batch_size = cu_seqlens.shape[0] - 1
        assert w.shape[0] == num_heads
        assert w.shape[1] == kernel_size * head_dim
        assert w.shape[2] == head_dim
        assert kernel_size % kernel_stride == 0
        assert kernel_size in {16, 32, 64, 128}
        assert head_dim % 8 == 0

        # compute seqlens after compression
        y = torch.zeros(y_cu_seqlens[-1], num_heads, head_dim, dtype=x.dtype, device=x.device)

        block_kernel_size = max(16, triton.next_power_of_2(kernel_size))
        block_head_dim = 8 if IS_HOPPER_GPU else 4
        block_headD_dim = 16
        block_output_seq_size = 64
        w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()

        grid = lambda META: (
            batch_size * num_heads,
            1,
            triton.cdiv(head_dim, META["BLOCK_HEADD_DIM"]),
        )

        linear_compress_fwd_kernel[grid](
            x,
            y,
            w,
            cu_seqlens,
            y_cu_seqlens,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            num_heads,
            kernel_size,
            kernel_stride,
            head_dim,
            head_dim,
            block_output_seq_size,
            block_kernel_size,
            block_head_dim,
            block_headD_dim,
            num_warps=4,
            num_stages=3,
        )
        return y, y_cu_seqlens


@triton.jit
def linear_compress_fwd_kernel(
    X,  # input pointer [total_len, num_heads, head_dim]
    Y,  # output pointer [total_compressed_len, num_heads, head_dim]
    W,  # weight matrix pointer [num_heads, kernel_size, head_dim, head_dim]
    cu_seqlens_x,  # cumulative sequence lengths before compression
    cu_seqlens_y,  # cumulative sequence lengths after compression
    stride_xn,  # stride for X's sequence dimension
    stride_xh,  # stride for X's num head dimension
    stride_xd,  # stride for X's head_dim dimension
    stride_wh,  # stride for W's num head dimension
    stride_wk,  # stride for W's kernel size  dimension
    stride_wd,  # stride for W's initial head dim dimension
    stride_wD,  # stride for W's final head dim dimension
    stride_yn,  # stride for Y's sequence dimension
    stride_yh,  # stride for Y's num head dimension
    stride_yd,  # stride for Y's head_dim dimension
    NUM_HEADS: tl.constexpr,  # total num heads
    KERNEL_SIZE: tl.constexpr,  # kernel size when calculate the output
    KERNEL_STRIDE: tl.constexpr,  # kernel stride when calculate the output
    HEADd_DIM: tl.constexpr,  # initial head dimension size
    HEADD_DIM: tl.constexpr,  # final head dimension size
    BLOCK_OUTPUT_SEQ_SIZE: tl.constexpr,  # Loaded output len
    BLOCK_KERNEL_SIZE: tl.constexpr,  # Loaded kernel size when calculate the output
    BLOCK_HEADd_DIM: tl.constexpr,  # Loaded  orignal head dimension size
    BLOCK_HEADD_DIM: tl.constexpr,  # loaded final head dimension size
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    pid_k = tl.program_id(1)
    pid_D = tl.program_id(2)

    x_start = tl.load(cu_seqlens_x + pid_b)
    x_end = tl.load(cu_seqlens_x + pid_b + 1)
    x_len = x_end - x_start

    y_start = tl.load(cu_seqlens_y + pid_b)
    y_end = tl.load(cu_seqlens_y + pid_b + 1)
    y_len = y_end - y_start
    if pid_k * BLOCK_OUTPUT_SEQ_SIZE >= y_len:
        return

    off_kernel_size = tl.arange(0, BLOCK_KERNEL_SIZE)
    off_d = tl.arange(0, BLOCK_HEADd_DIM)
    off_output_seq_size = tl.arange(0, BLOCK_OUTPUT_SEQ_SIZE)

    x_base_ptrs = (
        X
        + pid_h * stride_xh
        + x_start * stride_xn
        + (
            (pid_k * BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + off_output_seq_size * KERNEL_STRIDE)[:, None]
            + off_kernel_size[None, :]
        )[:, :, None]
        * stride_xn
        + off_d[None, None, :] * stride_xd
    )
    x_base_mask = (
        (
            (pid_k * BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + off_output_seq_size * KERNEL_STRIDE)[:, None]
            + off_kernel_size[None, :]
        )
        < x_len
    )[:, :, None]

    w_ptrs = tl.make_block_ptr(
        base=W + pid_h * stride_wh,
        shape=(KERNEL_SIZE, HEADd_DIM, HEADD_DIM),
        strides=(stride_wk, stride_wd, stride_wD),
        offsets=(0, 0, pid_D * BLOCK_HEADD_DIM),
        block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM, BLOCK_HEADD_DIM),
        order=(2, 1, 0),
    )

    y_ptrs = tl.make_block_ptr(
        base=Y + y_start * stride_yn + pid_h * stride_yh,
        shape=(y_len, HEADD_DIM),
        strides=(stride_yn, stride_yd),
        offsets=(pid_k * BLOCK_OUTPUT_SEQ_SIZE, pid_D * BLOCK_HEADD_DIM),
        block_shape=(BLOCK_OUTPUT_SEQ_SIZE, BLOCK_HEADD_DIM),
        order=(1, 0),
    )

    y_d = tl.full((BLOCK_OUTPUT_SEQ_SIZE, BLOCK_HEADD_DIM), 0, dtype=tl.float32)

    for i in range(0, HEADd_DIM, BLOCK_HEADd_DIM):

        x_ptrs = x_base_ptrs + i * stride_xd
        x_mask = x_base_mask & ((i + off_d) < HEADd_DIM)[None, None, :]

        x = tl.load(x_ptrs, mask=x_mask, other=0)
        x = tl.reshape(x, (BLOCK_OUTPUT_SEQ_SIZE, BLOCK_KERNEL_SIZE * BLOCK_HEADd_DIM))
        # x : [n, k * bd]

        w = tl.load(w_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        w = tl.reshape(w, (BLOCK_KERNEL_SIZE * BLOCK_HEADd_DIM, BLOCK_HEADD_DIM))
        # w: [k * bd, D]

        y_d += tl.dot(x, w)
        # y_d : [n, D]

        w_ptrs = tl.advance(w_ptrs, (0, BLOCK_HEADd_DIM, 0))

    tl.store(y_ptrs, y_d.to(y_ptrs.dtype.element_ty), boundary_check=(0, 1))


def linear_cmp_func(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    y_cu_seqlens,
) -> torch.Tensor:
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype, f"x dtype: {x.dtype}, w dtype: {w.dtype}"
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w.shape[0] == num_heads
    assert w.shape[1] == kernel_size * head_dim
    assert w.shape[2] == head_dim
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    assert head_dim % 8 == 0

    # compute seqlens after compression
    y = torch.zeros(y_cu_seqlens[-1], num_heads, head_dim, dtype=x.dtype, device=x.device)

    block_kernel_size = max(16, triton.next_power_of_2(kernel_size))
    block_head_dim = 8 if IS_HOPPER_GPU else 4
    block_headD_dim = 16
    block_output_seq_size = 64
    w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()

    grid = lambda META: (
        batch_size * num_heads,
        1,
        triton.cdiv(head_dim, META["BLOCK_HEADD_DIM"]),
    )

    linear_compress_fwd_kernel[grid](
        x,
        y,
        w,
        cu_seqlens,
        y_cu_seqlens,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        num_heads,
        kernel_size,
        kernel_stride,
        head_dim,
        head_dim,
        block_output_seq_size,
        block_kernel_size,
        block_head_dim,
        block_headD_dim,
        num_warps=4,
        num_stages=3,
    )
    return y


def linear_compress_with_pe(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    y_cu_seqlens,
    pe: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress key and value tensor with kernel_size and kernel_stride with linear projection.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): stride for each compress kernel
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """

    y = linear_cmp_func(x, w, cu_seqlens, kernel_size, kernel_stride, y_cu_seqlens)
    # position embedding as a bias
    if pe is not None:
        assert pe.dtype == x.dtype and pe.device == x.device
        pe = rearrange(pe, "h k d -> h (k d)")
        bias = einsum(pe, w, "h D, h D d -> h d")
        y = y + bias.unsqueeze(0)
    return y
