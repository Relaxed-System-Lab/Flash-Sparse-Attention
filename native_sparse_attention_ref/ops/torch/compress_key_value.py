# Copyright 2025 Xunhao Lai.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple

import torch
from einops import einsum, rearrange


def conv_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress key and value tensor with kernel_size and kernel_stride.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight of conv1d, shape (num_heads * head_dim, head_dim, kernel_size)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert num_heads * head_dim == w.shape[0]
    assert w.shape[1] == head_dim
    assert w.shape[2] == kernel_size
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}

    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n d -> b d n")
    # conv1d
    y = torch.nn.functional.conv1d(x, w, stride=kernel_stride, groups=num_heads)
    y = rearrange(y, "b (h d) n -> b n h d", h=num_heads)
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)

    # position embedding as a bias
    if pe is not None:
        bias = torch.nn.functional.conv1d(
            rearrange(pe, "h n d -> (h d) n"),
            w,
            stride=kernel_stride,
            groups=num_heads,
        )
        bias = rearrange(bias, "(h d) 1 -> 1 h d", h=num_heads)
        y = y + bias
    return y, y_cu_seqlens


def avgpool_compress_torch(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w is None, "don't need additional weight for avgpool"
    assert pe is None, "don't need additional positional embedding for avgpool"
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}

    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n d -> b d n")
    # avgpool
    y = torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=kernel_stride, ceil_mode=True)
    y = rearrange(y, "b (h d) n -> b n h d", h=num_heads)
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def weightedpool_compress_torch(
    x: torch.Tensor,
    w: torch.Tensor,  # [num_heads, kernel_size]
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    """Compress key and value tensor with kernel_size and kernel_stride.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, shape (num_heads, kernel_size)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32
    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w.shape[0] == num_heads
    assert w.shape[1] == kernel_size
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n (h d) -> b h n d", h=num_heads)
    x = x.as_strided(
        size=(batch_size, num_heads, y_seqlens.max().item(), kernel_size, head_dim),
        stride=(
            x.stride(0),
            x.stride(1),
            kernel_stride * x.stride(2),
            x.stride(2),
            x.stride(3),
        ),
    )
    y = einsum(x, w, "b h n k d, h k -> b n h d")
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def linear_compress_torch(
    x: torch.Tensor,
    w: torch.Tensor,  # [num_heads, kernel_size * head_dim, head_dim]
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    """Compress key and value tensor with kernel_size and kernel_stride. Similar to conv_compress.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32
    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w.shape[0] == num_heads
    assert w.shape[1] == kernel_size * head_dim
    assert w.shape[2] == head_dim
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n (h d) -> b h n d", h=num_heads)
    x = x.as_strided(
        size=(batch_size, num_heads, y_seqlens.max().item(), kernel_size, head_dim),
        stride=(
            x.stride(0),
            x.stride(1),
            kernel_stride * x.stride(2),
            x.stride(2),
            x.stride(3),
        ),
    )
    y = einsum(
        x,
        rearrange(w, "h (k d) D -> h k d D", k=kernel_size),
        "b h n k d, h k d D -> b n h D",
    )
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def maxpool_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w is None, "don't need additional weight for avgpool"
    assert pe is None, "don't need additional positional embedding for avgpool"
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}

    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n d -> b d n")
    # maxpool
    y = torch.nn.functional.max_pool1d(x, kernel_size=kernel_size, stride=kernel_stride, ceil_mode=True)
    y = rearrange(y, "b (h d) n -> b n h d", h=num_heads)
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def minpool_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w is None, "don't need additional weight for avgpool"
    assert pe is None, "don't need additional positional embedding for avgpool"
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}

    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n d -> b d n")
    # - maxpool as minpool
    y = -torch.nn.functional.max_pool1d(-x, kernel_size=kernel_size, stride=kernel_stride, ceil_mode=True)
    y = rearrange(y, "b (h d) n -> b n h d", h=num_heads)
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def maxminavgpool_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert cu_seqlens.dtype == torch.int32

    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w is None, "don't need additional weight for avgpool"
    assert pe is None, "don't need additional positional embedding for avgpool"
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}

    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n d -> b d n")
    # maxpool
    y_max = torch.nn.functional.max_pool1d(x, kernel_size=kernel_size, stride=kernel_stride, ceil_mode=True)
    y_max = rearrange(y_max, "b (h d) n -> b n h d", h=num_heads)
    # minpool
    y_min = -torch.nn.functional.max_pool1d(-x, kernel_size=kernel_size, stride=kernel_stride, ceil_mode=True)
    y_min = rearrange(y_min, "b (h d) n -> b n h d", h=num_heads)
    # avgpool
    y_avg = torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=kernel_stride, ceil_mode=True)
    y_avg = rearrange(y_avg, "b (h d) n -> b n h d", h=num_heads)
    # concat
    y = torch.cat([y_max, y_min, y_avg], dim=-1)
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def lowrank_compress(
    x: torch.Tensor,
    w: List[torch.Tensor],  # [num_heads, kernel_size * head_dim, head_dim]
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    """Compress key and value tensor with kernel_size and kernel_stride.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, w1 shape (num_heads, head_dim, lora_dim), w2 shape (num_heads, kernle_size * lora_dim, head_dim)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w[0].dtype and x.dtype == w[1].dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32
    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w[0].shape[0] == num_heads
    assert w[0].shape[1] == head_dim
    lora_dim = w[0].shape[2]
    assert w[1].shape[0] == num_heads
    assert w[1].shape[1] == kernel_size * lora_dim
    assert w[1].shape[2] == head_dim
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    # proj to low rank
    x = einsum(x, w[0], "n h d, h d r -> n h r")
    # pad and rearrange x
    x = rearrange(x, "n h r -> n (h r)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n (h r) -> b h n r", h=num_heads)
    x = x.as_strided(
        size=(batch_size, num_heads, y_seqlens.max().item(), kernel_size, lora_dim),
        stride=(
            x.stride(0),
            x.stride(1),
            kernel_stride * x.stride(2),
            x.stride(2),
            x.stride(3),
        ),
    )
    y = einsum(
        x,
        rearrange(w[1], "h (k r) d -> h k r d", k=kernel_size),
        "b h n k r, h k r d -> b n h d",
    )
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def swiglu_compress(
    x: torch.Tensor,
    w: List[torch.Tensor],  # [num_heads, kernel_size * head_dim, head_dim]
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    """Compress key and value tensor with kernel_size and kernel_stride.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, w1 shape (num_heads, head_dim, lora_dim * 2), w2 shape (num_heads, kernle_size * lora_dim, head_dim)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w[0].dtype and x.dtype == w[1].dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32
    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w[0].shape[0] == num_heads
    assert w[0].shape[1] == head_dim
    lora_dim = w[0].shape[2] // 2
    assert w[0].shape[2] == 2 * lora_dim
    assert w[1].shape[0] == num_heads
    assert w[1].shape[1] == kernel_size * lora_dim
    assert w[1].shape[2] == head_dim
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    # proj to low rank
    x = einsum(x, w[0], "n h d, h d R -> n h R")
    x, g = torch.split(x, [lora_dim, lora_dim], dim=-1)
    x = x * torch.nn.functional.silu(g)
    # pad and rearrange x
    x = rearrange(x, "n h r -> n (h r)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n (h r) -> b h n r", h=num_heads)
    x = x.as_strided(
        size=(batch_size, num_heads, y_seqlens.max().item(), kernel_size, lora_dim),
        stride=(
            x.stride(0),
            x.stride(1),
            kernel_stride * x.stride(2),
            x.stride(2),
            x.stride(3),
        ),
    )
    y = einsum(
        x,
        rearrange(w[1], "h (k r) d -> h k r d", k=kernel_size),
        "b h n k r, h k r d -> b n h d",
    )
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def softmaxpool_compress_torch(
    x: torch.Tensor,
    w: torch.Tensor,  # [num_heads, kernel_size]
    cu_seqlens,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    """Compress key and value tensor with kernel_size and kernel_stride.

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, shape (num_heads, kernel_size)
        cu_seqlens (_type_): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): kernel_stride for conv1d
        pe (Optional[torch.Tensor], optional): intra-block positional embedding with shape (num_heads, kernel_size, head_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32
    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w.shape[0] == num_heads
    assert w.shape[1] == kernel_size
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n (h d) -> b h n d", h=num_heads)
    x = x.as_strided(
        size=(batch_size, num_heads, y_seqlens.max().item(), kernel_size, head_dim),
        stride=(
            x.stride(0),
            x.stride(1),
            kernel_stride * x.stride(2),
            x.stride(2),
            x.stride(3),
        ),
    )
    w = w.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
    y = einsum(x, w, "b h n k d, h k -> b n h d")
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens


def adaptivepool_compress(
    x: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    pe: Optional[torch.Tensor] = None,
):
    """Compress key and value tensor with kernel_size and kernel_stride.
        input: [k, d], pe: [k, r], weight: [d+r, 1]
        (1) cat([k, d], [k, r]) @ [d+r, 1] -> [k, 1]
        (2) softmax([k, 1]) -> [k, 1] -> [1, k]
        (3) [1, k] @ [k, d] -> [1, d]

    Args:
        x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
        w (torch.Tensor): weight for each head, shape (num_heads, head_dim + pe_dim)
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
        kernel_stride (int): stride for each compress kernel
        pe (torch.Tensor, optional): intra-block positional embedding with shape (num_heads, kernel_size, pe_dim). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
    """
    # dtype check
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert x.dtype == pe.dtype if pe is not None else True
    assert cu_seqlens.dtype == torch.int32
    # shape check
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    if pe is not None:
        assert pe.shape[0] == num_heads and pe.shape[1] == kernel_size
        pe_dim = pe.shape[-1]
    else:
        pe_dim = 0
    assert w.shape[0] == num_heads
    assert w.shape[1] == head_dim + pe_dim
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(y_seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    # pad and rearrange x
    x = rearrange(x, "n h d -> n (h d)")
    splited_x = torch.split(x, seqlens.tolist(), 0)
    x = torch.nn.utils.rnn.pad_sequence(splited_x, batch_first=True)
    x = rearrange(x, "b n (h d) -> b h n d", h=num_heads)
    # get softmax weight
    x = x.as_strided(
        size=(batch_size, num_heads, y_seqlens.max().item(), kernel_size, head_dim),
        stride=(
            x.stride(0),
            x.stride(1),
            kernel_stride * x.stride(2),
            x.stride(2),
            x.stride(3),
        ),
    )
    ori_x = x
    if pe is not None:
        x = torch.cat([x, pe[None, :, None, :].expand(x.shape[0], -1, x.shape[2], -1, -1)], dim=-1)
    x = einsum(x, w, "b h n k d, h d -> b h n k")
    x = torch.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype)
    # weighted pool
    y = einsum(x, ori_x, "b h n k, b h n k d -> b n h d")
    # only keep useful part
    y = torch.cat([y[i, : y_seqlens[i]] for i in range(batch_size)], dim=0)
    return y, y_cu_seqlens
