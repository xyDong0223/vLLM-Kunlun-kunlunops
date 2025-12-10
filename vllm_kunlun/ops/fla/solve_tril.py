# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
from typing import Optional

import torch
import os
from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .utils import input_guard

base_dir = os.path.dirname(__file__)

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=["BT"],
# )
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
    )
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)

    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_modified(
    i_t,
    i_bh,
    i_n,
    bos,
    i_b,
    i_h,
    subA,
    subAd,
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,                   # 32
    H: tl.constexpr,     # 4
    BT: tl.constexpr,    # 64
    IS_VARLEN: tl.constexpr,
):
    A = A + (bos * H + i_h) * BT
    print("for A Base offset ", (bos * H + i_h) * BT)

    offset = (i_t * 16) % BT

    range16 = tl.arange(0, 16)
    newp_A = subA + range16[:, None] * 16 + range16[None, :]
    b_A = tl.load(newp_A).to(tl.float32)

    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T - i_t * 16)):
        print("[naive impl-0]loopIdx:", i)
        # print("for A start   (i_t * 16 + i) * H * BT", (i_t * 16 + i) * H * BT)
        # print("for A start   offset", offset)
        # print("for A start", (i_t * 16 + i) * H * BT + offset)
        print("[naive impl-1]b_A value in now loopIdx:", b_A)
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        # print("[naive impl-2]b_a value in now loopIdx:", b_a)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        print("[naive impl-2-1]b_a value after reduce in now loopIdx:", b_a)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
        print("[naive impl-2-2]b_A value after oimask in now loopIdx:", b_A)
        # print("[naive impl-3]b_A result in now loopIdx:", b_A)
    # print(f"[naive impl-4] b_A value after allLoop = {b_A}")
    b_A += o_i[:, None] == o_i[None, :]
    # print(f"[naive impl-5] b_A value after mask = {b_A}")

    newp_Ad = subAd + range16[:, None] * 16 + range16[None, :]
    tl.store(
        newp_Ad,
        b_A.to(subAd.dtype.element_ty, fp_downcast_rounding="rtne"),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=["BT"],
# )
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_modified_in_Loop(
    i_t,
    i_bh,
    i_n,
    bos,
    i_b,
    i_h,
    subA,
    subAd,
    AInLoop,
    ba_reduce,
    loopIdx,
    reduce_res,
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,                   # 32
    H: tl.constexpr,     # 4
    BT: tl.constexpr,    # 64
    IS_VARLEN: tl.constexpr,
):
    range16 = tl.arange(0, 16)
    newp_A = subA + range16[:, None] * 16 + range16[None, :]
    b_A = tl.load(newp_A).to(tl.float32)
    # print("[loop impl-0]loopIdx:", loopIdx)
    # print("[loop impl-1]b_A value in now loopIdx:", b_A)

    o_i = tl.arange(0, 16)
    i=loopIdx
    b_a = -tl.load(AInLoop + o_i)
    # print("[loop impl-2]b_a value in now loopIdx:", b_a)
    red_res = b_a[:, None] * b_A
    # print("[Triton]red_res=", red_res)
    tl.store(reduce_res + range16[:, None] * 16 + range16[None, :], red_res)
    # b_a = b_a + tl.sum(b_a[:, None] * b_A, 1) # TODO: revert to 0
    # # print("triton reduce b_a", b_a)
    # tl.store(ba_reduce + o_i, b_a)

    # mask = o_i == i
    # # print("mask", mask[:, None])
    # # print("b_a", b_a)
    # # print("b_A", b_A)
    # print("before b_A", b_A)
    # b_A = tl.where(mask[:, None], b_a, b_A)
    # print("[loop impl-3]b_A result in now loopIdx:", b_A)

    # tl.store(newp_A, b_A)


def solve_tril_16x16_kernel_new(
    NT,
    B,
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H,
    BT,
    IS_VARLEN,
):
    Ad_modify = Ad
    for loopX in range(NT):
        # i_n, i_t = tl.load(chunk_indices ...
        chunk_indices_load_offset_1 = loopX * 2
        row_idx = chunk_indices_load_offset_1 // chunk_indices.shape[1]
        col_idx = chunk_indices_load_offset_1 % chunk_indices.shape[1]
        i_n = int(chunk_indices[row_idx, col_idx])
        chunk_indices_load_offset_2 = loopX * 2 + 1
        row_idx = chunk_indices_load_offset_2 // chunk_indices.shape[1]
        col_idx = chunk_indices_load_offset_2 % chunk_indices.shape[1]
        i_t = int(chunk_indices[row_idx, col_idx])

        # bos, eos = tl.load(cu_seqlens ...
        cu_seqlens_load_offset_1 = i_n
        bos = int(cu_seqlens[cu_seqlens_load_offset_1])
        cu_seqlens_load_offset_2 = i_n + 1
        eos = int(cu_seqlens[cu_seqlens_load_offset_2])
        T = eos - bos

        for loopY in range(B * H):
            i_b = loopY // H
            i_h = loopY % H

            # get subA
            if (bos * H + i_h) < H:
                Tstart = loopX * 16 % BT
                Tend = Tstart + 16
                BTstart = loopX * 16 % BT
                BTend = BTstart + 16
                subA = A[0, Tstart:Tend, loopY, BTstart:BTend].contiguous().clone()
                # print(f"subA slice A dim[0, {Tstart}:{Tend}, {loopY}, {BTstart}:{BTend}]")
                if (Tend > T): # bondary check
                    subA[T-16:, :] = 0

                # subA.shape torch.Size([9, 16])
                # vvv
                # subA.shape torch.Size([16, 16]) 用0补齐
                if subA.shape[0] < 16:
                    pad_rows = 16 - subA.shape[0]
                    zeros = torch.zeros((pad_rows, subA.shape[1]), dtype=subA.dtype, device=subA.device)
                    subA = torch.cat([subA, zeros], dim=0)
            else:
                assert(0) & "need deal this situation"

            # get subAd
            if (bos * H + i_h) < H:
                Tstart = loopX * 16
                Tend = Tstart + 16
                BTstart = 0 * 16
                BTend = BTstart + 16
                subAd = Ad_modify[0, Tstart:Tend, loopY, BTstart:BTend].contiguous().clone()
                # print(f'T={T}, Tstart={Tstart}, Tend={Tend}, BTstart={BTstart}, BTend={BTend}')
            else:
                assert(0) & "need deal this situation"

            mask = (torch.arange(16, device=subA.device)[:, None] > torch.arange(16, device=subA.device)[None, :])
            subA = -torch.where(mask, subA, torch.zeros_like(subA))

            for inLoopIdx in range(1, min(16, T - i_t * 16)):
                # print(f"loopX={loopX}, loopY={loopY}, inLoopIdx={inLoopIdx}")
                offsetStart=loopX*16 % BT
                offsetEnd=offsetStart+16
                
                AInLoop = A[0, (loopX * 16 + inLoopIdx), loopY, offsetStart:offsetEnd]
                # print(f"AInLoop slice A dim[0, {(loopX * 16 + inLoopIdx)}, {loopY}, {offsetStart}:{offsetEnd}")

                ba_reduce = torch.empty_like(AInLoop)
                reduce_res = torch.empty_like(subA)
                solve_tril_16x16_kernel_modified_in_Loop[1, 1](
                    i_t,
                    loopY,
                    i_n,
                    bos,
                    i_b,
                    i_h,
                    subA=subA,
                    subAd=subAd,
                    AInLoop=AInLoop,
                    ba_reduce=ba_reduce,
                    loopIdx=inLoopIdx,
                    reduce_res=reduce_res,
                    A=A,
                    Ad=Ad_modify,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    T=T,
                    H=H,
                    BT=BT,
                    num_warps=1,
                    num_stages=4,
                )
                AInLoop = AInLoop.flatten()
                b_A = subA # [16x16]
                b_a = -AInLoop[0:16] # [16]
                b_a = b_a + torch.sum(reduce_res, 0)
                ba_reduce = b_a
                o_i = torch.arange(16, device=ba_reduce.device)
                mask = (o_i == inLoopIdx)
                mask_expand = mask[:, None]
                subA = torch.where(mask_expand, ba_reduce, subA)

            subAd = subA + (torch.arange(16, device=subA.device)[:, None] == torch.arange(16, device=subA.device)[None, :])

            # deal store mask
            Tstart = loopX * 16
            Tend = Tstart + 16
            BTstart = 0 * 16
            BTend = BTstart + 16
            # print(f"slice Ad_modify dim[0, {Tend-needMaskRow}:{Tend}, {loopY}, {BTstart}:{BTend}]")
            if (Tend > T): # bondary mask
                needMaskRow = Tend - T
                Ad_modify[0, Tstart:Tend, loopY, BTstart:BTend] = subAd[:T-Tstart, :]
            else: 
                # assert (Ad_modify[0, Tstart:Tend, loopY, BTstart:BTend].shape == subAd.shape)
                Ad_modify[0, Tstart:Tend, loopY, BTstart:BTend] = subAd

    # if BT == 16:
    #     return Ad

    return Ad_modify

# @input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the lower triangular matrix
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, K]
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    # cnt = 0
    # for b in range(B):
    #     for t in range(T):
    #         for h in range(H):
    #             for d in range(BT):
    #                 A[b, t, h, d] = cnt
    #                 cnt += 1

    Ad = -999 * torch.ones(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )
    # cnt = 0
    # for b in range(B):
    #     for t in range(T):
    #         for h in range(H):
    #             for d in range(16):
    #                 Ad[b, t, h, d] = cnt
    #                 cnt += 1

    Ad_modify = Ad.clone()

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 16)

    import os
    if os.getenv("TRITON_INTERPRET", None) == "1":
        solve_tril_16x16_kernel[NT, B * H](
            A=A,
            Ad=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            num_warps=1,
            num_stages=4,
        )
        return Ad

    Ad_modify = solve_tril_16x16_kernel_new(
        NT,
        B,
        A=A,
        Ad=Ad_modify,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        IS_VARLEN= True if cu_seqlens is not None else False,
        # num_warps=1,
        # num_stages=4,
    ).to(A.dtype)
    return Ad_modify