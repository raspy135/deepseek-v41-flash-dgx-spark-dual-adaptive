"""Lossless packed BF16 head projection with explicit native MMA geometry.

Byte-derived TL dot operands selected kWidth=4, changing summation grouping:
the tiny [2**25, 1, -2**25, 1, 0...] cancellation screen returned zero where
native kWidth=2 and guarded cuBLAS returned 640. Explicit Gluon operand layouts
keep kWidth=2 without a global BF16 scratch round trip. engine/native_head.py
uses project() on prepared tiled storage. project_native() is a benchmark-only
adapter that retains the original weight and must not be used for serving.

2026-10-07 bounded cold local TP2 head screen, M4/N64640/K5120, two seeded
BF16 activation cases (scales1/4): guarded production3.079ms, native TC2.909ms,
tiled Gluon BN32/64=2.278/2.342ms. Both packed plans matched every tested logit
and graph replay; all stored checkpoint bits were checked. BN64 did not improve
the screen, so start with BN32. Stored bytes661,913,600->508,819,072 per rank.
This kernel result excludes vocabulary gathering and full decode acceptance;
it does not establish universal cuBLAS accumulation equivalence.

Subsequent widths1–6 checked 5.43M logits and graphs exactly; a frozen-map TP2
code/prose ABBA found unchanged tokens/acceptance and ~2% faster full rounds.
Production remains opt-in. See RESULTS.md and tools/test_native_head.py.
"""
import torch
import triton
from triton.experimental import gluon as g

import decode_rows
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from head_packed import PackedHead
from head_packed_tiles import PackedHead as TiledPackedHead


_packed = {}


def clear_packed_cache():
    _packed.clear()


@g.jit
def _project(X, LOW, DELTA, HEADER, ESCAPE, OUT,
             M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
             XM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
             WARPS: gl.constexpr, RAW: gl.constexpr, TILED: gl.constexpr,
             BM: gl.constexpr = 16):
    xl: gl.constexpr = BlockedLayout([1, 8], [2, 16], [WARPS, 1], [1, 0])
    wl: gl.constexpr = BlockedLayout([8, 1], [16, 2], [1, WARPS], [0, 1])
    ol: gl.constexpr = BlockedLayout([1, 4], [2, 16], [WARPS, 1], [1, 0])
    ml: gl.constexpr = NVMMADistributedLayout([2, 0], [1, WARPS], [16, 8])
    al: gl.constexpr = DotOperandLayout(0, ml, 2)
    bl: gl.constexpr = DotOperandLayout(1, ml, 2)
    ns = gl.program_id(0)*BN + gl.arange(0, BN, layout=SliceLayout(0, wl))
    ks = gl.arange(0, BK, layout=SliceLayout(1, wl))
    # BM row tiles of the same m16n8k16 MMA: each logit is still one chain over K in BK steps,
    # so a row's value does not depend on how many rows share the call (tools/test_decode_rows.py).
    ms_x = gl.arange(0, BM, layout=SliceLayout(1, xl))
    ks_x = gl.arange(0, BK, layout=SliceLayout(0, xl))
    acc = gl.full((BM, BN), 0, gl.float32, ml)
    for start in range(0, K, BK):
        x = gl.load(X + ms_x[:, None]*XM + start + ks_x[None, :],
                    ms_x[:, None] < M, 0)
        # Tail rows must remain defined while every lane takes the same MMA.
        if TILED:
            group = (start//128)*N + ns
            lo_offset = group[None, :]*128 + ks[:, None]
            delta_offset = group[None, :]*64 + ks[:, None]//2
            header_offset = group
        else:
            lo_offset = ns[None, :]*K + start + ks[:, None]
            delta_offset = ns[None, :]*(K//2) + (start+ks[:, None])//2
            header_offset = ns*(K//128) + start//128
        low = gl.load(LOW + lo_offset,
                      ns[None, :] < N, 0).to(gl.uint16)
        delta = gl.load(DELTA + delta_offset,
                        ns[None, :] < N, 0).to(gl.uint16)
        delta = (delta >> ((ks[:, None] & 1)*4)) & 15
        header = gl.load(HEADER + header_offset,
                         ns < N, 0).to(gl.uint32)
        slot = header >> 8
        escaped = gl.load(ESCAPE + (slot[None, :]-1)*128 + ks[:, None],
                          (ns[None, :] < N) & (slot[None, :] != 0), 0).to(gl.uint16)
        exponent = gl.where(slot[None, :] != 0, escaped,
                            (header[None, :] & 255)-delta).to(gl.uint16)
        bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
        w = bits.to(gl.bfloat16, bitcast=True)
        acc = mma_v2(gl.convert_layout(x, al), gl.convert_layout(w, bl), acc)
    value = acc if RAW else acc.to(gl.bfloat16).to(gl.float32)
    value = gl.convert_layout(value, ol)
    ms = gl.arange(0, BM, layout=SliceLayout(1, ol))
    no = gl.program_id(0)*BN + gl.arange(0, BN, layout=SliceLayout(0, ol))
    gl.store(OUT + ms[:, None]*N + no[None, :], value,
             (ms[:, None] < M) & (no[None, :] < N))


def project(x, weight, *, bn=64, bk=128, split=1, warps=4, stages=1, raw=False):
    """BF16 [1..16,K] and PackedHead -> BF16-rounded FP32 local logits. Up to 64 rows inside
    decode_rows.wide() (batched multi-request decode), in one 32- or 64-row tile."""
    if not isinstance(weight, (PackedHead, TiledPackedHead)):
        raise TypeError('prepare PackedHead before capture')
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or not 0 < x.shape[0] <= decode_rows.limit()
            or x.shape[1] != weight.shape[1] or x.stride(1) != 1
            or x.device != weight.device or x.device.type != 'cuda'):
        raise ValueError('expected BF16 decode activations matching the packed head')
    if (bk != 128 or split != 1 or warps not in (4, 8)
            or bn not in (32, 64, 128) or bn < warps*8 or stages != 1):
        raise ValueError('initial synchronous plan requires BK128/split1/stages1 and a valid MMA tile')
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _project[(triton.cdiv(n, bn),)](x, weight.low, weight.delta, weight.header,
        weight.escape, out, m, n, k, x.stride(0), BN=bn, BK=bk, WARPS=warps,
        RAW=raw, TILED=isinstance(weight, TiledPackedHead), BM=decode_rows.block_m(m),
        num_warps=warps, num_stages=stages)
    return out


def project_native(x, weight, **kwargs):
    """Benchmark driver adapter; the original matrix stays retained for A/B."""
    if id(weight) not in _packed:
        _packed[id(weight)] = (weight, TiledPackedHead(weight))
    return project(x, _packed[id(weight)][1], **kwargs)
