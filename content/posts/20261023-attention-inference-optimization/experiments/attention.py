"""Forward-only causal MHA: readable reference and a small Triton kernel.

Inputs are contiguous [batch, heads, sequence, head_dim], FP16 for Triton.
No dropout, backward, GQA, variable lengths, or fully masked valid rows.
"""
import math

import torch
import triton
import triton.language as tl


def online_softmax(x, block=64):
    """Stream the normalizer, then reread x to return the full softmax vector."""
    x = x.float()
    m = torch.full(x.shape[:-1], -float("inf"), device=x.device)
    l = torch.zeros_like(m)
    for start in range(0, x.shape[-1], block):
        xb = x[..., start:start + block]
        m_new = torch.maximum(m, xb.amax(dim=-1))
        l = torch.exp(m - m_new) * l + torch.exp(xb - m_new[..., None]).sum(-1)
        m = m_new
    return torch.exp(x - m[..., None]) / l[..., None]


def naive_attention(q, k, v):
    n = q.shape[-2]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    causal = torch.arange(n, device=q.device)
    scores.masked_fill_(causal[:, None] < causal[None, :], -float("inf"))
    return scores.softmax(dim=-1) @ v


def online_attention(q, k, v, block=64):
    """FP32 state, no full NxN matrix; Python loops are for derivation only."""
    q, k, v = q.float(), k.float(), v.float()
    n, d = q.shape[-2:]
    out = torch.empty_like(q)
    for start in range(0, n, block):
        qi = q[..., start:start + block, :]
        rows = torch.arange(start, start + qi.shape[-2], device=q.device)
        m = torch.full(qi.shape[:-1], -float("inf"), device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros_like(qi)
        for j in range(0, start + qi.shape[-2], block):
            kj, vj = k[..., j:j + block, :], v[..., j:j + block, :]
            cols = torch.arange(j, j + kj.shape[-2], device=q.device)
            s = (qi @ kj.transpose(-1, -2)) / math.sqrt(d)
            s.masked_fill_(rows[:, None] < cols[None, :], -float("inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new[..., None])
            acc = alpha[..., None] * acc + p @ vj
            l = alpha * l + p.sum(dim=-1)
            m = m_new
        out[..., start:start + qi.shape[-2], :] = acc / l[..., None]
    return out


@triton.jit
def _flash_fwd(Q, K, V, O, N: tl.constexpr, D: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, D)
    base = bh * N * D
    q = tl.load(Q + base + rows[:, None] * D + dims[None, :],
                rows[:, None] < N, other=0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    # Skip blocks strictly above the causal diagonal.
    for start in range(0, tl.minimum((tile + 1) * BM, N), BN):
        c = start + cols
        k = tl.load(K + base + dims[:, None] + c[None, :] * D,
                    c[None, :] < N, other=0)
        v = tl.load(V + base + c[:, None] * D + dims[None, :],
                    c[:, None] < N, other=0)
        s = tl.dot(q, k) * (1.0 / tl.sqrt(float(D)))
        s = tl.where((rows[:, None] >= c[None, :]) & (c[None, :] < N),
                     s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        l = alpha * l + tl.sum(p, axis=1)
        m = m_new
    out = acc / l[:, None]
    tl.store(O + base + rows[:, None] * D + dims[None, :], out,
             rows[:, None] < N)


def flash_attention(q, k, v, bm=64, bn=64):
    assert q.is_cuda and q.dtype == torch.float16
    assert q.shape == k.shape == v.shape and q.ndim == 4
    assert all(x.is_contiguous() and x.dtype == q.dtype and x.device == q.device
               for x in (q, k, v))
    b, h, n, d = q.shape
    assert d in (32, 64, 128)
    assert bm in (32, 64, 128) and bn in (32, 64, 128)
    out = torch.empty_like(q)
    _flash_fwd[(triton.cdiv(n, bm), b * h)](q, k, v, out, n, d, bm, bn,
                                          num_warps=4, num_stages=2)
    return out
