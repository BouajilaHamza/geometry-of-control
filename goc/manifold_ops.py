"""
Inference-time Manifold-Constrained mixing (the novel operator).

Background
----------
DeepSeek's mHC (arXiv:2512.24880) stabilizes *training* by constraining the
hyper-connection mixing matrix between multiple residual streams to the manifold
of doubly-stochastic matrices (the Birkhoff polytope), enforced with the
Sinkhorn-Knopp algorithm. A doubly-stochastic mix is a convex combination per
row, so it cannot amplify (no explosion) and preserves the column mass (no
vanishing).

This module brings that constraint to *inference* time, training-free, as a
control operator. At a chosen layer we build a small set of candidate "streams"
for the current token (the original hidden state plus steered variants), form a
non-negative affinity matrix, project it onto the doubly-stochastic manifold via
Sinkhorn-Knopp, and emit the convex mixture. The output is therefore guaranteed
to lie in the convex hull of the streams -> norm-bounded by the largest stream,
which is the inference-time analog of mHC's stability guarantee.

This is contrasted in experiments against:
  * add     : h + alpha*v               (unconstrained; norm can blow up)
  * renorm  : (h + alpha*v) rescaled to ||h||   (sphere projection; ad hoc)
  * ds_mix  : this operator                       (Birkhoff-projected convex mix)

Everything here is weight-independent and unit-tested in tests/test_manifold_ops.py.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Sinkhorn-Knopp projection onto the doubly-stochastic manifold
# --------------------------------------------------------------------------- #
def sinkhorn_knopp(
    log_alpha: torch.Tensor,
    n_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project a square matrix onto (approximately) doubly-stochastic via
    Sinkhorn-Knopp iterations, done in log-space for numerical stability.

    Args:
        log_alpha: [..., n, n] log-affinities (any real values).
        n_iters:   number of row/column normalization passes (mHC uses 20).
    Returns:
        [..., n, n] doubly-stochastic matrix (rows and columns ~sum to 1).
    """
    if log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError("Sinkhorn requires square matrices.")
    la = log_alpha
    for _ in range(n_iters):
        la = la - torch.logsumexp(la, dim=-1, keepdim=True)  # row normalize
        la = la - torch.logsumexp(la, dim=-2, keepdim=True)  # col normalize
    return torch.exp(la).clamp_min(eps)


def doubly_stochastic_error(M: torch.Tensor) -> float:
    """Max deviation of row/column sums from 1.0 (0.0 == perfectly DS)."""
    row = (M.sum(dim=-1) - 1.0).abs().max().item()
    col = (M.sum(dim=-2) - 1.0).abs().max().item()
    return float(max(row, col))


# --------------------------------------------------------------------------- #
# Stream construction + convex mixing
# --------------------------------------------------------------------------- #
def build_streams(
    h: torch.Tensor,
    v: torch.Tensor,
    alphas: tuple[float, ...],
) -> torch.Tensor:
    """
    Build n = len(alphas)+1 candidate streams for the current position.

    Stream 0 is the untouched hidden state (the capability-preserving anchor);
    the rest are steered variants h + alpha_k * v at increasing strengths.

    Args:
        h: [B, 1, d] current-position hidden state.
        v: [1, 1, d] unit steering direction.
        alphas: steering strengths for the steered streams.
    Returns:
        [B, n, d] stacked streams.
    """
    streams = [h]  # anchor
    for a in alphas:
        streams.append(h + v * float(a))
    return torch.cat(streams, dim=1)  # [B, n, d]


def stream_affinity(
    streams: torch.Tensor,
    gate: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Build log-affinities between streams for Sinkhorn.

    Geometry term: cosine similarity between streams (so the mix respects how
    far apart candidates are). Control term: a `gate` in [0,1] biases mass
    toward the steered streams (index > 0) -- this is the single knob that says
    "how hard to steer". The doubly-stochastic projection then turns these
    preferences into a valid convex mixing operator that cannot explode.

    Returns [B, n, n] log-affinity.
    """
    B, n, d = streams.shape
    s = streams / streams.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = torch.matmul(s, s.transpose(-1, -2))  # [B, n, n] in [-1,1]
    log_aff = cos / max(temperature, 1e-6)

    # Control bias: raise the anchor<->steered *off-diagonal* affinities so the
    # anchor row sends mass to the steered streams. We deliberately do NOT add a
    # whole-column constant: Sinkhorn's column normalization would cancel it
    # exactly (a full-column additive shift is a no-op after logsumexp), leaving
    # the gate inert. Off-diagonal entries survive the projection.
    if n > 1 and gate != 0.0:
        bias = torch.zeros(n, n, device=streams.device, dtype=streams.dtype)
        bias[0, 1:] = float(gate)   # anchor -> steered
        bias[1:, 0] = float(gate)   # steered -> anchor (keep matrix feasible/symmetric)
        log_aff = log_aff + bias.view(1, n, n)
    return log_aff


def ds_mix_anchors(
    h: torch.Tensor,
    anchors: torch.Tensor,
    gate: float,
    n_iters: int = 20,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-anchor mHC guard.

    Args:
        h:       [B, 1, d] current-position hidden state.
        anchors: [K, d]    K precomputed in-domain hidden states.
    Returns:
        h_new: [B, 1, d] anchor row of the doubly-stochastic mix; constrained
               to the convex hull of (current, anchors).
        M:     [B, K+1, K+1] the doubly-stochastic mixing matrix.
    """
    B = h.shape[0]
    K = anchors.shape[0]
    a = anchors.to(dtype=h.dtype, device=h.device).unsqueeze(0).expand(B, K, -1)  # [B,K,d]
    streams = torch.cat([h, a], dim=1)                                            # [B,K+1,d]
    log_aff = stream_affinity(streams, gate, temperature)
    M = sinkhorn_knopp(log_aff, n_iters=n_iters)
    mixed = torch.matmul(M, streams)
    return mixed[:, 0:1, :], M


def ds_mix_steer(
    h: torch.Tensor,
    v: torch.Tensor,
    alphas: tuple[float, ...],
    gate: float,
    n_iters: int = 20,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The inference-time mHC operator.

    Returns:
        h_new:  [B, 1, d] mixed current-position hidden (anchor row of M @ S).
        M:      [B, n, n] the doubly-stochastic mixing matrix (for diagnostics).
    """
    streams = build_streams(h, v, alphas)            # [B, n, d]
    log_aff = stream_affinity(streams, gate, temperature)
    M = sinkhorn_knopp(log_aff, n_iters=n_iters)     # [B, n, n], doubly stochastic
    mixed = torch.matmul(M, streams)                 # [B, n, d] convex combinations
    # Emit the anchor stream's updated value: a convex combination of all
    # candidates, hence guaranteed within their convex hull (norm-bounded).
    h_new = mixed[:, 0:1, :]
    return h_new, M
