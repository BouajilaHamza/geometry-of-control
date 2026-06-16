"""
Correctness + property tests for the inference-time mHC operator.

These are weight-independent: they validate the mathematical guarantees that
motivate the operator (doubly-stochasticity, norm-boundedness, control gate
monotonicity) and contrast it with the `add` baseline that can explode.
Run: python -m tests.test_manifold_ops
"""

from __future__ import annotations

import torch

from goc.manifold_ops import (
    doubly_stochastic_error,
    ds_mix_steer,
    sinkhorn_knopp,
    stream_affinity,
    build_streams,
)


def test_sinkhorn_is_doubly_stochastic():
    torch.manual_seed(0)
    log_alpha = torch.randn(4, 6, 6)
    M = sinkhorn_knopp(log_alpha, n_iters=30)
    err = doubly_stochastic_error(M)
    assert err < 1e-4, f"row/col sums deviate by {err}"
    assert (M >= 0).all(), "doubly-stochastic matrices are non-negative"
    return err


def test_convex_mix_cannot_explode():
    """The mHC guarantee: a doubly-stochastic mix is norm-bounded by the
    largest stream, whereas plain activation addition is not."""
    torch.manual_seed(0)
    d = 256
    h = torch.randn(8, 1, d)
    v = torch.randn(1, 1, d)
    v = v / v.norm(dim=-1, keepdim=True)
    alphas = (4.0, 8.0, 16.0)  # aggressive strengths

    streams = build_streams(h, v, alphas)            # [8, n, d]
    max_stream_norm = streams.norm(dim=-1).max(dim=1).values  # [8]

    h_ds, M = ds_mix_steer(h, v, alphas, gate=2.0, n_iters=20)
    ds_norm = h_ds.norm(dim=-1).squeeze(-1)          # [8]

    # add baseline pushed BEYOND the strongest stream (alpha=32 > hull's 16):
    # this is what an over-eager steering strength does in the `add` arm.
    h_add = h + v * 32.0
    add_norm = h_add.norm(dim=-1).squeeze(-1)

    ds_ok = bool((ds_norm <= max_stream_norm + 1e-3).all())
    add_explodes = bool((add_norm > max_stream_norm).any())
    assert ds_ok, "ds_mix exceeded convex-hull norm bound"
    return {
        "ds_norm_max": float(ds_norm.max()),
        "max_stream_norm": float(max_stream_norm.max()),
        "add_norm_max": float(add_norm.max()),
        "ds_within_hull": ds_ok,
        "add_can_exceed_hull": add_explodes,
        "ds_matrix_error": doubly_stochastic_error(M),
    }


def test_gate_monotonicity():
    """Higher gate -> mixed state moves further from the anchor toward steer."""
    torch.manual_seed(1)
    d = 256
    h = torch.randn(1, 1, d)
    v = torch.randn(1, 1, d)
    v = v / v.norm(dim=-1, keepdim=True)
    alphas = (5.0, 10.0)

    dists = []
    for gate in (0.0, 1.0, 3.0, 6.0):
        h_new, _ = ds_mix_steer(h, v, alphas, gate=gate, n_iters=20)
        dists.append(float((h_new - h).norm()))
    # non-decreasing displacement from anchor as gate grows
    mono = all(dists[i] <= dists[i + 1] + 1e-4 for i in range(len(dists) - 1))
    assert mono, f"gate displacement not monotonic: {dists}"
    return {"displacements": [round(x, 3) for x in dists]}


if __name__ == "__main__":
    err = test_sinkhorn_is_doubly_stochastic()
    print(f"[PASS] sinkhorn doubly-stochastic (max row/col err = {err:.2e})")

    stats = test_convex_mix_cannot_explode()
    print(f"[PASS] convex mix norm-bounded:")
    print(f"        ds_norm_max      = {stats['ds_norm_max']:.2f}")
    print(f"        max_stream_norm  = {stats['max_stream_norm']:.2f}  (ds stays under this)")
    print(f"        add_norm_max     = {stats['add_norm_max']:.2f}  (add can exceed hull = {stats['add_can_exceed_hull']})")
    print(f"        ds_matrix_error  = {stats['ds_matrix_error']:.2e}")

    mono = test_gate_monotonicity()
    print(f"[PASS] gate monotonicity: displacements = {mono['displacements']}")
    print("\nAll manifold-operator property tests passed.")
