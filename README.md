# Geometry of Control at Inference Time

Inference-time control of small instruction-tuned LLMs via residual-stream
interventions — and, centrally, a **training-free implementation of DeepSeek's
manifold-constrained mixing (mHC, arXiv:2512.24880) at inference time**.

## What's here

```
goc/                      experiment package (the rigorous harness)
  manifold_ops.py         Sinkhorn doubly-stochastic projection + convex stream mix  ← the contribution
  steering.py             arm-based residual intervention (none/add/renorm/caa/ds_mix), hooks, capture
  vectors.py              steering vectors: essence (v1) vs contrastive/CAA
  attacks.py              attack ladder: none / logit-bias / template / prefix (+ GCG extension point)
  data.py                 harmful / benign-over-refusal / utility eval sets (+ AdvBench/XSTest loaders)
  metrics.py              manifold probes (norm drift, cosine, linear CKA) + behavior scoring
  experiment.py           runner: arms × attacks × seeds × eval sets → JSONL + mean±95%CI
  analysis.py             summary.json → reviewer-facing markdown tables
modal_app.py              GPU runner; fans out across model scales/families
tests/                    operator property tests + end-to-end pipeline smoke test
docs/EXPERIMENT_PLAN.md   maps every reviewer gap → treatment, with hypotheses
experiments/              the original v1 scripts (kept for provenance)
```

## The idea in one paragraph

v1's "renorm" steering only preserved the L2 norm of a single hidden vector (a
sphere). DeepSeek's mHC instead constrains the *mixing matrix* between residual
streams to be **doubly-stochastic** (Birkhoff polytope) via **Sinkhorn–Knopp**,
guaranteeing convex (non-amplifying, non-vanishing) mixing — but only during
training. `goc/manifold_ops.py` brings that constraint to **inference**: at a
chosen layer we build candidate streams (original + steered), Sinkhorn-project
their affinity to doubly-stochastic, and emit the convex mixture. The result is
provably norm-bounded by the convex hull of the streams.

## Run it

```bash
# 1) Validate the operator + pipeline on CPU (no model download):
python -m tests.test_manifold_ops      # doubly-stochasticity, norm-bound, gate monotonicity
python -m tests.test_pipeline          # full runner on a random-init tiny Qwen2

# 2) Real experiment on GPU via Modal (HuggingFace weights):
pip install modal && modal setup
modal run modal_app.py --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-7B-Instruct" --seeds 5
modal volume get geometry-of-control-results /results ./results
python -m goc.analysis results/<model>__summary.json --md report.md
```

See `docs/EXPERIMENT_PLAN.md` for the full design, the reviewer-gap mapping, and
the hypotheses (including the honest null result it can return).

## Status

The operator and the entire experiment pipeline are implemented and validated on
CPU. Behavioral safety/scaling numbers require real model weights, which run on
the Modal GPU runner (the dev container has no GPU and blocks HuggingFace).
