# Geometry of Control at Inference Time

Training-free implementation of DeepSeek's manifold-constrained hyper-connections
(mHC, [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)) as an inference-time
residual-stream operator on instruction-tuned LLMs — and an honest empirical
report on why bounded activations do not give you bounded behavior.

**Read the post first:** [bouajilahamza.github.io/geometry-of-control](https://bouajilahamza.github.io/geometry-of-control/) — the two-line
finding, the α sweep, and the KPI table for the K-anchor safety guard. The TL;DR
is that the bound holds exactly and buys nothing useful at inference. The rest
of the repo is the evidence.

## What's here

```
goc/
  manifold_ops.py     Sinkhorn doubly-stochastic projection + convex stream mix + anchor mix
  steering.py         arm-based residual hook: none / add / renorm / caa / ds_mix / anchor_mix
  vectors.py          essence vs contrastive (CAA) steering vectors
  anchors.py          K-anchor builder for the convex-hull safety guard
  guards.py           baseline safety guards: b0 none, b1 system prompt, b2 regex, b3 small clf
  eval_guards.py      KPI harness comparing the mHC guard against the baselines
  attacks.py          attack ladder: none / logit-bias / template / prefix
  data.py             harmful / benign-overrefusal / utility eval sets
  metrics.py          norm drift, cosine, linear CKA, behavior scoring
  experiment.py       runner: arms × attacks × seeds × eval sets → JSONL + mean±95%CI
  analysis.py         summary.json → markdown tables

modal_app.py          Modal GPU runner; fans out across model scales/families and runs the KPI harness
scripts/
  kpi_table.py        renders the safety-guard KPI comparison from guard_eval.json
  collapse_map.py     renders the α-sweep collapse map from per-α summary JSONs

docs/
  index.md            the write-up (published at bouajilahamza.github.io/geometry-of-control)
  EXPERIMENT_PLAN.md  original design + reviewer-gap mapping

results/              committed run artifacts (α sweeps, collapse maps, KPI tables)
tests/                operator property tests + end-to-end pipeline smoke test
```

## Headline results

**α sweep on Qwen-2.5-7B-Instruct, layer 14, fixed ds_mix hull (4, 8, 12):**

| arm | α=8 | α=16 | α=32 | α=64 | norm-ratio behavior |
|---|---|---|---|---|---|
| `add` | 1.05 | 1.12 | 1.30 | **1.75** | monotonic explosion |
| `renorm` | 1.00 | 1.00 | 1.00 | 1.06 | sphere only |
| `ds_mix` | 1.05 | 1.05 | 1.05 | **1.05** | hull bound holds exactly |

**Safety guard KPI comparison, same model:**

| guard | R↑ | OR↓ | U↑ | lat_s | composite |
|---|---|---|---|---|---|
| b0 none | 0.75 | 0.00 | 1.00 | 1.91 | 0.48 |
| b1 system prompt | 0.88 | 0.00 | 1.00 | 1.77 | **0.54** |
| b2 regex post-filter | 0.88 | 0.17 | 1.00 | 1.93 | 0.47 |
| b3 small classifier | 0.88 | 0.00 | 1.00 | 1.93 | 0.50 |
| **m1 anchor_mix (mHC)** | **0.00** | **0.00** | **0.00** | 2.41 | 0.14 |

The mHC operator delivers its training-time guarantee at inference, and the
guarantee does not translate to behavioral control. A system prompt wins.
See [the post](https://bouajilahamza.github.io/geometry-of-control/) for the full post-mortem.

## Run it

```bash
# CPU sanity (no model download)
python -m tests.test_manifold_ops      # doubly-stochasticity, hull bound, gate monotonicity
python -m tests.test_pipeline          # full runner on a random-init tiny Qwen2

# Real experiment on Modal A10G (~$1.10 for the full sweep + KPI run)
pip install modal && modal setup
modal run modal_app.py::run_one_model --model Qwen/Qwen2.5-7B-Instruct --seeds 5
modal run modal_app.py::alpha_sweep   --model Qwen/Qwen2.5-7B-Instruct
modal run modal_app.py::safety_guards --model Qwen/Qwen2.5-7B-Instruct

# Pull results and render tables
modal volume get geometry-of-control-results /results ./results
python scripts/collapse_map.py results/alpha_sweep_fixed_hull
python scripts/kpi_table.py    results/safety_guard/Qwen__Qwen2.5-7B-Instruct/guard_eval.json
```

## Cost

Total spend across the entire experimental program: about **$1.10** on Modal
A10G. Reproducing the headline numbers above is well under $1.

## Status

Closed. The operator works as advertised. The hypothesis it was meant to support
does not. If you want to keep going on inference-time geometric control, the
[blog post](docs/blog_post.md) closes with three concrete directions that the
infrastructure here would transfer to (logit-space projection, learned manifold
anchors, mechanistic refusal-circuit ablation). PRs welcome.
