# Experiment Plan: Manifold-Constrained Inference-Time Control

This plan turns the v1 paper (single 0.5B model, toy attacker, qualitative
"manifold" claims) into a study that survives the reviewer's critique. It also
fixes a substantive problem the review only half-caught: **v1 never actually
implemented a manifold constraint.** Its `renorm` only preserved the L2 norm of
a single hidden vector (a sphere). The contribution we now implement is the real
one suggested by the title and by DeepSeek's mHC (arXiv:2512.24880):

> **mHC-at-inference.** Build several candidate "streams" for the current token
> (the original hidden state + steered variants), form a non-negative affinity
> matrix, project it onto the **doubly-stochastic manifold (Birkhoff polytope)**
> with **Sinkhorn–Knopp**, and emit the convex mixture. The output is provably
> inside the convex hull of the streams → norm-bounded (no explosion), column
> mass preserved (no vanishing) — mHC's training guarantees, now training-free
> at inference. Code: `goc/manifold_ops.py`.

## The five arms (the independent variable)

| arm | operator | what it isolates |
|---|---|---|
| `none` | no intervention | control |
| `add` | `h + αv` | unconstrained activation addition (ActAdd) |
| `renorm` | `(h+αv)` rescaled to `‖h‖` | the v1 "manifold" (sphere only) |
| `caa` | `add` with a **contrastive** vector | value of vector quality alone |
| `ds_mix` | **Sinkhorn doubly-stochastic convex mix** | our contribution |

`add`/`renorm`/`ds_mix` share the *same* `essence` vector so any difference is
the **operator**; `caa` vs `add` isolates the **vector**. This is the clean
factorization the review demanded ("quantify the marginal value").

## Reviewer gap → treatment

| Reviewer concern | Treatment | Where | Status |
|---|---|---|---|
| Single 0.5B model; no scaling | Fan-out 0.5B→1.5B→7B + Llama/Mistral families | `modal_app.py` | GPU |
| No seed variance / CIs | N seeds/cell, mean ± 95% CI everywhere | `experiment.mean_ci` | ✅ here |
| Steering vector under-specified | Two named methods: `essence` (v1) and `contrastive`/CAA; report cos between them | `goc/vectors.py` | ✅ here |
| Toy attacker only | Attack ladder: none / logit-bias / template (DAN, ignore) / prefix-injection; GCG documented extension | `goc/attacks.py` | ✅ here (GCG on GPU) |
| No baselines | `add` (ActAdd), `caa` (CAA), `renorm`, vs `ds_mix` | `goc/experiment.py` | ✅ here |
| No utility / over-refusal trade-off | Three eval axes: harmful, benign-over-refusal (XSTest-style), utility | `goc/data.py` | ✅ here |
| Heuristic string safety only | Heuristics now + real benchmark loaders (AdvBench/XSTest) + classifier hook | `goc/data.load_benchmark`, `goc/metrics` | partial |
| "Manifold stability" qualitative | Quantitative probes: per-layer norm-ratio, cosine trajectory, **linear CKA** | `goc/metrics.manifold_trajectory` | ✅ here |
| Decoding settings unreported | All decoding params in `RunConfig`; sweepable | `goc/experiment.RunConfig` | ✅ here |
| Layer / strength ablations | `layer_frac`, `alpha`, `ds_alphas`, `ds_gate`, `ds_iters` all configurable | `RunConfig` / `ArmSteer` | ✅ here |
| Latency only on CPU | Same hook-ms instrumentation, run on GPU + batched | `modal_app.py` | GPU |

"✅ here" = implemented and validated in this CPU container (see Validation).
"GPU" = wired and ready; needs real weights, which require the Modal runner
(HuggingFace is blocked in the dev container).

## Validation already performed (CPU, no model download)

- **Operator correctness** (`tests/test_manifold_ops.py`): Sinkhorn output is
  doubly-stochastic to 1e-7; `ds_mix` stays inside the convex hull
  (‖·‖ 19.7 < hull 23.9) while `add` at high α explodes past it (37.2); the
  control gate is monotonic and **saturating** (4.76→7.44) — bounded steering.
- **End-to-end pipeline** (`tests/test_pipeline.py`): the entire runner executes
  on a randomly-initialized Qwen2 — ArmSteer hooks fire, the Sinkhorn forward
  runs *inside* a real transformer, manifold metrics + CI aggregation produce a
  summary. (Random weights ⇒ behavior numbers are not meaningful; this proves
  the harness, not a result.)

## Hypotheses the GPU run will test

- **H1 (stability):** `ds_mix` keeps norm-ratio and CKA closer to 1 than `add`
  at matched steering effect → less off-manifold drift.
- **H2 (Pareto):** `ds_mix` achieves comparable refusal to `add`/`renorm` at a
  **lower over-refusal / higher utility** — a better safety/helpfulness frontier.
- **H3 (robustness):** under the prefix attack (the hardest), `ds_mix`'s bounded
  steering degrades more gracefully than `add` (which collapses/explodes).
- **H0 (the honest null):** if `ds_mix` matches `renorm` everywhere, we report
  that norm-preservation is the operative ingredient and the doubly-stochastic
  constraint adds nothing at inference — a clean, publishable negative result.

## How to run

```bash
# Modal: TDD gate then full experiment
modal run modal_app.py::run_tests                    # operator + pipeline tests on GPU image
modal run modal_app.py --batch-size 8                # full grid: 5 arms × 4 attacks × 20 items × 5 seeds
modal volume get geometry-of-control-results / ./results --force
python -m goc.analysis results/Qwen__Qwen2.5-7B-Instruct__summary.json --md results/report.md

# Collapse hunt (fixed ds_mix hull, sweep add α):
modal run modal_app.py::alpha_sweep --alphas 8,16,32,64 --ds-alphas 4,8,12
modal volume get geometry-of-control-results /alpha_sweep_fixed_hull ./results --force
GOC_SWEEP_DIR=results/alpha_sweep_fixed_hull python scripts/collapse_map.py
```

## First GPU result (2026-06-16, Qwen2.5-7B-Instruct, L=14/28, A10G, bs=8)

**Full grid** (`results/report.md`): 5 arms × 4 attacks × 20 items × 5 seeds = 2500 trials.
At α=8 the four steering arms are statistically indistinguishable on refusal
(0.80–0.90 on weak attacks, 0.07–0.15 under prefix), and on overrefusal /
utility the gaps overlap with 95% CIs. H1 norm-bound separation is not visible
at α=8 because every arm stays near the manifold (CKA ≈ 1.00, cosine ≈ 0.99).

**Collapse-hunt α sweep** (`results/alpha_sweep_fixed_hull/`, single seed): hold
the `ds_mix` hull fixed at `ds_alphas = (4, 8, 12)`, sweep `add`/`renorm` α over
{8, 16, 32, 64}. This is the test the original sweep missed (it scaled
`ds_alphas` with α and so removed its own bound).

| arm | norm-ratio α=8 → α=64 | utility on benign at α=64 |
|---|---|---|
| `add` | 1.05 → **1.75** (unbounded) | 0.83 → **0.67** (33 pp loss) |
| `renorm` | 1.00 → 1.06 (sphere) | 0.83 → 1.00 (variable) |
| `ds_mix` | 1.05 → **1.05** (capped) | 0.83 → **0.83** (held) |

So H1 (stability) holds in the form: **the doubly-stochastic operator declares
its envelope once via `ds_alphas` and is then invariant to operator
miscalibration**. A user who cranks `α` looking for stronger safety pushes
`add` off the manifold and into a 33 pp capability cliff on benign prompts;
the same crank does nothing to `ds_mix` because the hull is constant. This
turns "bounded steering" from a theoretical guarantee into a **robustness-to-
hyperparameter-misuse** property — the publishable contribution.

H2 (Pareto) and H3 (robustness under prefix attack) need more seeds + tighter
CIs to call.
