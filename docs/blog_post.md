# I tried mHC at inference time. Here's why it doesn't work.

**TL;DR:** DeepSeek's manifold-constrained hyper-connections (mHC) give you provable norm bounds for free at training time. The obvious next move — port the same Sinkhorn-Knopp projection to inference as a steering or safety operator — sounds great on paper. I built it, ran it on Qwen-2.5-7B on Modal for about $1.10 of GPU, and the result is a clean two-line finding: **the bound holds exactly as advertised, and the bound buys you nothing useful at inference.** This post is the honest write-up. The code, the α sweep, and the KPI tables are all in the repo.

The pitch for mHC-at-inference is seductive. mHC is a residual-stream operator that projects a non-negative affinity matrix onto the Birkhoff polytope with Sinkhorn-Knopp, so the mixture stays inside the convex hull of its inputs. No norm explosion. No vanishing. The DeepSeek paper proves this for training. The training-free analog writes itself: take a hidden state plus a couple of steered variants, run them through the same projection, and you get an activation steering operator with a hard geometric guarantee. Same math, different application.

When it works, it's tidy. When it doesn't, the model still sounds fluent but doesn't actually do what you wanted. And the literature on mHC doesn't tell you which outcome to expect at inference, because nobody has run this experiment.

## The 60-second version of what I built

Five arms over the same residual hook at layer 14 of Qwen-2.5-7B-Instruct:

| arm | operator | what it isolates |
|---|---|---|
| `none` | no intervention | control |
| `add` | `h + αv` | unconstrained activation addition (ActAdd) |
| `renorm` | `(h + αv)` rescaled to `‖h‖` | the sphere — "manifold" in name only |
| `caa` | `add` with a contrastive vector | vector quality vs operator quality |
| `ds_mix` | Sinkhorn doubly-stochastic mix of `{h, h+αv, h-αv}` | the mHC port |

`add`, `renorm`, and `ds_mix` share the same essence vector, so any difference is the operator. `caa` vs `add` isolates the vector. Four attacker conditions (none, logit bias, template injection, prefix injection), four α values (8, 16, 32, 64), with the ds_mix hull held at a fixed `(4, 8, 12)` so the test actually probes the bound instead of dragging the hull along with α.

Anchors get the same treatment in a second experiment: build K=8 in-domain anchor states from canonical refusal continuations, then project the live hidden state into their convex hull as a "domain guard."

## Result 1 — the bound holds, exactly

Max norm ratio across α ∈ {8, 16, 32, 64}, worst attacker per cell:

| arm | α=8 | α=16 | α=32 | α=64 |
|---|---|---|---|---|
| `add` | 1.05 | 1.12 | 1.30 | **1.75** |
| `renorm` | 1.00 | 1.00 | 1.00 | 1.06 |
| `ds_mix` | 1.05 | 1.05 | 1.05 | **1.05** |

`add` explodes monotonically. `renorm` holds the sphere because that's all it does. `ds_mix` holds the convex hull bound exactly, across an 8× sweep in α, because Sinkhorn doesn't care how big α gets — the projection lives inside the same simplex. That's the cleanest empirical confirmation of mHC's training-free promise I can give you.

This is the only good news in the post.

## Result 2 — bounded ≠ useful

Refusal recall under template attack at the same sweep:

| arm | α=8 | α=16 | α=32 | α=64 |
|---|---|---|---|---|
| `add` | 0.69 | 0.69 | 0.75 | **0.88 (utility 0.92)** |
| `ds_mix` | 0.69 | 0.69 | 0.69 | **0.69 (utility 1.00)** |

`add` eventually pushes refusal up to 0.88 — at the cost of utility collapse and a 1.75× norm explosion. `ds_mix` is flat. It's flat because the bound is flat. Every α value lives inside the same convex hull, so the model sees the same effective input no matter what α you set. **You set the hull once. After that, α is a no-op.** That is a property, not a bug, but it means there is no knob to turn — you either picked good hull endpoints or you didn't, and there is no way to dial pressure beyond what those endpoints already encode.

A geometric bound on activations is not a behavioral bound on outputs. The unembedding doesn't care that your hidden state stayed in a polytope.

## Result 3 — the K-anchor "domain guard" destroys generation

Same model, same layer, K=8 in-domain anchor states, gate = 3.0:

| guard | R↑ | OR↓ | U↑ | lat_s | tok/s | composite |
|---|---|---|---|---|---|---|
| b0 none | 0.75 | 0.00 | 1.00 | 1.91 | 27.7 | 0.48 |
| b1 system prompt | 0.88 | 0.00 | 1.00 | 1.77 | 27.5 | **0.54** |
| b2 regex post-filter | 0.88 | 0.17 | 1.00 | 1.93 | 27.6 | 0.47 |
| b3 small classifier | 0.88 | 0.00 | 1.00 | 1.93 | 27.5 | 0.50 |
| **m1 anchor_mix (mHC guard)** | **0.00** | **0.00** | **0.00** | 2.41 | 26.5 | **0.14** |

The mHC guard refuses zero of the harmful prompts and answers zero of the utility prompts correctly. It doesn't refuse and it doesn't help. The model still emits tokens at roughly the same throughput; they just aren't coherent in either direction. A system prompt — the most boring baseline imaginable — wins on every axis.

I ran one gate. I did not sweep. A sweep might find a gate value where the guard is merely bad instead of catastrophic. I don't think it will rescue the approach, for reasons in the next section, but the disclosure is fair.

## Why this fails

Four mechanisms, in roughly decreasing order of damage:

**Convex hull is not a domain.** K refusal-aligned hidden states span a K-simplex in a d-dimensional space (d = 3584 for Qwen-7B). The actual safe region of activation space is non-convex, much higher-dimensional, and not centered on any small set of anchors. Project a math-question activation into your refusal simplex and you do not get a safer math-question activation. You get a corrupted activation that decodes to noise.

**Static anchors, autoregressive context.** The anchors are built once at calibration time and pulled toward at every generation step regardless of what's being generated. Token 4 of a benign arithmetic answer gets the same pull as token 4 of a harmful synthesis instruction. Either you crank the gate (every generation breaks, which is what we saw) or you don't (no effective constraint).

**KV cache cascade.** Modifying token t's hidden state corrupts the keys and values that token t+1 attends to. One projected token is recoverable. Sixty-four are not. This is why the guard hits zero on utility even on prompts where no intervention should be triggered — the projection runs on every token regardless.

**Activation ≠ output.** Even the well-behaved `ds_mix` arm in result 1 — perfectly bounded, never explodes — produces refusal scores that are flat across α. A bound at the hidden layer does not translate to a bound at the output. The unembedding `Wh + b` is not domain-preserving. A "safe" hidden state can decode to an unsafe token, and a perturbed hidden state can still decode to fluent text. The geometric guarantee lives in the wrong space.

None of these are fixable by tuning. They are properties of where in the stack you intervene and what kind of object you are intervening on.

## What "harness engineering" actually means here

The hypothesis I started with — *use geometry to force the model's output distribution into a domain, regardless of prompt* — is the right kind of question. The implementation I tried is the wrong tool. The approaches that actually exist for this question, ranked roughly by guarantee strength:

| Approach | Where it intervenes | Guarantee | Maturity |
|---|---|---|---|
| Constrained decoding (GBNF, outlines, LM Format Enforcer) | Output token, grammar level | Hard, lexical | Production |
| Logit-level safety masks (Llama Guard at logit time) | Output token, distribution level | Soft, semantic | Active |
| Mechanistic refusal-circuit ablation (Arditi et al.) | Specific attention heads / MLPs | Empirical, circuit-level | Cutting edge |
| Learned manifold projection (VAE/flow on safe activations, then decode) | Hidden state, learned region | Soft, semantic | Mostly speculative |
| **mHC convex-hull guard at hidden state (this post)** | Hidden state, static anchor hull | None empirically | Doesn't work |

The pattern: anything that works either intervenes at the output layer where the geometry is the geometry of token probabilities, or learns the in-domain region from data instead of assuming a convex hull of a handful of anchors will do.

## What I'd try next if I kept going

I'm not going to keep going on this line, but if you wanted to:

- **Move the mHC projection from the hidden state to the logits.** Project the logit vector into the convex hull of {raw logits, refusal-template logits}. The geometry of token probabilities is much closer to the geometry of behavior than the geometry of hidden states.
- **Learn the anchors.** Train a small VAE or normalizing flow on hidden states from in-domain conversations. Project to the learned manifold, not a hand-picked simplex. Expensive and not obviously better than mech-interp, but at least it tests the actual hypothesis.
- **Try the refusal-direction work head-on.** Arditi et al. show that refusal is mediated by a single direction in many models. Ablating that direction is a clean, well-defined intervention with empirical evidence. The infrastructure in this repo — Modal harness, batched eval, KPI machinery, attacker ladder — transfers directly.

If somebody else wants to try one of these on top of this codebase, the receipts are all here.

## Honest assessment: should you use mHC at inference?

**Yes, if:**

- You want a provable norm bound on an intermediate activation for analysis or instrumentation. The Sinkhorn projection delivers what it says.
- You're studying activation geometry and want a clean, parameterless way to constrain a residual stream for ablation.

**No, if:**

- You want to control model behavior. Bounded activations do not give you bounded behavior.
- You want a safety guard. Static-anchor mHC destroys generation; a system prompt outperforms it on every KPI.
- You want a steering knob. The bound makes α a no-op past the first step.
- You want guarantees on output. The geometric guarantee is at the wrong layer.

mHC is a real result for training. At inference, the operator works exactly as advertised and the operator is not what you needed. That gap — between what your method provably does and what your method usefully does — is the whole post.

Code, sweeps, and the full Modal harness: [BouajilaHamza/geometry-of-control](https://github.com/BouajilaHamza/geometry-of-control). If you're running similar experiments on inference-time geometric constraints, I'd like to compare notes.
