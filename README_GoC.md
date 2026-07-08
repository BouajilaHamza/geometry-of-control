# Geometry of Control (GoC) — Strategy Notes

## Current State
- Preprint: "The Geometry of Control at Inference Time: Stochastic Constraints, Normalized Steering, and a Safety Clamp for Small LLMs"
- Model: Qwen2.5-0.5B-Instruct, CPU inference
- Scope: 8 phases — hard/soft/internal constraints + dual-site safety clamp + 2D grid sweep
- Status: complete draft, arXiv-ready after fixes

## Verdict
- **Not publishable at Core B/A as-is**
- **Not aligned with Karima Echihabi's field** (her area: data series indexing, similarity search, high-D indexing — NOT LLM safety/interpretability)
- **Do not anchor PhD collaboration on this paper**

## Key Gaps (what reviewers will reject on)
- Model scale: 0.5B on CPU = "toy setup" signal. Need Llama-3-8B / Qwen2.5-7B minimum.
- Missing baselines (critical):
  - ITI (Li et al. NeurIPS 2023)
  - ActAdd / CAA (Panickssery et al.)
  - RepE (Zou et al.)
  - **Arditi et al. NeurIPS 2024 "Refusal is mediated by a single direction"** — overlaps safety clamp territory directly
- Safety eval weak: string markers + redaction not a real benchmark. Need HarmBench, AdvBench, XSTest, StrongREJECT.
- N=3 prompts x 3 trials = statistically meaningless. Need hundreds.
- No seeds, no confidence intervals.
- "Entropy correlates with failure" = correlation claim without causal test.

## Crowded Field
LLM safety / activation steering exploding in 2025-2026. Arditi refusal-direction paper eats much of the territory. Standalone "dual-site safety clamp" framing will struggle for novelty.

## Recommended Path
1. **Post to arXiv** (after minor limitations-paragraph fix) — visibility, citation anchor.
2. **Workshop submission** — NeurIPS SoLaR, ICLR SafeAI, SafetyML. NOT Core B but credible.
3. **Long-term upgrade target**: TMLR (rolling submission, rigor-based). Core A main tracks (NeurIPS/ICML/ICLR) unrealistic without group-level resources.
4. **Do NOT submit to Karima as PhD project**. Her community won't review it charitably. Keep separate from UM6P track.

## Reusable Assets
- Forward-hook steering pipeline (code in `experiments/`)
- 2D grid sweep methodology
- Renormalized injection formula (mHC-adjacent)
- Attacker logit-bias threat model framing

Keep these. Useful for future steering-adjacent projects or workshop resubmission.

## Do NOT
- Mention GoC as lead topic to Karima
- Invest more effort before OSP/retrieval benchmark is settled
- Expect Mohammad or Sylergy to fund GoC extensions — they won't
