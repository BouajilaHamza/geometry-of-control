# Open Semantic Protocol (OSP) — Strategy Notes

## Current State
- Preprint: "The Open Semantic Protocol: Decoupling Search Intelligence from Indexing via Client-Side Vector Projection"
- Core result: ridge regression projector between all-MiniLM-L6-v2 (384-dim) → all-mpnet-base-v2 (768-dim), 99.0% Recall@1 STS, 99.4% AG News
- Architecture: 3-layer stack (Semantic DNS / IPFS storage / client-side linear projector)
- Status: preprint drafted, NOT YET on arXiv — DO NOT SUBMIT AS-IS

## Verdict
- **Current form: NOT novel.** Core claim (linear alignment works cross-model) covered by:
  - Mikolov 2013 (original linear projection for embeddings)
  - MUSE / Conneau & Lample 2018 (orthogonal Procrustes — *the* standard baseline, missing from paper)
  - **Moschella et al. ICLR 2023 "Relative Representations"** — closest prior, not cited, eats paper's lunch
  - Huh et al. 2024 "Platonic Representation Hypothesis"
  - Bansal et al. 2021 "Model Stitching"
- **As reframed benchmark study: publishable at Core B, plausibly Core A.**
- **Strongly aligned with Karima's field** (retrieval, similarity search, indexing). Her community: SIGIR, CIKM, VLDB, ICDE, EDBT.

## Why OSP > GoC for Karima Track
- OSP's subject matter IS her field
- Benchmark-shaped work matches her supervision style (few hours/month of "which baselines, which benchmarks, which metrics")
- Venues she publishes at accept this archetype
- Existing experimental scaffold is hers to critique, not outside her competence

## Reframe: Benchmark Paper
Working title: *"Revisiting Cross-Model Dense Retrieval: A Systematic Study of Alignment Methods, Scaling, and Quantization Interactions"*

Contribution shape: systematic empirical study + novel findings (not new method).

## Required Upgrades
- **Methods**: ridge (current) + orthogonal Procrustes + CCA + Moschella relative representations + learned MLP + identity baseline + (optional) nonlinear adapter
- **Benchmarks**: BEIR (18 datasets) + MS MARCO. Drop STS-only reliance.
- **Metrics**: MRR, nDCG@10, Recall@{1,10,100}. Recall@1 alone insufficient.
- **Model pairs**: 10-12 pairs across architecture families (MiniLM, MPNet, BGE, E5, GTE, Jina, Contriever). Not just MiniLM→MPNet.
- **Scaling studies**:
  - dimension gap (small → large)
  - index size (10k → 1M docs)
  - quantization × projection interaction (product quantization preservation under projector)
- **Findings required** (the actual contribution):
  - When does Procrustes beat learned methods?
  - Dimension-gap threshold where linear breaks down
  - Quantization codebook transferability under projection
  - Low-resource language alignment gap (if benchmarks available)
- **Ablations + theory**: when is linear projection sufficient? Orthogonality constraints impact?

## Paper Archetype Alternatives
- **Benchmark + small method combo** (stronger, common pattern): benchmark study (60%) + propose AdaptiveProj conditioned on dimension gap (40%). Mirrors BEIR / MTEB / HarmBench style.
- **Low-resource retrieval angle**: Tunisian Arabic / Darija benchmark as novelty-through-domain. Higher risk (dataset availability), higher institutional fit with UM6P.

## Venue Targets (ranked by realism)
1. **TMLR** — rolling, rigor-based, best for lone-wolf + 6-8 month timeline
2. **CIKM 2026** (Core A) — info & knowledge management, benchmark-friendly
3. **ECIR 2027** (Core B) — European IR, benchmark track
4. **SIGIR 2027** (Core A*) — stretch, if benchmark is comprehensive + findings sharp
5. **NeurIPS D&B track** — Datasets & Benchmarks track fits if toolkit is released

## Reading List (Before First Call with Karima)
1. Moschella et al. 2023 — Relative Representations (ICLR)
2. Conneau & Lample 2018 — Word Translation Without Parallel Data / MUSE
3. Thakur et al. 2021 — BEIR
4. Huh et al. 2024 — Platonic Representation Hypothesis
5. Muennighoff et al. 2023 — MTEB (benchmark archetype template)

## Kept Assets from Current Preprint
- Ridge projector training pipeline (STS-based)
- Recall@1 retrieval evaluation loop
- Cross-model retrieval framing
- Sovereignty / low-resource motivation (optional wrapper)

## What to Drop from Current Preprint
- Semantic DNS / IPFS protocol layer — engineering, not research. Separate demo artifact if at all.
- "Protocol" title framing — switch to benchmark paper framing
- Single-model-pair + single-metric claims

## Timeline (6-8 months, 10-15h/week)
- **Month 1**: baseline implementations (Procrustes, CCA, Moschella, MLP). Validate on MiniLM→MPNet.
- **Month 2**: BEIR pipeline. Scale to 10+ model pairs.
- **Month 3-4**: scaling studies (dimension gap, index size).
- **Month 5**: quantization × projection interaction.
- **Month 6**: writing, ablations, findings crystallization.
- **Month 7-8**: rebuttal / venue-specific polish.

## Compute Constraints
- No Sylergy GPU guarantee (Mohammad/Sylergy not research-invested)
- Plan: Lightning AI credits, GPU rentals, modest compute
- Design: choose BEIR subset if full 18 datasets too heavy. Prioritize diverse 6-8 datasets over exhaustive 18.

## Do NOT
- Submit current OSP to arXiv (locks naive claim into search results before proper paper exists)
- Pitch "decentralized sovereign web protocol" to Karima — her community doesn't care
- Oversell Mohammad / Sylergy as research collaborators
