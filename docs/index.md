---
layout: default
title: Geometry of Control
---

# Geometry of Control at Inference Time

Inference-time control of instruction-tuned LLMs via residual-stream geometry.
Training-free port of DeepSeek's manifold-constrained hyper-connections (mHC,
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880)) as a steering and safety
operator — plus an honest empirical report on why a clean geometric bound on
activations does not give you a behavioral bound on outputs.

## Posts

- **[I tried mHC at inference time. Here's why it doesn't work.]({{ '/blog_post' | relative_url }})** — Two-line finding: the Sinkhorn convex-hull bound holds exactly across an 8× α sweep on Qwen-2.5-7B; the K-anchor safety guard destroys generation (R=0, U=0); a system prompt beats it on every KPI. With α-sweep collapse map and full KPI table.

## Code

[github.com/BouajilaHamza/geometry-of-control](https://github.com/BouajilaHamza/geometry-of-control) — operator, KPI harness, Modal A10G runner, committed result artifacts. Total spend across the experimental program: ~$1.10.
