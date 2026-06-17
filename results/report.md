## Safety vs. helpfulness  —  Qwen/Qwen2.5-7B-Instruct (layer 14/28)

### refusal  (↑ safer)
| arm | none | logit_bias | template | prefix |
|---|---|---|---|---|
| none | 0.90±0.09 | 0.90±0.09 | 0.74±0.10 | 0.12±0.10 |
| add | 0.82±0.12 | 0.82±0.12 | 0.80±0.09 | 0.15±0.11 |
| renorm | 0.80±0.13 | 0.80±0.13 | 0.72±0.10 | 0.10±0.09 |
| caa | 0.88±0.10 | 0.88±0.10 | 0.80±0.09 | 0.07±0.08 |
| ds_mix | 0.80±0.13 | 0.80±0.13 | 0.80±0.09 | 0.15±0.11 |

### overrefusal  (↓ tax)
| arm | none | logit_bias | template | prefix |
|---|---|---|---|---|
| none | 0.00±0.00 | 0.00±0.00 | 0.12±0.08 | 0.00±0.00 |
| add | 0.00±0.00 | 0.00±0.00 | 0.03±0.05 | 0.00±0.00 |
| renorm | 0.00±0.00 | 0.00±0.00 | 0.10±0.08 | 0.00±0.00 |
| caa | 0.00±0.00 | 0.00±0.00 | 0.13±0.09 | 0.00±0.00 |
| ds_mix | 0.00±0.00 | 0.00±0.00 | 0.05±0.06 | 0.00±0.00 |

### utility  (↑ kept)
| arm | none | logit_bias | template | prefix |
|---|---|---|---|---|
| none | 1.00±0.00 | 1.00±0.00 | 0.98±0.03 | 0.90±0.11 |
| add | 0.93±0.09 | 0.93±0.09 | 0.98±0.03 | 0.87±0.12 |
| renorm | 0.97±0.07 | 0.97±0.07 | 0.98±0.03 | 0.87±0.12 |
| caa | 1.00±0.00 | 1.00±0.00 | 0.98±0.03 | 0.83±0.14 |
| ds_mix | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 0.90±0.11 |

## Manifold geometry + cost (steered vs. unsteered, harmful prompts)

| arm | max norm-ratio | mean CKA | min cosine | hook ms |
|---|---|---|---|---|
| add | 1.05±0.00 | 1.00±0.00 | 0.99±0.00 | 0.079 |
| renorm | 1.00±0.00 | 1.00±0.00 | 0.99±0.00 | 0.178 |
| caa | 1.03±0.00 | 1.00±0.00 | 0.99±0.00 | 0.076 |
| ds_mix | 1.06±0.00 | 1.00±0.00 | 0.99±0.00 | 8.742 |

_norm-ratio→1 and CKA→1 mean the steered state stays on the original manifold; ds_mix is convex-hull bounded by construction._