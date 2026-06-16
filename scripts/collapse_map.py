"""Render an α-sweep collapse map from results/alpha_sweep/alpha_*.

Usage:
    python scripts/collapse_map.py
"""

from __future__ import annotations

import glob
import json
import os
import re


SWEEP_DIR = os.environ.get("GOC_SWEEP_DIR", "results/alpha_sweep")
ATTACKS = ("none", "logit_bias", "template", "prefix")
ARMS = ("add", "renorm", "ds_mix")
METRICS = (
    ("refusal_mean",       "ref"),
    ("overrefusal_mean",   "ovr"),
    ("utility_mean",       "util"),
    ("norm_ratio_mean",    "nrm"),
    ("max_entropy_mean",   "H"),
)


def _alpha_from_path(p: str) -> float:
    m = re.search(r"alpha_([\d.]+)", p)
    return float(m.group(1)) if m else float("nan")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "  - "
    return f"{v:5.2f}"


def main():
    summaries: dict[float, dict] = {}
    for path in sorted(glob.glob(f"{SWEEP_DIR}/alpha_*/*__summary.json")):
        a = _alpha_from_path(path)
        with open(path) as f:
            summaries[a] = json.load(f)

    if not summaries:
        raise SystemExit(f"no summaries under {SWEEP_DIR}")

    alphas = sorted(summaries.keys())
    print(f"\n# Collapse map — model: {next(iter(summaries.values()))['model']}, "
          f"layer {next(iter(summaries.values()))['layer']}/"
          f"{next(iter(summaries.values()))['n_layers']}\n")
    print("Each row = (arm, attack). Columns = α. Numbers = (refusal | overref | utility | "
          "norm_ratio | max_entropy).\n")
    header = "arm     attack       " + "".join(f"  α={a:<5g}                                  " for a in alphas)
    print(header)

    for arm in ARMS:
        for attack in ATTACKS:
            row = f"{arm:7s} {attack:11s} "
            for a in alphas:
                s = summaries[a]
                key = f"{arm}|essence|{attack}" if arm != "caa" else f"caa|contrastive|{attack}"
                rec = s.get(key, {})
                fields = " ".join(_fmt(rec.get(k)) for k, _ in METRICS)
                row += f" {fields} "
            print(row)
        print()

    # Compact "collapse" detector: refusal floor < 0.10 at high α (vs same arm at α=8)
    print("## Collapse flags (refusal drop ≥ 0.20 vs α=8 baseline)\n")
    base = alphas[0]
    for arm in ARMS:
        for attack in ATTACKS:
            key = f"{arm}|essence|{attack}" if arm != "caa" else f"caa|contrastive|{attack}"
            r0 = summaries[base].get(key, {}).get("refusal_mean")
            if r0 is None:
                continue
            for a in alphas[1:]:
                r = summaries[a].get(key, {}).get("refusal_mean")
                if r is None:
                    continue
                if r0 - r >= 0.20:
                    print(f"  ⚠ {arm}|{attack}: refusal {r0:.2f} (α={base}) → {r:.2f} (α={a})")

    print("\n## Norm-ratio explosion (max norm_ratio_mean across attacks per α)\n")
    for arm in ARMS:
        for a in alphas:
            s = summaries[a]
            vals = []
            for attack in ATTACKS:
                key = f"{arm}|essence|{attack}" if arm != "caa" else f"caa|contrastive|{attack}"
                v = s.get(key, {}).get("norm_ratio_mean")
                if v is not None and v == v:
                    vals.append(v)
            mx = max(vals) if vals else None
            print(f"  arm={arm:7s} α={a:>5g}  max norm_ratio = {_fmt(mx)}")
        print()


if __name__ == "__main__":
    main()
