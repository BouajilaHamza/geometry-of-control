"""
Turn a summary.json into reviewer-facing comparison tables.

Usage:
    python -m goc.analysis results/<model>__summary.json
    python -m goc.analysis results/<model>__summary.json --md report.md

Produces two tables:
  1. Safety/helpfulness by arm x attack: refusal (safer=higher), over-refusal
     (tax=lower), utility (capability=higher), all with 95% CIs.
  2. Manifold + cost by arm: max norm-ratio, mean CKA, min cosine, hook ms.
This is the head-to-head that shows whether ds_mix earns its place over the
add / renorm / caa baselines.
"""

from __future__ import annotations

import json
import sys


def _fmt(m, ci):
    if m != m:  # nan
        return "  -  "
    return f"{m:.2f}±{ci:.2f}"


def safety_table(summary: dict) -> str:
    cells = {k: v for k, v in summary.items() if isinstance(v, dict) and "|" in k}
    arms, attacks = [], []
    for k in cells:
        arm, _vec, atk = k.split("|")
        if arm not in arms:
            arms.append(arm)
        if atk not in attacks:
            attacks.append(atk)

    lines = [f"## Safety vs. helpfulness  —  {summary.get('model','?')} "
             f"(layer {summary.get('layer')}/{summary.get('n_layers')})", ""]
    for metric, want in (("refusal", "↑ safer"), ("overrefusal", "↓ tax"), ("utility", "↑ kept")):
        lines.append(f"### {metric}  ({want})")
        lines.append("| arm | " + " | ".join(attacks) + " |")
        lines.append("|" + "---|" * (len(attacks) + 1))
        for arm in arms:
            row = [arm]
            for atk in attacks:
                # prefer contrastive vector cell if present, else essence
                rec = cells.get(f"{arm}|contrastive|{atk}") or cells.get(f"{arm}|essence|{atk}")
                row.append(_fmt(rec[f"{metric}_mean"], rec[f"{metric}_ci"]) if rec else " - ")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def manifold_table(summary: dict) -> str:
    cells = {k: v for k, v in summary.items() if isinstance(v, dict) and "|" in k}
    # collapse over attacks: report the cell with attack 'none' (or any) per arm
    by_arm: dict[str, dict] = {}
    for k, rec in cells.items():
        arm, _vec, atk = k.split("|")
        if rec.get("norm_ratio_n", 0) > 0 and arm not in by_arm:
            by_arm[arm] = rec

    lines = ["## Manifold geometry + cost (steered vs. unsteered, harmful prompts)", "",
             "| arm | max norm-ratio | mean CKA | min cosine | hook ms |",
             "|---|---|---|---|---|"]
    for arm, rec in by_arm.items():
        lines.append(
            f"| {arm} | {_fmt(rec['norm_ratio_mean'], rec['norm_ratio_ci'])} "
            f"| {_fmt(rec['cka_mean'], rec['cka_ci'])} "
            f"| {_fmt(rec['cosine_mean'], rec['cosine_ci'])} "
            f"| {rec['hook_ms_mean']:.3f} |" if rec.get('hook_ms_mean') else f"| {arm} | ... |")
    lines.append("")
    lines.append(f"_norm-ratio→1 and CKA→1 mean the steered state stays on the "
                 f"original manifold; ds_mix is convex-hull bounded by construction._")
    return "\n".join(lines)


def render(summary: dict) -> str:
    return safety_table(summary) + "\n" + manifold_table(summary)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    summary = json.load(open(sys.argv[1]))
    md = render(summary)
    if "--md" in sys.argv:
        out = sys.argv[sys.argv.index("--md") + 1]
        open(out, "w").write(md)
        print(f"wrote {out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
