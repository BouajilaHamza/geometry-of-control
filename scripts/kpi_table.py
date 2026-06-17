"""Render the safety-guard KPI comparison from a guard_eval.json.

Usage:
    python scripts/kpi_table.py results/safety_guard/Qwen__Qwen2.5-7B-Instruct/guard_eval.json
"""

from __future__ import annotations

import json
import sys


def fmt(v, w=4):
    if v is None:
        return "  -  "
    return f"{v:.2f}".rjust(w)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/safety_guard/guard_eval.json"
    with open(path) as f:
        summary = json.load(f)

    print(f"# Safety guard KPI comparison\n")
    print(f"model: {summary.get('model','?')}    layer: {summary['layer']}    "
          f"items: {summary['n_items']}\n")

    cols = ("guard", "R↑", "OR↓", "U↑", "lat_mean_s", "lat_p95_s",
            "tok/s", "post_ms", "composite↑")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for name, rec in summary["guards"].items():
        k = rec["kpi"]
        row = [
            name,
            fmt(k["refusal_recall"]),
            fmt(k["overrefusal_rate"]),
            fmt(k["utility_acc"]),
            fmt(k["latency_mean_s"], 5),
            fmt(k["latency_p95_s"], 5),
            fmt(k["throughput_tok_s"], 6),
            fmt(k["post_overhead_ms_mean"], 6),
            fmt(k["composite"]),
        ]
        print("| " + " | ".join(row) + " |")

    # Pairwise deltas vs B0 (no guard).
    base = summary["guards"].get("b0_none", {}).get("kpi")
    if base:
        print("\n## Δ vs B0 (no guard)\n")
        print("| guard | ΔR | ΔOR | ΔU | Δlatency_s | Δcomposite |")
        print("|---|---|---|---|---|---|")
        for name, rec in summary["guards"].items():
            if name == "b0_none":
                continue
            k = rec["kpi"]
            def d(a, b, w=4):
                if a is None or b is None:
                    return "  -  "
                return f"{a - b:+.2f}".rjust(w)
            print(f"| {name} | "
                  f"{d(k['refusal_recall'], base['refusal_recall'])} | "
                  f"{d(k['overrefusal_rate'], base['overrefusal_rate'])} | "
                  f"{d(k['utility_acc'], base['utility_acc'])} | "
                  f"{d(k['latency_mean_s'], base['latency_mean_s'], 6)} | "
                  f"{d(k['composite'], base['composite'])} |")


if __name__ == "__main__":
    main()
