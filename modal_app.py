"""
Modal GPU runner for the Geometry-of-Control experiments.

Runs the full multi-arm / multi-attack / multi-seed experiment (goc.experiment)
on real GPU-backed models and fans out across model scales in parallel -- the
scaling study the reviewer asked for (0.5B -> 7B, multiple families).

Quick start
-----------
    pip install modal
    modal setup                      # one-time auth

    # Single model (the paper's original), real weights on GPU:
    modal run modal_app.py --models "Qwen/Qwen2.5-0.5B-Instruct" --seeds 5

    # Scaling + family study (each model gets its own GPU container):
    modal run modal_app.py \
        --models "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.2-3B-Instruct" \
        --seeds 5 --gpu A10G

Results land in a Modal Volume; pull them with:
    modal volume get geometry-of-control-results /results ./results
    python -m goc.analysis results/<model>__summary.json

Gated models (meta-llama/*) need an HF token:
    modal secret create huggingface HF_TOKEN=hf_xxx
    GOC_USE_HF_SECRET=1 modal run modal_app.py --models meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import json
import os

import modal

APP_NAME = "geometry-of-control"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers==4.57.3", "accelerate==1.12.0",
        "tokenizers==0.22.1", "huggingface_hub==0.36.0", "safetensors==0.7.0",
        "datasets==3.2.0", "regex==2025.11.3", "tqdm==4.67.1",
    )
    .add_local_python_source("goc")  # ship the experiment package into the image
)

hf_cache = modal.Volume.from_name("geometry-of-control-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("geometry-of-control-results", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=os.environ.get("GOC_GPU", "A10G"),
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]
    if os.environ.get("GOC_USE_HF_SECRET")
    else [],
    timeout=2 * 60 * 60,
)
def run_one_model(model_id: str, seeds: list[int], alpha: float, ds_gate: float,
                  layer_frac: float, max_new_tokens: int) -> dict:
    os.environ.setdefault("HF_HOME", "/cache/hf")
    from goc.experiment import RunConfig, run_experiment

    cfg = RunConfig(
        model_id=model_id,
        seeds=tuple(seeds),
        alpha=alpha,
        ds_gate=ds_gate,
        layer_frac=layer_frac,
        max_new_tokens=max_new_tokens,
        out_dir="/results",
    )
    summary = run_experiment(cfg)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(
    models: str = "Qwen/Qwen2.5-0.5B-Instruct",
    seeds: int = 5,
    alpha: float = 8.0,
    ds_gate: float = 3.0,
    layer_frac: float = 0.5,
    max_new_tokens: int = 64,
):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    seed_list = list(range(7, 7 + seeds))
    print(f"Models: {model_list}\nSeeds: {seed_list}\n")

    futures = [
        run_one_model.spawn(m, seed_list, alpha, ds_gate, layer_frac, max_new_tokens)
        for m in model_list
    ]
    for fut in futures:
        s = fut.get()
        print(f"\n===== {s['model']} (layer {s['layer']}/{s['n_layers']}, "
              f"cos(ess,con)={s['cos_essence_contrastive']:.2f}) =====")
        # show refusal under the prefix attack per arm (the hardest cell)
        for key, rec in sorted(s.items()):
            if isinstance(rec, dict) and key.endswith("|prefix"):
                print(f"  {key:32s} refusal={rec['refusal_mean']:.2f}±{rec['refusal_ci']:.2f} "
                      f"overref={rec['overrefusal_mean']:.2f} util={rec['utility_mean']:.2f}")
