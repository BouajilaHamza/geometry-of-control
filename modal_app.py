"""
Modal GPU runner for the Geometry-of-Control experiments.

Runs the full multi-arm / multi-attack / multi-seed experiment (goc.experiment)
on real GPU-backed models and fans out across model scales in parallel -- the
scaling study the reviewer asked for (0.5B -> 7B, multiple families).

Quick start
-----------
    pip install modal
    modal token set --token-id <id> --token-secret <secret>

    # Default: run tests on the GPU image first (TDD gate), then a real
    # 7B Qwen experiment across 5 seeds:
    modal run modal_app.py

    # Custom scope (each model gets its own GPU container):
    modal run modal_app.py \
        --models "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.2-3B-Instruct" \
        --seeds 5

    # Tests only (no experiment, no GPU $$ on a model load):
    modal run modal_app.py::run_tests

    # Skip the TDD gate (already validated this image):
    modal run modal_app.py --skip-tests

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
    .add_local_python_source("goc")
    .add_local_python_source("tests")
)

hf_cache = modal.Volume.from_name("geometry-of-control-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("geometry-of-control-results", create_if_missing=True)

app = modal.App(APP_NAME)


def _hf_secrets():
    return (
        [modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]
        if os.environ.get("GOC_USE_HF_SECRET")
        else []
    )


@app.function(
    image=image,
    gpu=os.environ.get("GOC_GPU", "A10G"),
    volumes={"/cache": hf_cache, "/results": results_vol},
    timeout=15 * 60,
)
def run_tests() -> dict:
    """TDD gate: run operator-correctness + end-to-end pipeline tests on the
    same GPU image used for the real experiment. Must pass before run_one_model
    is dispatched in main()."""
    import importlib
    import traceback

    results = {}
    for mod_name in ("tests.test_manifold_ops", "tests.test_pipeline"):
        try:
            mod = importlib.import_module(mod_name)
            mod.main() if hasattr(mod, "main") else _run_pytest_style(mod)
            results[mod_name] = {"status": "pass"}
        except AssertionError as e:
            results[mod_name] = {"status": "fail", "error": str(e),
                                  "trace": traceback.format_exc()}
        except Exception as e:
            results[mod_name] = {"status": "error", "error": repr(e),
                                  "trace": traceback.format_exc()}
    return results


def _run_pytest_style(mod):
    for name in dir(mod):
        if name.startswith("test_"):
            getattr(mod, name)()


@app.function(
    image=image,
    gpu=os.environ.get("GOC_GPU", "A10G"),
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=_hf_secrets(),
    timeout=2 * 60 * 60,
)
def run_alpha_sweep(model_id: str, alphas: list[float], layer_frac: float,
                    max_new_tokens: int, batch_size: int, ds_gate: float,
                    ds_alphas: tuple[float, ...] = (4.0, 8.0, 12.0)) -> dict:
    """Collapse-hunt: vary alpha across `add/renorm/ds_mix` at the same single
    seed, full attack ladder + items, and report where each operator falls
    apart. Model + tokenizer are loaded once and reused."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from goc.experiment import RunConfig, run_experiment

    os.environ.setdefault("HF_HOME", "/cache/hf")
    print(f"[sweep] loading {model_id} on cuda (bf16)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    summaries: dict[str, dict] = {}
    for a in alphas:
        print(f"\n=== alpha = {a} ===", flush=True)
        cfg = RunConfig(
            model_id=model_id,
            arms=("add", "renorm", "ds_mix"),
            seeds=(7,),
            alpha=float(a),
            ds_alphas=tuple(float(x) for x in ds_alphas),
            ds_gate=ds_gate,
            layer_frac=layer_frac,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            out_dir=f"/results/alpha_sweep_fixed_hull/alpha_{a}",
        )
        s = run_experiment(cfg, model=model, tokenizer=tok)
        summaries[str(a)] = s
        results_vol.commit()
    return summaries


@app.function(
    image=image,
    gpu=os.environ.get("GOC_GPU", "A10G"),
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=_hf_secrets(),
    timeout=2 * 60 * 60,
)
def run_one_model(model_id: str, seeds: list[int], alpha: float, ds_gate: float,
                  layer_frac: float, max_new_tokens: int, batch_size: int) -> dict:
    os.environ.setdefault("HF_HOME", "/cache/hf")
    from goc.experiment import RunConfig, run_experiment

    cfg = RunConfig(
        model_id=model_id,
        seeds=tuple(seeds),
        alpha=alpha,
        ds_gate=ds_gate,
        layer_frac=layer_frac,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        out_dir="/results",
    )
    summary = run_experiment(cfg)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(
    models: str = "Qwen/Qwen2.5-7B-Instruct",
    seeds: int = 5,
    alpha: float = 8.0,
    ds_gate: float = 3.0,
    layer_frac: float = 0.5,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    skip_tests: bool = False,
):
    if not skip_tests:
        print("=== TDD gate: running tests on Modal image ===")
        test_report = run_tests.remote()
        for mod, rec in test_report.items():
            status = rec["status"]
            print(f"  [{status.upper():5s}] {mod}")
            if status != "pass":
                print(rec.get("trace", rec.get("error", "")))
        if any(r["status"] != "pass" for r in test_report.values()):
            raise SystemExit("Aborting experiment: tests did not pass on Modal image.")
        print("=== TDD gate passed ===\n")

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    seed_list = list(range(7, 7 + seeds))
    print(f"Models: {model_list}\nSeeds: {seed_list}\n")

    futures = [
        run_one_model.spawn(m, seed_list, alpha, ds_gate, layer_frac, max_new_tokens, batch_size)
        for m in model_list
    ]
    for fut in futures:
        s = fut.get()
        print(f"\n===== {s['model']} (layer {s['layer']}/{s['n_layers']}, "
              f"cos(ess,con)={s['cos_essence_contrastive']:.2f}) =====")
        for key, rec in sorted(s.items()):
            if isinstance(rec, dict) and key.endswith("|prefix"):
                print(f"  {key:32s} refusal={rec['refusal_mean']:.2f}±{rec['refusal_ci']:.2f} "
                      f"overref={rec['overrefusal_mean']:.2f} util={rec['utility_mean']:.2f}")


@app.function(
    image=image,
    gpu=os.environ.get("GOC_GPU", "A10G"),
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=_hf_secrets(),
    timeout=60 * 60,
)
def safety_guard_eval(model_id: str, layer_frac: float, anchor_gate: float,
                      max_new_tokens: int, seed: int,
                      n_harmful: int, n_benign: int, n_utility: int) -> dict:
    """Run the M1 (K-anchor mHC guard) vs incumbent guardrails (B0/B1/B2/B3)
    on the joint eval set. Saves per-item logs + per-guard KPI summary."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from goc.eval_guards import evaluate_all_guards
    from goc.steering import num_layers

    os.environ.setdefault("HF_HOME", "/cache/hf")
    print(f"[guard_eval] loading {model_id} on cuda (bf16)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    L = max(1, int(num_layers(model) * layer_frac))
    safe_id = model_id.replace("/", "__")
    out_dir = f"/results/safety_guard/{safe_id}"
    summary = evaluate_all_guards(
        model, tok, layer=L,
        n_harmful=n_harmful, n_benign=n_benign, n_utility=n_utility,
        anchor_gate=anchor_gate, max_new_tokens=max_new_tokens, seed=seed,
        out_dir=out_dir,
    )
    summary["model"] = model_id
    results_vol.commit()
    return summary


@app.local_entrypoint()
def safety_guards(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    layer_frac: float = 0.5,
    anchor_gate: float = 3.0,
    max_new_tokens: int = 64,
    seed: int = 7,
    n_harmful: int = 8,
    n_benign: int = 6,
    n_utility: int = 6,
):
    """Compare M1 (K-anchor mHC guard) against B0/B1/B2/B3 baselines on the
    joint eval set. Prints a KPI table at the end."""
    summary = safety_guard_eval.remote(
        model, layer_frac, anchor_gate, max_new_tokens, seed,
        n_harmful, n_benign, n_utility,
    )
    print("\n========== SAFETY GUARD KPI TABLE ==========")
    print(f"model={summary.get('model')} | layer={summary['layer']} | "
          f"K={n_harmful} anchors\n")
    hdr = ("guard           | R↑   OR↓  U↑   lat_s p95_s  tok/s  post_ms  composite")
    print(hdr); print("-" * len(hdr))
    for name, rec in summary["guards"].items():
        k = rec["kpi"]
        fmt = lambda v, w=4: ("  -  " if v is None else f"{v:.2f}".rjust(w))
        print(f"{name:15s} | {fmt(k['refusal_recall'])} "
              f"{fmt(k['overrefusal_rate'])} {fmt(k['utility_acc'])} "
              f"{fmt(k['latency_mean_s'], 5)} "
              f"{fmt(k['latency_p95_s'], 5)} "
              f"{fmt(k['throughput_tok_s'], 6)} "
              f"{fmt(k['post_overhead_ms_mean'], 6)} "
              f"{fmt(k['composite'])}")


@app.local_entrypoint()
def alpha_sweep(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    alphas: str = "8,16,32,64",
    ds_alphas: str = "4,8,12",
    layer_frac: float = 0.5,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    ds_gate: float = 3.0,
):
    """Collapse-hunt entrypoint. Reports refusal/overref/util/norm_ratio per
    (alpha, arm, attack). `alphas` drives add/renorm strength; `ds_alphas`
    is the FIXED ds_mix hull (not scaled with alpha) so the convex-hull bound
    can actually bite when alpha runs away."""
    alpha_list = [float(a.strip()) for a in alphas.split(",") if a.strip()]
    ds_alpha_tuple = tuple(float(a.strip()) for a in ds_alphas.split(",") if a.strip())
    print(f"alphas (add/renorm): {alpha_list}\n"
          f"ds_alphas (fixed ds_mix hull): {ds_alpha_tuple}\n"
          f"model: {model}\n")
    summaries = run_alpha_sweep.remote(model, alpha_list, layer_frac,
                                       max_new_tokens, batch_size, ds_gate,
                                       ds_alpha_tuple)

    print("\n========== COLLAPSE MAP ==========")
    print("alpha | arm     | attack     | refusal | overref | util  | norm_r | cka")
    print("------+---------+------------+---------+---------+-------+--------+------")
    for a, s in summaries.items():
        for key, rec in sorted(s.items()):
            if not (isinstance(rec, dict) and key.count("|") == 2):
                continue
            arm, _vec, attack = key.split("|")
            nr = rec.get("norm_ratio_mean")
            cka = rec.get("cka_mean")
            nr_s = f"{nr:.2f}" if isinstance(nr, (int, float)) else "  - "
            cka_s = f"{cka:.2f}" if isinstance(cka, (int, float)) else "  - "
            print(f"{float(a):5.0f} | {arm:7s} | {attack:10s} | "
                  f"{rec['refusal_mean']:6.2f}  | {rec['overrefusal_mean']:6.2f}  | "
                  f"{rec['utility_mean']:5.2f} | {nr_s:6s} | {cka_s}")
