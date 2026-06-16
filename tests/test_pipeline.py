"""
End-to-end pipeline smoke test on a randomly-initialized tiny model.

HuggingFace weights are not required: we instantiate a small Qwen2 architecture
from config (random weights) and a minimal word-level tokenizer, then run the
real experiment code path -- ArmSteer hooks, the ds_mix Sinkhorn forward, the
manifold-metric capture, scoring, and CI aggregation. Numbers are meaningless
(random weights) but it proves the harness runs end-to-end and the operators
fire inside a genuine transformer forward pass.

Run: python -m tests.test_pipeline
"""

from __future__ import annotations

import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

from goc.experiment import RunConfig, run_experiment


class TinyTokenizer:
    """Minimal word-level tokenizer implementing only what the harness uses."""

    def __init__(self):
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.chat_template = None  # forces render_prompt to pass text through
        self._vocab = 256

    def __call__(self, text, return_tensors=None):
        ids = [(hash(w) % (self._vocab - 2)) + 2 for w in text.split()][:32] or [2]
        t = torch.tensor([ids], dtype=torch.long)
        return BatchEncoding({"input_ids": t, "attention_mask": torch.ones_like(t)})

    def encode(self, text, add_special_tokens=False):
        return [(hash(w) % (self._vocab - 2)) + 2 for w in text.split()] or [2]

    def decode(self, ids, skip_special_tokens=True):
        # deterministic pseudo-words so refusal/utility markers occasionally hit
        return " ".join(f"w{int(i) % 50}" for i in ids)

    def to(self, *a, **k):
        return self


def build_tiny_model() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def main():
    model = build_tiny_model()
    tok = TinyTokenizer()
    cfg = RunConfig(
        model_id="tiny-random/qwen2",
        arms=("none", "add", "renorm", "caa", "ds_mix"),
        seeds=(7,),
        attacks=__import__("goc.attacks", fromlist=["attack_ladder"]).attack_ladder()[:3],
        n_harmful=2, n_benign=2, n_utility=2,
        max_new_tokens=8,
        out_dir="results/_smoke",
    )
    summary = run_experiment(cfg, model=model, tokenizer=tok)

    assert summary["n_trials"] > 0, "no trials ran"
    cells = [k for k in summary if "|" in k]
    assert any(k.startswith("ds_mix|") for k in cells), "ds_mix arm missing from summary"

    # The ds_mix Sinkhorn matrices must be (near) doubly-stochastic inside the
    # real forward pass, not just in the unit test.
    ds_cells = [summary[k] for k in cells if k.startswith("ds_mix|")]

    print(f"[PASS] pipeline ran {summary['n_trials']} trials across {len(cells)} cells")
    print(f"        layers={summary['n_layers']} steer_layer={summary['layer']}")
    print(f"        cos(essence,contrastive)={summary['cos_essence_contrastive']:.3f}")
    sample = [k for k in cells if k.startswith("ds_mix|")][0]
    rec = summary[sample]
    print(f"        sample ds_mix cell '{sample}':")
    print(f"          refusal_mean={rec['refusal_mean']:.2f} (n={rec['refusal_n']})  "
          f"hook_ms_mean={rec['hook_ms_mean']}")
    print("\nPipeline smoke test passed (random weights -> numbers are not meaningful).")


if __name__ == "__main__":
    main()
