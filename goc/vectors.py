"""
Steering-vector construction.

Answers the reviewer's first question ("how is v constructed?") explicitly by
providing two named methods and letting experiments compare them:

  essence     : mean activation of a single hand-written refusal sentence,
                normalized. This is what the original paper did -- kept as a
                baseline so we can quantify how much it costs vs. the standard.

  contrastive : difference-of-means between activations on harmful-prompt vs.
                harmless-prompt sets (CAA / Arditi-style). This is the
                field-standard refusal direction and the recommended default.

Both return a unit vector [1, 1, d] at the requested layer.
"""

from __future__ import annotations

import torch

from goc.steering import get_layer_module, render_prompt


def _mean_layer_activation(model, tokenizer, text: str, layer_idx: int, *, chat: bool) -> torch.Tensor:
    captured = []

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured.append(hs.detach())

    rendered = render_prompt(tokenizer, text) if chat else text
    h = get_layer_module(model, layer_idx).register_forward_hook(hook)
    try:
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        h.remove()
    return captured[0].mean(dim=1, keepdim=True)  # [1,1,d] mean over tokens


def essence_vector(model, tokenizer, refusal_text: str, layer_idx: int) -> torch.Tensor:
    """Original-paper method: normalized mean activation of one refusal sentence."""
    vec = _mean_layer_activation(model, tokenizer, refusal_text, layer_idx, chat=False)
    return (vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(model.device)


def contrastive_vector(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer_idx: int,
) -> torch.Tensor:
    """
    CAA / difference-of-means refusal direction:
        v = normalize( mean_act(harmful) - mean_act(harmless) )
    Computed on chat-rendered prompts at the last token, averaged over the set.
    """
    def set_mean(prompts: list[str]) -> torch.Tensor:
        acc = None
        for p in prompts:
            a = _mean_layer_activation(model, tokenizer, p, layer_idx, chat=True)  # [1,1,d]
            acc = a if acc is None else acc + a
        return acc / max(1, len(prompts))

    h_harm = set_mean(harmful_prompts)
    h_safe = set_mean(harmless_prompts)
    diff = h_harm - h_safe
    return (diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)).to(model.device)


def build_vector(method: str, *, model, tokenizer, layer_idx: int, refusal_text: str = "",
                 harmful_prompts: list[str] | None = None,
                 harmless_prompts: list[str] | None = None) -> torch.Tensor:
    if method == "essence":
        return essence_vector(model, tokenizer, refusal_text, layer_idx)
    if method == "contrastive":
        return contrastive_vector(model, tokenizer, harmful_prompts or [], harmless_prompts or [], layer_idx)
    raise ValueError("method must be 'essence' or 'contrastive'")
