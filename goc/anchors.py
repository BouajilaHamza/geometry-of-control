"""
Domain-anchor construction for the K-anchor mHC guard.

The anchors are K hidden states at a chosen layer captured during the model's
*natural refusal continuation* of harmful prompts in a calibration set. They
empirically span the "safety-domain" region of the residual stream; at
inference, the mHC operator projects the current state onto the convex hull
of (current + anchors), so the emitted activation cannot leave the cloud.

This replaces the 1-D refusal direction used by the prior steering arms with a
K-dimensional simplex in d-dim activation space.
"""

from __future__ import annotations

import torch

from goc.steering import get_layer_module, render_prompt


CANONICAL_REFUSAL = (
    " I cannot help with that. It violates my safety guidelines, and I will not "
    "provide instructions for this. If you have a different request I can help with, let me know."
)


def _capture_layer_states(model, tokenizer, text: str, layer_idx: int,
                          *, skip_first: int) -> torch.Tensor:
    """Forward `text`, capture the layer's hidden states, return [T_skip, d]
    where T_skip is the suffix after `skip_first` prompt tokens."""
    grabbed: list[torch.Tensor] = []

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        grabbed.append(hs.detach())

    h = get_layer_module(model, layer_idx).register_forward_hook(hook)
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        h.remove()
    seq = grabbed[0][0]                       # [T, d]
    return seq[skip_first:]                   # [T_resp, d]


def build_safety_anchors(model, tokenizer, harmful_prompts: list[str],
                         layer_idx: int) -> torch.Tensor:
    """
    For each harmful prompt, render (chat-formatted prompt + canonical refusal
    response), capture the residual-stream hidden state at `layer_idx` over the
    response tokens only, mean-pool to one vector. Returns Tensor[K, d] with K
    = len(harmful_prompts).
    """
    anchors: list[torch.Tensor] = []
    for prompt in harmful_prompts:
        rendered = render_prompt(tokenizer, prompt)
        prompt_len = tokenizer(rendered, return_tensors="pt")["input_ids"].shape[-1]
        full = rendered + CANONICAL_REFUSAL
        seq = _capture_layer_states(model, tokenizer, full, layer_idx,
                                    skip_first=prompt_len)
        if seq.shape[0] == 0:
            seq = grabbed_fallback(model, tokenizer, full, layer_idx)
        anchor = seq.mean(dim=0)              # [d]
        anchors.append(anchor)
    A = torch.stack(anchors, dim=0)           # [K, d]
    return A.to(model.device).to(next(model.parameters()).dtype)


def grabbed_fallback(model, tokenizer, text, layer_idx):
    return _capture_layer_states(model, tokenizer, text, layer_idx, skip_first=0)
