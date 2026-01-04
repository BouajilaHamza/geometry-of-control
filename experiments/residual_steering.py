import argparse
import os
import re
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def render_prompt(tokenizer, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


def keyword_hits(text: str, keywords: list[str]) -> dict[str, int]:
    t = text.lower()
    hits: dict[str, int] = {}
    for kw in keywords:
        hits[kw] = len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", t))
    return hits


def get_layer_module(model, layer_idx: int):
    # Qwen/Llama-style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # GPT2-style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError("Unsupported model architecture: cannot locate transformer layers.")

def get_final_norm_module(model):
    # Qwen/Llama-style
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # GPT2-style
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise RuntimeError("Unsupported model architecture: cannot locate final norm module.")

def get_output_embeddings_weight(model) -> torch.Tensor:
    out = model.get_output_embeddings()
    if out is not None and hasattr(out, "weight"):
        return out.weight
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight
    raise RuntimeError("Could not locate output embedding weight.")


def extract_hidden_from_layer_output(output):
    # Some HF layers return tensor; others return tuple(hidden_states, ...)
    if isinstance(output, tuple):
        return output[0], "tuple"
    return output, "tensor"


def rewrap_layer_output(output, hidden_states_new, kind: str):
    if kind == "tuple":
        return (hidden_states_new,) + output[1:]
    return hidden_states_new


class ResidualSteerer:
    def __init__(self, model, tokenizer, layer_idx: int):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = int(layer_idx)
        self.steering_vector = None  # [1,1,d] on model.device
        self._handle = None
        self.hook_ms: list[float] = []

    def _capture_raw(self, text: str, *, capture_mode: str) -> torch.Tensor:
        """
        Runs `text` through the model and captures the layer output hidden states, then reduces to a [1,1,d] vector.
        capture_mode: mean | last
        """
        layer = get_layer_module(self.model, self.layer_idx)
        captured = []

        def capture_hook(_module, _inp, out):
            hs, _kind = extract_hidden_from_layer_output(out)
            captured.append(hs.detach())

        handle = layer.register_forward_hook(capture_hook)
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Failed to capture activations.")

        hs = captured[0]  # [1, seq, d] typically
        if hs.ndim != 3:
            raise RuntimeError(f"Unexpected hidden state shape: {tuple(hs.shape)}")

        if capture_mode == "mean":
            vec = hs.mean(dim=1, keepdim=True)
        elif capture_mode == "last":
            vec = hs[:, -1:, :]
        else:
            raise ValueError("capture_mode must be 'mean' or 'last'")

        # Normalize for stability
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return vec.to(self.model.device)

    def capture_vector(
        self,
        target_text: str,
        *,
        capture_mode: str = "mean",
        contrast: bool = False,
        neutral_text: str | None = None,
    ) -> torch.Tensor:
        """
        If contrast=True, compute v = normalize(v_target - v_neutral).
        This is often much stronger/more stable than using v_target directly.
        """
        v_target = self._capture_raw(target_text, capture_mode=capture_mode)
        if contrast:
            if not neutral_text:
                raise ValueError("neutral_text is required when contrast=True")
            v_neutral = self._capture_raw(neutral_text, capture_mode=capture_mode)
            v = v_target - v_neutral
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            v = v_target
        self.steering_vector = v
        return self.steering_vector

    def enable(self, strength: float):
        if self.steering_vector is None:
            raise RuntimeError("Call capture_vector() first.")
        if self._handle is not None:
            raise RuntimeError("Hook already enabled.")

        layer = get_layer_module(self.model, self.layer_idx)
        strength = float(strength)

        def inject_hook(_module, _inp, out):
            t0 = time.perf_counter()
            hs, kind = extract_hidden_from_layer_output(out)
            # hs: [batch, seq, d]
            vec = self.steering_vector
            if vec is not None:
                # Only steer the *current* position to avoid corrupting KV-cache history.
                hs2 = hs.clone()
                hs2[:, -1:, :] = hs2[:, -1:, :] + (vec * strength)
            else:
                hs2 = hs
            self.hook_ms.append((time.perf_counter() - t0) * 1000.0)
            return rewrap_layer_output(out, hs2, kind)

        self._handle = layer.register_forward_hook(inject_hook)

    def disable(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class MultiLayerResidualSteerer:
    """
    Convenience wrapper for injecting (potentially different) steering vectors across multiple layers.
    """

    def __init__(self, model, tokenizer, layer_indices: list[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = [ResidualSteerer(model, tokenizer, li) for li in layer_indices]

    def capture_vectors(self, target_text: str, *, capture_mode: str, contrast: bool, neutral_text: str | None):
        vecs = []
        for s in self.layers:
            vecs.append(
                s.capture_vector(
                    target_text,
                    capture_mode=capture_mode,
                    contrast=contrast,
                    neutral_text=neutral_text if contrast else None,
                )
            )
        return vecs

    def enable(self, strength: float):
        for s in self.layers:
            s.enable(strength)

    def disable(self):
        for s in self.layers:
            s.disable()

    def hook_times_ms(self) -> list[float]:
        out = []
        for s in self.layers:
            out.extend(s.hook_ms)
        return out

class FinalNormSteerer:
    """
    Injects a steering vector into the final hidden state stream (post-transformer, pre-lm_head)
    by hooking the model's final normalization module.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vector = None
        self._handle = None
        self.hook_ms: list[float] = []
        self.lexicon_token_debug: list[tuple[str, int, str]] = []

    def _capture_raw(self, text: str, *, capture_mode: str) -> torch.Tensor:
        mod = get_final_norm_module(self.model)
        captured = []

        def capture_hook(_module, _inp, out):
            # out: [batch, seq, d]
            captured.append(out.detach())

        handle = mod.register_forward_hook(capture_hook)
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Failed to capture final norm activations.")

        hs = captured[0]
        if capture_mode == "mean":
            vec = hs.mean(dim=1, keepdim=True)
        elif capture_mode == "last":
            vec = hs[:, -1:, :]
        else:
            raise ValueError("capture_mode must be 'mean' or 'last'")
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return vec.to(self.model.device)

    def capture_vector(
        self,
        target_text: str,
        *,
        capture_mode: str,
        contrast: bool,
        neutral_text: str | None,
    ) -> torch.Tensor:
        v_target = self._capture_raw(target_text, capture_mode=capture_mode)
        if contrast:
            if not neutral_text:
                raise ValueError("neutral_text is required when contrast=True")
            v_neutral = self._capture_raw(neutral_text, capture_mode=capture_mode)
            v = v_target - v_neutral
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            v = v_target
        self.steering_vector = v
        return v

    def capture_from_lexicon_tokens(
        self,
        lexicon_words: list[str],
        *,
        neg_sample: int = 5000,
        neg_seed: int = 0,
    ) -> torch.Tensor:
        """
        Build a steering direction directly in hidden space by averaging the LM head vectors
        for a lexicon of target words. This aligns with logits more directly than activation capture.
        """
        W = get_output_embeddings_weight(self.model).detach()  # [vocab, d]
        vocab_size = int(W.shape[0])
        token_ids: list[int] = []
        debug: list[tuple[str, int, str]] = []
        for w in lexicon_words:
            w = w.strip().lower()
            if not w:
                continue
            # Prefer tokens that decode exactly to " <word>"
            found = False
            for tok_id in range(vocab_size):
                piece = self.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                if piece.startswith(" ") and piece.strip().lower() == w:
                    token_ids.append(tok_id)
                    debug.append((w, tok_id, piece))
                    found = True
                    break
            if not found:
                # fallback: use tokenizer encoding and take first token id (best-effort)
                ids = self.tokenizer.encode(" " + w, add_special_tokens=False)
                if ids:
                    token_ids.append(int(ids[0]))
                    piece0 = self.tokenizer.decode([int(ids[0])], clean_up_tokenization_spaces=False)
                    debug.append((w, int(ids[0]), piece0))

        if not token_ids:
            raise RuntimeError("No token ids found for lexicon.")

        pos = W[torch.tensor(token_ids, dtype=torch.long, device=W.device)].mean(dim=0, keepdim=True)  # [1,d]

        # Optional: subtract an average "background word" vector to make this direction more discriminative.
        if neg_sample and neg_sample > 0:
            # Candidate negatives: tokens that look like normal words (leading space + ascii letters).
            rng = torch.Generator(device="cpu").manual_seed(int(neg_seed))
            candidates: list[int] = []
            lex_set = set(int(t) for t in token_ids)
            for tok_id in range(vocab_size):
                if tok_id in lex_set:
                    continue
                piece = self.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                if not piece.startswith(" "):
                    continue
                s = piece.strip()
                if not s:
                    continue
                if any(ord(ch) > 127 for ch in s):
                    continue
                if not any(("a" <= ch.lower() <= "z") for ch in s):
                    continue
                candidates.append(tok_id)
            if candidates:
                # Sample without replacement (best-effort)
                cand_t = torch.tensor(candidates, dtype=torch.long)
                n = min(int(neg_sample), int(cand_t.numel()))
                idx = torch.randperm(cand_t.numel(), generator=rng)[:n]
                neg_ids = cand_t[idx].to(device=W.device)
                neg = W[neg_ids].mean(dim=0, keepdim=True)  # [1,d]
            else:
                neg = torch.zeros_like(pos)
        else:
            neg = torch.zeros_like(pos)

        vec = (pos - neg).unsqueeze(0)  # [1,1,d]
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.steering_vector = vec.to(self.model.device)
        self.lexicon_token_debug = debug
        return self.steering_vector

    def enable(self, strength: float):
        if self.steering_vector is None:
            raise RuntimeError("Call capture_vector() first.")
        if self._handle is not None:
            raise RuntimeError("Hook already enabled.")
        strength = float(strength)
        mod = get_final_norm_module(self.model)

        def inject_hook(_module, _inp, out):
            t0 = time.perf_counter()
            hs = out
            vec = self.steering_vector
            if vec is not None:
                hs2 = hs.clone()
                hs2[:, -1:, :] = hs2[:, -1:, :] + (vec * strength)
            else:
                hs2 = hs
            self.hook_ms.append((time.perf_counter() - t0) * 1000.0)
            return hs2

        self._handle = mod.register_forward_hook(inject_hook)

    def disable(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def timed_generate(model, tokenizer, rendered_prompt: str, *, gen_kwargs: dict) -> tuple[str, dict]:
    t0 = time.perf_counter()
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    in_len = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    t1 = time.perf_counter()
    out_ids = out[0]
    gen_tokens = int(max(0, out_ids.shape[-1] - in_len))
    sec = float(t1 - t0)
    tps = (gen_tokens / sec) if sec > 0 else float("nan")
    text = tokenizer.decode(out_ids[in_len:], skip_special_tokens=True)
    return text, {"seconds": sec, "gen_tokens": gen_tokens, "tokens_per_sec": tps}

def topk_tokens_from_logits(tokenizer, logits_1d: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
    vals, idxs = torch.topk(logits_1d, k=k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist(), strict=False):
        piece = tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)
        out.append((piece, float(v)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5: internal manifold injection via residual steering.")
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument(
        "--prompt",
        default="Write a fast-paced action scene about a medieval knight fighting a dragon.",
    )
    ap.add_argument(
        "--target",
        default="synergy leverage agile deliverables Q3 roadmap stakeholders bandwidth optimization strategy alignment",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.12)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--layer", type=int, default=-1, help="Single layer index (default: middle layer).")
    ap.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer indices to inject (overrides --layer). Example: 18,19,20,21",
    )
    ap.add_argument(
        "--site",
        choices=["layers", "final_norm"],
        default="final_norm",
        help="Where to inject: transformer layers or final norm (pre-lm_head).",
    )
    ap.add_argument(
        "--vector-source",
        choices=["activation", "lexicon_tokens"],
        default="lexicon_tokens",
        help="How to construct steering vector: captured activations or LM-head lexicon token vectors.",
    )
    ap.add_argument(
        "--diagnose-logits",
        action="store_true",
        default=True,
        help="Print a logits-diff sanity check at prompt end (final_norm only).",
    )
    ap.add_argument("--no-diagnose-logits", dest="diagnose_logits", action="store_false")
    ap.add_argument("--diag-strength", type=float, default=10.0, help="Steering strength used in logits diagnostic.")
    ap.add_argument("--neg-sample", type=int, default=5000, help="(lexicon_tokens) negative sample size for contrast.")
    ap.add_argument("--neg-seed", type=int, default=0, help="(lexicon_tokens) random seed for negative sampling.")
    ap.add_argument("--capture", choices=["mean", "last"], default="mean")
    ap.add_argument("--strengths", default="0,0.15,0.30,0.45", help="Comma-separated strengths; 0 is baseline.")
    ap.add_argument("--contrast", action="store_true", default=True, help="Use contrastive steering vector (recommended).")
    ap.add_argument("--no-contrast", dest="contrast", action="store_false")
    ap.add_argument(
        "--neutral",
        default="I went for a walk today and enjoyed the weather. It was calm and ordinary.",
        help="Neutral text used to compute contrastive steering direction.",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to("cpu")
    model.eval()

    # Determine layer count and default middle layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n_layers = len(model.transformer.h)
    else:
        raise RuntimeError("Unsupported model architecture for layer counting.")

    if args.layers.strip():
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layer_indices = [args.layer if args.layer >= 0 else (n_layers // 2)]

    rendered = render_prompt(tokenizer, args.prompt)
    lexicon = [
        "synergy",
        "leverage",
        "agile",
        "deliverables",
        "roadmap",
        "stakeholders",
        "bandwidth",
        "optimization",
        "strategy",
        "alignment",
        "pipeline",
        "deadline",
        "kpi",
        "okr",
        "roi",
        "q3",
    ]
    narrative = ["knight", "dragon", "sword", "shield", "castle", "fire", "battle"]

    strengths = [float(x.strip()) for x in args.strengths.split(",") if x.strip()]
    if not strengths or strengths[0] != 0.0:
        raise SystemExit("Provide --strengths with leading 0 for baseline, e.g. 0,0.2,0.4")

    print(f"Model: {args.model}")
    print(f"Layers: {n_layers} | injection_site={args.site} | capture={args.capture}")
    if args.site == "layers":
        print(f"Injection layers: {layer_indices}")
    print(f"Contrastive vector: {args.contrast}")
    print(f"Prompt: {args.prompt}")
    print(f"Target (steering source): {args.target}")
    if args.contrast:
        print(f"Neutral (steering baseline): {args.neutral}")
    print(f"Strengths: {strengths}")
    print()

    if args.site == "layers":
        steerer = MultiLayerResidualSteerer(model, tokenizer, layer_indices)
        vecs = steerer.capture_vectors(
            args.target,
            capture_mode=args.capture,
            contrast=args.contrast,
            neutral_text=args.neutral if args.contrast else None,
        )
        print(
            "Captured steering vectors: "
            + ", ".join(
                f"layer{li}:shape={tuple(v.shape)} norm={float(v.norm().item()):.3f}"
                for li, v in zip(layer_indices, vecs)
            )
        )
    else:
        steerer = FinalNormSteerer(model, tokenizer)
        if args.vector_source == "activation":
            v = steerer.capture_vector(
                args.target,
                capture_mode=args.capture,
                contrast=args.contrast,
                neutral_text=args.neutral if args.contrast else None,
            )
            print(
                f"Captured final-norm activation steering vector: shape={tuple(v.shape)} norm={float(v.norm().item()):.3f}"
            )
        else:
            v = steerer.capture_from_lexicon_tokens(lexicon, neg_sample=args.neg_sample, neg_seed=args.neg_seed)
            print(
                f"Captured final-norm lexicon-token steering vector: shape={tuple(v.shape)} norm={float(v.norm().item()):.3f}"
            )
            print("Lexicon token mapping used for steering (word -> token_id:piece):")
            for w, tid, piece in steerer.lexicon_token_debug:
                print(f"  {w} -> {tid}:{piece!r}")
    print()

    # Sanity-check that steering changes logits in the intended direction (final_norm only).
    if args.site == "final_norm" and args.diagnose_logits and isinstance(steerer, FinalNormSteerer):
        corp_tok_ids = [tid for _w, tid, _piece in getattr(steerer, "lexicon_token_debug", [])]
        corp_tok_ids = [int(t) for t in corp_tok_ids if isinstance(t, int)]
        if corp_tok_ids:
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.no_grad():
                base_out = model(**inputs)
                base_logits = base_out.logits[0, -1, :].detach().cpu()
            steerer.disable()
            steerer.hook_ms.clear()
            steerer.enable(float(args.diag_strength))
            with torch.no_grad():
                steered_out = model(**inputs)
                steered_logits = steered_out.logits[0, -1, :].detach().cpu()
            steerer.disable()

            print("=== Logits diagnostic (prompt next-token) ===")
            print("Top-10 base next tokens:", topk_tokens_from_logits(tokenizer, base_logits, k=10))
            print("Top-10 steered next tokens:", topk_tokens_from_logits(tokenizer, steered_logits, k=10))
            deltas = []
            for tid in sorted(set(corp_tok_ids)):
                piece = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
                deltas.append((piece, float((steered_logits[tid] - base_logits[tid]).item())))
            deltas_sorted = sorted(deltas, key=lambda kv: -kv[1])[:12]
            print("Largest corp-token logit deltas (steered - base):", deltas_sorted)
            print()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    results = []
    for strength in strengths:
        steerer.disable()
        # clear hook timing buffers
        if isinstance(steerer, MultiLayerResidualSteerer):
            _ = steerer.hook_times_ms()
            for s in steerer.layers:
                s.hook_ms.clear()
        else:
            steerer.hook_ms.clear()
        if strength != 0.0:
            steerer.enable(strength)
        text, timing = timed_generate(model, tokenizer, rendered, gen_kwargs=gen_kwargs)
        hits_corp = keyword_hits(text, lexicon)
        hits_narr = keyword_hits(text, narrative)
        if isinstance(steerer, MultiLayerResidualSteerer):
            hook_ms = steerer.hook_times_ms()
        else:
            hook_ms = list(steerer.hook_ms)
        results.append((strength, text, timing, hits_corp, hits_narr, hook_ms))

    # Print summaries
    print("=== Inference time (critical) ===")
    for strength, _text, timing, hits_c, hits_n, hook_ms in results:
        corp_total = sum(hits_c.values())
        narr_total = sum(hits_n.values())
        hook_over = ""
        if hook_ms:
            hook_over = f" | hook_avg_ms={statistics.mean(hook_ms):.3f} p95_ms={sorted(hook_ms)[int(0.95*len(hook_ms))-1]:.3f}"
        print(
            f"strength={strength:g}: seconds={timing['seconds']:.2f} gen_tokens={timing['gen_tokens']} "
            f"tok/s={timing['tokens_per_sec']:.2f} | corp_hits={corp_total} narr_hits={narr_total}{hook_over}"
        )
    print()

    for strength, text, _timing, hits_c, hits_n, _hook_ms in results:
        corp_top = sorted(((k, v) for k, v in hits_c.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))[:8]
        narr_top = sorted(((k, v) for k, v in hits_n.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))[:8]
        print(f"--- strength={strength:g} ---")
        print(f"corp_top={corp_top}")
        print(f"narr_top={narr_top}")
        print(text.strip() or "<EMPTY>")
        print()

    steerer.disable()


if __name__ == "__main__":
    torch.set_num_threads(max(1, int(os.environ.get('TORCH_NUM_THREADS', '2'))))
    main()

