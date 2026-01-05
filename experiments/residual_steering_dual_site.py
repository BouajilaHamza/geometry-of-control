import argparse
import os
import re
import statistics
import time
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, set_seed


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


def get_layer_module(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError("Unsupported model architecture: cannot locate transformer layers.")


def get_final_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
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


def keyword_hits(text: str, keywords: list[str]) -> dict[str, int]:
    t = text.lower()
    hits: dict[str, int] = {}
    for kw in keywords:
        hits[kw] = len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", t))
    return hits


def entropy_from_logits(logits_1d: torch.Tensor) -> float:
    probs = torch.softmax(logits_1d, dim=-1).clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


def is_ascii_wordpiece(piece: str) -> bool:
    if not piece:
        return False
    if not piece.startswith(" "):
        return False
    s = piece.strip()
    if not s:
        return False
    if any(ord(ch) > 127 for ch in s):
        return False
    return any(("a" <= ch.lower() <= "z") for ch in s)


def build_lexicon_token_ids(tokenizer, vocab_size: int, lexicon_words: list[str]) -> tuple[list[int], list[tuple[str, int, str]]]:
    token_ids: list[int] = []
    debug: list[tuple[str, int, str]] = []
    for w in lexicon_words:
        w = w.strip().lower()
        if not w:
            continue
        found = False
        for tok_id in range(vocab_size):
            piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            if piece.startswith(" ") and piece.strip().lower() == w:
                token_ids.append(tok_id)
                debug.append((w, tok_id, piece))
                found = True
                break
        if not found:
            ids = tokenizer.encode(" " + w, add_special_tokens=False)
            if ids:
                tok0 = int(ids[0])
                token_ids.append(tok0)
                piece0 = tokenizer.decode([tok0], clean_up_tokenization_spaces=False)
                debug.append((w, tok0, piece0))
    seen = set()
    out = []
    for tid in token_ids:
        if tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out, debug


def build_discriminative_lexicon_direction(
    model,
    tokenizer,
    lexicon_ids: list[int],
    *,
    neg_sample: int,
    neg_seed: int,
    rounds: int,
    avoid_n: int,
) -> torch.Tensor:
    W = get_output_embeddings_weight(model).detach()
    vocab_size = int(W.shape[0])
    d = int(W.shape[1])

    pos = W[torch.tensor(lexicon_ids, dtype=torch.long, device=W.device)].mean(dim=0, keepdim=True)  # [1,d]

    # background average
    if neg_sample > 0:
        candidates: list[int] = []
        lex_set = set(int(t) for t in lexicon_ids)
        for tok_id in range(vocab_size):
            if tok_id in lex_set:
                continue
            piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            if is_ascii_wordpiece(piece):
                candidates.append(tok_id)
        if candidates:
            rng = torch.Generator(device="cpu").manual_seed(int(neg_seed))
            cand_t = torch.tensor(candidates, dtype=torch.long)
            n = min(int(neg_sample), int(cand_t.numel()))
            idx = torch.randperm(cand_t.numel(), generator=rng)[:n]
            neg_ids = cand_t[idx].to(device=W.device)
            neg = W[neg_ids].mean(dim=0, keepdim=True)
        else:
            neg = torch.zeros_like(pos)
    else:
        neg = torch.zeros_like(pos)

    vv = (pos - neg).reshape(d)
    lex_set = set(lexicon_ids)

    def orthogonalize(v, avoid_ids):
        for tid in avoid_ids:
            w = W[tid]
            denom = float((w @ w).item())
            if denom <= 1e-12:
                continue
            proj = float((v @ w).item()) / denom
            v = v - proj * w
        return v

    for _ in range(max(0, int(rounds))):
        scores = (W @ vv).detach().cpu()
        _, idxs = torch.topk(scores, k=min(250, scores.numel()))
        avoid = []
        for tid in idxs.tolist():
            if tid in lex_set:
                continue
            piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            if (not is_ascii_wordpiece(piece)) or (
                piece.strip()
                in {"The", "As", "I", "In", "A", "It", "Once", "INT", "**", "[", "]", "Scene", "Knight", "Suddenly"}
            ):
                avoid.append(tid)
            if len(avoid) >= int(avoid_n):
                break
        if not avoid:
            break
        vv = orthogonalize(vv, avoid)

    vv = vv / vv.norm().clamp_min(1e-8)
    return vv.unsqueeze(0).unsqueeze(0)  # [1,1,d]


def capture_layer_mean_vector(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    layer = get_layer_module(model, layer_idx)
    captured = []

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured.append(hs.detach())

    h = layer.register_forward_hook(hook)
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        h.remove()
    if not captured:
        raise RuntimeError("Failed to capture activations.")
    hs = captured[0]  # [1, seq, d]
    vec = hs.mean(dim=1, keepdim=True)  # [1,1,d]
    vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return vec.to(model.device)


class Injector:
    def __init__(self):
        self.hook_ms: list[float] = []

    def enable(self):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError


class LayerInjector(Injector):
    def __init__(self, model, layer_idx: int, v: torch.Tensor, strength: float):
        super().__init__()
        self.model = model
        self.layer_idx = int(layer_idx)
        self.v = v.to(model.device)  # [1,1,d]
        self.strength = float(strength)
        self._handle = None

    def enable(self):
        if self._handle is not None:
            raise RuntimeError("Already enabled.")
        layer = get_layer_module(self.model, self.layer_idx)

        def hook(_m, _inp, out):
            t0 = time.perf_counter()
            hs, kind = (out[0], "tuple") if isinstance(out, tuple) else (out, "tensor")
            hs2 = hs.clone()
            hs2[:, -1:, :] = hs2[:, -1:, :] + (self.v * self.strength)
            self.hook_ms.append((time.perf_counter() - t0) * 1000.0)
            if kind == "tuple":
                return (hs2,) + out[1:]
            return hs2

        self._handle = layer.register_forward_hook(hook)

    def disable(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class FinalNormInjector(Injector):
    def __init__(self, model, v: torch.Tensor, strength: float):
        super().__init__()
        self.model = model
        self.v = v.to(model.device)
        self.current_strength = float(strength)
        self._handle = None

    def enable(self):
        if self._handle is not None:
            raise RuntimeError("Already enabled.")
        mod = get_final_norm_module(self.model)

        def hook(_m, _inp, out):
            t0 = time.perf_counter()
            hs = out
            hs2 = hs.clone()
            hs2[:, -1:, :] = hs2[:, -1:, :] + (self.v * float(self.current_strength))
            self.hook_ms.append((time.perf_counter() - t0) * 1000.0)
            return hs2

        self._handle = mod.register_forward_hook(hook)

    def disable(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class Controller(LogitsProcessor):
    """
    Adjusts final_norm strength using corp-rate + (optional) entropy floor.
    Returns logits unchanged.
    """

    def __init__(
        self,
        final_injector: FinalNormInjector,
        *,
        lexicon_token_ids: set[int],
        target_corp_rate: float,
        window: int,
        k_rate: float,
        entropy_floor: float,
        k_ent: float,
        strength_min: float,
        strength_max: float,
    ):
        super().__init__()
        self.final = final_injector
        self.lex = set(int(x) for x in lexicon_token_ids)
        self.target_corp_rate = float(target_corp_rate)
        self.k_rate = float(k_rate)
        self.entropy_floor = float(entropy_floor)
        self.k_ent = float(k_ent)
        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        self.corp_window = deque(maxlen=int(window))
        self.entropy_hist: list[float] = []
        self.str_hist: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 1:
            return scores
        last_id = int(input_ids[0, -1].item())
        self.corp_window.append(1 if last_id in self.lex else 0)
        corp_rate = sum(self.corp_window) / len(self.corp_window)

        H = entropy_from_logits(scores[0].detach())
        self.entropy_hist.append(H)

        ds = self.k_rate * (self.target_corp_rate - corp_rate)
        if H < self.entropy_floor:
            ds -= self.k_ent * (self.entropy_floor - H)
        new_s = float(self.final.current_strength + ds)
        new_s = float(max(self.strength_min, min(self.strength_max, new_s)))
        self.final.current_strength = new_s
        self.str_hist.append(new_s)
        return scores


class LexiconLogitBias(LogitsProcessor):
    """
    Soft output shaping: adds a constant bias to a set of token ids.
    (No masking; just shifts probability mass.)
    """

    def __init__(self, token_ids: set[int], bias: float):
        super().__init__()
        self.token_ids = sorted(int(t) for t in token_ids)
        self.bias = float(bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.numel() == 0:
            return scores
        scores[:, self.token_ids] = scores[:, self.token_ids] + self.bias
        return scores


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-site residual steering: mid-layer topic + final-norm lexical pull.")
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--prompt", default="Write a fast-paced action scene about a medieval knight fighting a dragon.")
    ap.add_argument("--target", default="synergy leverage agile roadmap stakeholders bandwidth optimization strategy alignment")
    ap.add_argument("--neutral", default="I went for a walk today and enjoyed the weather. It was calm and ordinary.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=170)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.12)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)

    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--layer-strength", type=float, default=0.9)

    ap.add_argument("--final-strength-fixed", type=float, default=18.0)
    ap.add_argument("--final-strength-adaptive-start", type=float, default=12.0)
    ap.add_argument("--final-strength-min", type=float, default=0.0)
    ap.add_argument("--final-strength-max", type=float, default=30.0)
    ap.add_argument("--logit-bias", type=float, default=0.0, help="Optional soft logit bias for lexicon tokens.")
    ap.add_argument(
        "--ablation",
        action="store_true",
        default=True,
        help="Run ablation: baseline vs bias-only vs internal-only vs internal+bias.",
    )
    ap.add_argument("--no-ablation", dest="ablation", action="store_false")

    # lexicon direction
    ap.add_argument("--neg-sample", type=int, default=8000)
    ap.add_argument("--neg-seed", type=int, default=1)
    ap.add_argument("--purify-rounds", type=int, default=2)
    ap.add_argument("--avoid-n", type=int, default=50)

    # controller
    ap.add_argument("--target-corp-rate", type=float, default=0.08)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--k-rate", type=float, default=2.5)
    ap.add_argument("--entropy-floor", type=float, default=1.2)
    ap.add_argument("--k-ent", type=float, default=2.0)

    args = ap.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to("cpu")
    model.eval()

    rendered = render_prompt(tokenizer, args.prompt)

    W = get_output_embeddings_weight(model)
    vocab_size = int(W.shape[0])

    lexicon_words = [
        "synergy",
        "leverage",
        "agile",
        "roadmap",
        "stakeholders",
        "bandwidth",
        "optimization",
        "strategy",
        "alignment",
        "pipeline",
        "deadline",
        "roi",
    ]
    narrative_words = ["knight", "dragon", "sword", "shield", "castle", "fire", "battle"]

    lex_ids, lex_debug = build_lexicon_token_ids(tokenizer, vocab_size, lexicon_words)
    lex_set = set(lex_ids)

    print(f"Model: {args.model}")
    print(f"Injection layer: {args.layer} (strength={args.layer_strength}) + final_norm (fixed/adaptive)")
    print("Lexicon token mapping:")
    for w, tid, piece in lex_debug:
        print(f"  {w} -> {tid}:{piece!r}")
    print()

    # Mid-layer steering direction: contrastive activation vector
    v_pos = capture_layer_mean_vector(model, tokenizer, args.target, args.layer)
    v_neg = capture_layer_mean_vector(model, tokenizer, args.neutral, args.layer)
    v_layer = (v_pos - v_neg)
    v_layer = v_layer / v_layer.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Final-norm lexicon direction: discriminative W-space vector
    v_final = build_discriminative_lexicon_direction(
        model,
        tokenizer,
        lex_ids,
        neg_sample=args.neg_sample,
        neg_seed=args.neg_seed,
        rounds=args.purify_rounds,
        avoid_n=args.avoid_n,
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Baseline
    base_text, base_t = timed_generate(model, tokenizer, rendered, gen_kwargs=gen_kwargs)

    # Bias-only (no hooks)
    if args.ablation and args.logit_bias != 0.0:
        bias_only_text, bias_only_t = timed_generate(
            model,
            tokenizer,
            rendered,
            gen_kwargs={**gen_kwargs, "logits_processor": LogitsProcessorList([LexiconLogitBias(lex_set, args.logit_bias)])},
        )
    else:
        bias_only_text, bias_only_t = "", {"seconds": 0.0, "gen_tokens": 0, "tokens_per_sec": float("nan")}

    # Internal-only (hooks, no logits processor)
    layer_inj = LayerInjector(model, args.layer, v_layer, args.layer_strength)
    final_fixed = FinalNormInjector(model, v_final, args.final_strength_fixed)
    layer_inj.enable()
    final_fixed.enable()
    internal_only_text, internal_only_t = timed_generate(model, tokenizer, rendered, gen_kwargs=gen_kwargs)
    internal_only_hook = layer_inj.hook_ms + final_fixed.hook_ms
    layer_inj.disable()
    final_fixed.disable()

    # Internal + bias (same hooks + lexicon bias)
    if args.logit_bias != 0.0:
        layer_inj_b = LayerInjector(model, args.layer, v_layer, args.layer_strength)
        final_fixed_b = FinalNormInjector(model, v_final, args.final_strength_fixed)
        layer_inj_b.enable()
        final_fixed_b.enable()
        lp_fixed = LogitsProcessorList([LexiconLogitBias(lex_set, args.logit_bias)])
        fixed_text, fixed_t = timed_generate(model, tokenizer, rendered, gen_kwargs={**gen_kwargs, "logits_processor": lp_fixed})
        fixed_hook = layer_inj_b.hook_ms + final_fixed_b.hook_ms
        layer_inj_b.disable()
        final_fixed_b.disable()
    else:
        fixed_text, fixed_t, fixed_hook = "", {"seconds": 0.0, "gen_tokens": 0, "tokens_per_sec": float("nan")}, []

    # Adaptive dual-site
    layer_inj2 = LayerInjector(model, args.layer, v_layer, args.layer_strength)
    final_ad = FinalNormInjector(model, v_final, args.final_strength_adaptive_start)
    ctrl = Controller(
        final_ad,
        lexicon_token_ids=lex_set,
        target_corp_rate=args.target_corp_rate,
        window=args.window,
        k_rate=args.k_rate,
        entropy_floor=args.entropy_floor,
        k_ent=args.k_ent,
        strength_min=args.final_strength_min,
        strength_max=args.final_strength_max,
    )
    layer_inj2.enable()
    final_ad.enable()
    processors = [ctrl]
    if args.logit_bias != 0.0:
        processors.append(LexiconLogitBias(lex_set, args.logit_bias))
    ad_text, ad_t = timed_generate(model, tokenizer, rendered, gen_kwargs={**gen_kwargs, "logits_processor": LogitsProcessorList(processors)})
    ad_hook = layer_inj2.hook_ms + final_ad.hook_ms
    layer_inj2.disable()
    final_ad.disable()

    def summarize(text: str):
        corp = keyword_hits(text, lexicon_words)
        narr = keyword_hits(text, narrative_words)
        return sum(corp.values()), sum(narr.values()), sorted(((k, v) for k, v in corp.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))[:8]

    b_c, b_n, _ = summarize(base_text)
    bo_c, bo_n, bo_top = summarize(bias_only_text) if bias_only_text else (0, 0, [])
    io_c, io_n, io_top = summarize(internal_only_text)
    f_c, f_n, f_top = summarize(fixed_text) if fixed_text else (0, 0, [])
    a_c, a_n, a_top = summarize(ad_text)

    print("=== Inference time (critical) ===")
    rows = [("BASELINE", base_t, [], b_c, b_n)]
    if args.ablation and args.logit_bias != 0.0:
        rows.append((f"BIAS_ONLY(bias={args.logit_bias:g})", bias_only_t, [], bo_c, bo_n))
    rows.append(("INTERNAL_ONLY", internal_only_t, internal_only_hook, io_c, io_n))
    if args.logit_bias != 0.0:
        rows.append((f"INTERNAL+BIAS(bias={args.logit_bias:g})", fixed_t, fixed_hook, f_c, f_n))
    rows.append((f"ADAPTIVE_DUAL(bias={args.logit_bias:g})", ad_t, ad_hook, a_c, a_n))
    for name, tinfo, hook_ms, corp_hits, narr_hits in rows:
        hook_over = ""
        if hook_ms:
            hook_over = f" | hook_avg_ms={statistics.mean(hook_ms):.3f} p95_ms={sorted(hook_ms)[int(0.95*len(hook_ms))-1]:.3f}"
        print(
            f"{name}: seconds={tinfo['seconds']:.2f} gen_tokens={tinfo['gen_tokens']} tok/s={tinfo['tokens_per_sec']:.2f} "
            f"| corp_hits={corp_hits} narr_hits={narr_hits}{hook_over}"
        )
    print()

    print("=== Adaptive controller trace ===")
    if ctrl.str_hist:
        print(
            f"final_strength: avg={statistics.mean(ctrl.str_hist):.2f} min={min(ctrl.str_hist):.2f} max={max(ctrl.str_hist):.2f}"
        )
        print(
            f"entropy: avg={statistics.mean(ctrl.entropy_hist):.2f} min={min(ctrl.entropy_hist):.2f} max={max(ctrl.entropy_hist):.2f}"
        )
        print("first_steps (entropy,final_strength): " + ", ".join(
            f"({ctrl.entropy_hist[i]:.2f},{ctrl.str_hist[i]:.2f})" for i in range(min(10, len(ctrl.str_hist)))
        ))
    print()

    print("=== Outputs ===")
    print("--- BASELINE ---")
    print(base_text.strip() or "<EMPTY>")
    print()
    print("--- FIXED_DUAL ---")
    print(f"corp_top={f_top}")
    print(fixed_text.strip() or "<EMPTY>")
    print()
    if args.ablation and args.logit_bias != 0.0:
        print("--- BIAS_ONLY ---")
        print(f"corp_top={bo_top}")
        print(bias_only_text.strip() or "<EMPTY>")
        print()
    print("--- INTERNAL_ONLY ---")
    print(f"corp_top={io_top}")
    print(internal_only_text.strip() or "<EMPTY>")
    print()
    print("--- ADAPTIVE_DUAL ---")
    print(f"corp_top={a_top}")
    print(ad_text.strip() or "<EMPTY>")
    print()


if __name__ == "__main__":
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    main()

