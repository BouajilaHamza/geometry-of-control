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
    # de-dupe while preserving order
    seen = set()
    out = []
    for tid in token_ids:
        if tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out, debug


def build_direction_from_lexicon(
    model,
    tokenizer,
    lexicon_ids: list[int],
    *,
    neg_sample: int,
    neg_seed: int,
) -> torch.Tensor:
    W = get_output_embeddings_weight(model).detach()  # [vocab, d]
    vocab_size = int(W.shape[0])
    pos = W[torch.tensor(lexicon_ids, dtype=torch.long, device=W.device)].mean(dim=0, keepdim=True)  # [1,d]

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

    v = (pos - neg).unsqueeze(0)  # [1,1,d]
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return v


def orthogonalize_against(W: torch.Tensor, v: torch.Tensor, avoid_ids: list[int]) -> torch.Tensor:
    """
    v: [d] unit-ish
    W: [vocab, d]
    Remove components of v along the span of W[avoid_ids].
    """
    vv = v
    for tid in avoid_ids:
        w = W[tid]
        denom = float((w @ w).item())
        if denom <= 1e-12:
            continue
        proj = float((vv @ w).item()) / denom
        vv = vv - proj * w
    vv = vv / vv.norm().clamp_min(1e-8)
    return vv


def purify_direction(
    model,
    tokenizer,
    v: torch.Tensor,
    *,
    lexicon_ids: set[int],
    rounds: int = 2,
    topk: int = 200,
    avoid_n: int = 40,
) -> torch.Tensor:
    """
    Iteratively orthogonalize v against top boosted non-lexicon tokens,
    so the direction becomes more discriminative for the lexicon set.
    """
    W = get_output_embeddings_weight(model).detach()
    d = W.shape[1]
    vv = v.reshape(d)
    for _ in range(rounds):
        scores = (W @ vv).detach().cpu()
        # top boosted ids
        _, idxs = torch.topk(scores, k=min(topk, scores.numel()))
        avoid = []
        for tid in idxs.tolist():
            if tid in lexicon_ids:
                continue
            piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            # avoid "weird" tokens and common starters that steal mass
            if (not is_ascii_wordpiece(piece)) or (
                piece.strip()
                in {"The", "As", "I", "In", "A", "It", "Once", "INT", "**", "[", "]", "Scene", "Knight", "Suddenly"}
            ):
                avoid.append(tid)
            if len(avoid) >= avoid_n:
                break
        if not avoid:
            break
        vv = orthogonalize_against(W, vv, avoid)
    vv = vv / vv.norm().clamp_min(1e-8)
    return vv.unsqueeze(0).unsqueeze(0)  # [1,1,d]


class FinalNormInjector:
    """
    Injects v into the final hidden stream (only last position), with dynamic strength.
    """

    def __init__(self, model, v: torch.Tensor):
        self.model = model
        self.v = v.to(model.device)  # [1,1,d]
        self.current_strength = 0.0
        self._handle = None
        self.hook_ms: list[float] = []

    def enable(self, strength: float):
        if self._handle is not None:
            raise RuntimeError("Already enabled.")
        self.current_strength = float(strength)
        mod = get_final_norm_module(self.model)

        def hook(_module, _inp, out):
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


class StrengthController(LogitsProcessor):
    """
    Controls injector.current_strength based on:
      - rolling corporate token rate (based on input_ids)
      - entropy of the *current* logits (post-injection, since this runs after forward)

    IMPORTANT: returns scores unchanged (no output shaping).
    """

    def __init__(
        self,
        injector: FinalNormInjector,
        *,
        lexicon_token_ids: set[int],
        target_corp_rate: float,
        rate_window: int,
        target_entropy: float,
        k_rate: float,
        k_ent: float,
        strength_min: float,
        strength_max: float,
    ):
        super().__init__()
        self.injector = injector
        self.lexicon_token_ids = set(int(x) for x in lexicon_token_ids)
        self.target_corp_rate = float(target_corp_rate)
        self.target_entropy = float(target_entropy)
        self.k_rate = float(k_rate)
        self.k_ent = float(k_ent)
        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        self.corp_window = deque(maxlen=int(rate_window))
        self.entropy_hist: list[float] = []
        self.strength_hist: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 1:
            return scores

        # Corporate token rate from last selected token id
        if input_ids.shape[-1] > 0:
            last_id = int(input_ids[0, -1].item())
            self.corp_window.append(1 if last_id in self.lexicon_token_ids else 0)
        corp_rate = (sum(self.corp_window) / len(self.corp_window)) if len(self.corp_window) > 0 else 0.0

        H = entropy_from_logits(scores[0].detach())
        self.entropy_hist.append(H)

        # Control:
        # - if corp_rate too low => increase strength
        # - if entropy too low (collapse) => decrease strength
        d_strength = self.k_rate * (self.target_corp_rate - corp_rate) - self.k_ent * max(0.0, self.target_entropy - H)
        new_strength = float(self.injector.current_strength + d_strength)
        new_strength = float(max(self.strength_min, min(self.strength_max, new_strength)))
        self.injector.current_strength = new_strength
        self.strength_hist.append(new_strength)

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

def topk_tokens(tokenizer, logits_1d: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
    vals, idxs = torch.topk(logits_1d, k=k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist(), strict=False):
        piece = tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)
        out.append((piece, float(v)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Residual steering with discriminative direction + adaptive strength.")
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--prompt", default="Write a fast-paced action scene about a medieval knight fighting a dragon.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.12)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)

    # direction construction
    ap.add_argument("--neg-sample", type=int, default=8000)
    ap.add_argument("--neg-seed", type=int, default=1)
    ap.add_argument("--purify-rounds", type=int, default=2)
    ap.add_argument("--avoid-n", type=int, default=40)
    ap.add_argument("--diagnose-logits", action="store_true", default=True)
    ap.add_argument("--no-diagnose-logits", dest="diagnose_logits", action="store_false")
    ap.add_argument("--diag-strength", type=float, default=30.0)

    # fixed vs adaptive
    ap.add_argument("--fixed-strength", type=float, default=20.0)
    ap.add_argument("--adaptive-start", type=float, default=15.0)
    ap.add_argument("--strength-min", type=float, default=0.0)
    ap.add_argument("--strength-max", type=float, default=40.0)

    # controller params
    ap.add_argument("--target-corp-rate", type=float, default=0.10)
    ap.add_argument("--rate-window", type=int, default=30)
    ap.add_argument("--target-entropy", type=float, default=2.5)
    ap.add_argument("--k-rate", type=float, default=8.0)
    ap.add_argument("--k-ent", type=float, default=6.0)

    args = ap.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to("cpu")
    model.eval()

    rendered = render_prompt(tokenizer, args.prompt)

    # Keep lexicon aligned with single-token pieces we can actually steer reliably.
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

    W = get_output_embeddings_weight(model)
    vocab_size = int(W.shape[0])
    lex_ids, lex_debug = build_lexicon_token_ids(tokenizer, vocab_size, lexicon_words)
    lex_set = set(lex_ids)

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("Lexicon token mapping (word -> token_id:piece):")
    for w, tid, piece in lex_debug:
        print(f"  {w} -> {tid}:{piece!r}")
    print()

    v0 = build_direction_from_lexicon(model, tokenizer, lex_ids, neg_sample=args.neg_sample, neg_seed=args.neg_seed)
    v = purify_direction(
        model,
        tokenizer,
        v0,
        lexicon_ids=lex_set,
        rounds=args.purify_rounds,
        avoid_n=args.avoid_n,
    )
    print(f"Built steering direction: shape={tuple(v.shape)} norm={float(v.norm().item()):.3f}")
    print()

    injector = FinalNormInjector(model, v)

    if args.diagnose_logits:
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            base_out = model(**inputs)
            base_logits = base_out.logits[0, -1, :].detach().cpu()
        injector.disable()
        injector.hook_ms.clear()
        injector.enable(args.diag_strength)
        with torch.no_grad():
            steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().cpu()
        injector.disable()
        # show deltas for lexicon ids
        deltas = []
        for tid in sorted(set(lex_ids)):
            piece = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
            deltas.append((piece, float((steered_logits[tid] - base_logits[tid]).item())))
        print("=== Logits diagnostic (prompt next-token) ===")
        print("Top-10 base:", topk_tokens(tokenizer, base_logits, k=10))
        print("Top-10 steered:", topk_tokens(tokenizer, steered_logits, k=10))
        print("Lexicon token deltas (steered-base):", sorted(deltas, key=lambda kv: -kv[1]))
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

    # Baseline
    baseline, base_t = timed_generate(model, tokenizer, rendered, gen_kwargs=gen_kwargs)

    # Fixed steering
    injector.disable()
    injector.hook_ms.clear()
    injector.enable(args.fixed_strength)
    fixed, fixed_t = timed_generate(model, tokenizer, rendered, gen_kwargs=gen_kwargs)
    fixed_hook = list(injector.hook_ms)
    injector.disable()

    # Adaptive steering (same direction; controller adjusts injector.current_strength)
    injector.disable()
    injector.hook_ms.clear()
    injector.enable(args.adaptive_start)
    controller = StrengthController(
        injector,
        lexicon_token_ids=lex_set,
        target_corp_rate=args.target_corp_rate,
        rate_window=args.rate_window,
        target_entropy=args.target_entropy,
        k_rate=args.k_rate,
        k_ent=args.k_ent,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
    )
    adaptive, adaptive_t = timed_generate(
        model,
        tokenizer,
        rendered,
        gen_kwargs={**gen_kwargs, "logits_processor": LogitsProcessorList([controller])},
    )
    adaptive_hook = list(injector.hook_ms)
    injector.disable()

    def summarize(label: str, text: str):
        corp = keyword_hits(text, lexicon_words)
        narr = keyword_hits(text, narrative_words)
        corp_total = sum(corp.values())
        narr_total = sum(narr.values())
        corp_top = sorted(((k, v) for k, v in corp.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))[:8]
        return corp_total, narr_total, corp_top

    b_c, b_n, b_top = summarize("baseline", baseline)
    f_c, f_n, f_top = summarize("fixed", fixed)
    a_c, a_n, a_top = summarize("adaptive", adaptive)

    print("=== Inference time (critical) ===")
    for name, tinfo, hook_ms, corp_total, narr_total in [
        ("BASELINE", base_t, [], b_c, b_n),
        ("FIXED", fixed_t, fixed_hook, f_c, f_n),
        ("ADAPTIVE", adaptive_t, adaptive_hook, a_c, a_n),
    ]:
        hook_over = ""
        if hook_ms:
            hook_over = f" | hook_avg_ms={statistics.mean(hook_ms):.3f} p95_ms={sorted(hook_ms)[int(0.95*len(hook_ms))-1]:.3f}"
        print(
            f"{name}: seconds={tinfo['seconds']:.2f} gen_tokens={tinfo['gen_tokens']} tok/s={tinfo['tokens_per_sec']:.2f} "
            f"| corp_hits={corp_total} narr_hits={narr_total}{hook_over}"
        )
    print()

    if controller.strength_hist:
        print("=== Adaptive controller trace ===")
        print(
            f"strength: avg={statistics.mean(controller.strength_hist):.2f} "
            f"min={min(controller.strength_hist):.2f} max={max(controller.strength_hist):.2f}"
        )
        print(
            f"entropy:  avg={statistics.mean(controller.entropy_hist):.2f} "
            f"min={min(controller.entropy_hist):.2f} max={max(controller.entropy_hist):.2f}"
        )
        print("first_steps (entropy,strength): " + ", ".join(
            f"({controller.entropy_hist[i]:.2f},{controller.strength_hist[i]:.2f})" for i in range(min(12, len(controller.strength_hist)))
        ))
        print()

    print("=== Outputs ===")
    print("--- BASELINE ---")
    print(baseline.strip() or "<EMPTY>")
    print()
    print("--- FIXED ---")
    print(fixed.strip() or "<EMPTY>")
    print()
    print("--- ADAPTIVE ---")
    print(adaptive.strip() or "<EMPTY>")
    print()


if __name__ == "__main__":
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    main()

