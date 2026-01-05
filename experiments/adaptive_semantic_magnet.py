import argparse
import os
import re
import statistics
import time
from collections import deque

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)


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


def get_output_embeddings_weight(model) -> torch.Tensor:
    out = model.get_output_embeddings()
    if out is not None and hasattr(out, "weight"):
        return out.weight
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight
    raise RuntimeError("Could not locate embeddings.")


def entropy_from_logits(scores_1d: torch.Tensor) -> float:
    probs = torch.softmax(scores_1d, dim=-1).clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


def keyword_hits(text: str, keywords: list[str]) -> dict[str, int]:
    t = text.lower()
    hits: dict[str, int] = {}
    for kw in keywords:
        hits[kw] = len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", t))
    return hits


def build_semantic_log_prior(
    *,
    model,
    tokenizer,
    concept_text: str,
    q_temperature: float,
    lexicon_words: list[str] | None,
    lexicon_floor: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a log prior log q over vocab based on embedding-space similarity to a concept center.
    Optionally apply a *soft* lexicon floor: tokens outside lexicon get a low (but non-zero) prior.
    Returns CPU tensor [vocab].
    """
    with torch.no_grad():
        W = get_output_embeddings_weight(model)  # [vocab, d]
        vocab_size = int(W.shape[0])
        ids = tokenizer.encode(concept_text, return_tensors="pt").to(model.device)
        tvec = W[ids[0]].mean(dim=0)
        tvec = tvec / tvec.norm().clamp_min(eps)
        sim = (W @ tvec) / W.norm(dim=1).clamp_min(eps)  # [vocab]

    q_logits = (sim / max(q_temperature, eps)).to(dtype=torch.float32, device="cpu")

    if lexicon_words:
        lex = set(w.lower() for w in lexicon_words)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tok_id in range(vocab_size):
            piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            if not piece.startswith(" "):
                continue
            if piece.strip().lower() in lex:
                mask[tok_id] = True
        floor = torch.tensor(-abs(lexicon_floor), dtype=q_logits.dtype)
        q_logits = torch.where(mask, q_logits, floor)

    return torch.log_softmax(q_logits, dim=-1).detach().to(dtype=torch.float32, device="cpu")


def build_lexicon_token_id_set(tokenizer, vocab_size: int, lexicon_words: list[str]) -> set[int]:
    lex = set(w.lower() for w in lexicon_words)
    ids: set[int] = set()
    for tok_id in range(vocab_size):
        piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        if not piece.startswith(" "):
            continue
        if piece.strip().lower() in lex:
            ids.add(tok_id)
    return ids


class StaticPoESemanticMagnet(LogitsProcessor):
    """
    Soft constraint via product-of-experts:
      log p' = (1-λ) log p + λ log q
    where q is a semantic prior over tokens.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_concept: str,
        lam: float,
        *,
        q_temperature: float,
        lexicon_words: list[str] | None,
        lexicon_floor: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lam = float(lam)
        self.q_temperature = float(q_temperature)
        self.lexicon_words = [w.lower() for w in (lexicon_words or [])]
        self.lexicon_floor = float(lexicon_floor)
        self.eps = float(eps)

        self.log_q_cpu = build_semantic_log_prior(
            model=model,
            tokenizer=tokenizer,
            concept_text=target_concept,
            q_temperature=self.q_temperature,
            lexicon_words=self.lexicon_words if self.lexicon_words else None,
            lexicon_floor=self.lexicon_floor,
            eps=self.eps,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 1:
            return scores
        lam = float(max(0.0, min(1.0, self.lam)))
        log_p = torch.log_softmax(scores, dim=-1)
        log_q = self.log_q_cpu.to(device=scores.device).unsqueeze(0)
        return (1.0 - lam) * log_p + lam * log_q


class AdaptivePoESemanticMagnet(LogitsProcessor):
    """
    Entropy-bounded adaptive control of λ:

    - Measure entropy H_t of the *base* distribution p_t (before constraint).
    - Update λ_t via proportional control to move H_t toward a target entropy.
      If entropy too low (collapse/repetition): decrease λ (relax constraint).
      If entropy too high (drift): increase λ (tighten constraint).

    Then apply the same PoE mixing:
      log p'_t = (1-λ_t) log p_t + λ_t log q
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_concept: str,
        *,
        base_lam: float,
        target_entropy: float,
        adaptation_rate: float,
        lam_min: float,
        lam_max: float,
        q_temperature: float,
        lexicon_words: list[str] | None,
        lexicon_floor: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.current_lam = float(base_lam)
        self.target_entropy = float(target_entropy)
        self.adaptation_rate = float(adaptation_rate)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.q_temperature = float(q_temperature)
        self.lexicon_words = [w.lower() for w in (lexicon_words or [])]
        self.lexicon_floor = float(lexicon_floor)
        self.eps = float(eps)

        self.entropy_history: list[float] = []
        self.post_entropy_history: list[float] = []
        self.lambda_history: list[float] = []

        self.log_q_cpu = build_semantic_log_prior(
            model=model,
            tokenizer=tokenizer,
            concept_text=target_concept,
            q_temperature=self.q_temperature,
            lexicon_words=self.lexicon_words if self.lexicon_words else None,
            lexicon_floor=self.lexicon_floor,
            eps=self.eps,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 1:
            return scores

        # 1) Measure entropy of base distribution (pre-constraint).
        H_pre = entropy_from_logits(scores[0])
        self.entropy_history.append(H_pre)

        # 2) Apply constraint with current λ to estimate *post* entropy (collapse indicator).
        log_p = torch.log_softmax(scores, dim=-1)
        log_q = self.log_q_cpu.to(device=scores.device).unsqueeze(0)
        mixed = (1.0 - self.current_lam) * log_p + self.current_lam * log_q
        H_post = entropy_from_logits(mixed[0])
        self.post_entropy_history.append(H_post)

        # 3) Proportional control update of λ based on post-constraint entropy.
        # If post entropy too low (collapse): relax λ.
        # If post entropy too high (drift): tighten λ.
        error = self.target_entropy - H_post
        adjustment = -error * self.adaptation_rate
        self.current_lam = float(max(self.lam_min, min(self.lam_max, self.current_lam + adjustment)))
        self.lambda_history.append(self.current_lam)

        # 4) Return logits using updated λ for this step.
        lam = self.current_lam
        return (1.0 - lam) * log_p + lam * log_q


class DualPriorAdaptiveMagnet(LogitsProcessor):
    """
    Adaptive semantic magnet with *two* priors:
      - q_corp: corporate manifold prior (lexicon-filtered, soft floor)
      - q_anchor: prompt-anchor manifold prior (keeps narrative tokens attractive)

    Control:
      - λ_t: overall constraint strength, adapted by post-constraint entropy.
      - β_t: corporate-vs-anchor mix inside the constraint, adapted by recent corporate token rate.

    log p'_t = (1-λ_t) log p_t + λ_t * ( β_t log q_corp + (1-β_t) log q_anchor )
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        rendered_prompt: str,
        target_concept: str,
        lexicon_words: list[str],
        # entropy control for λ
        base_lam: float,
        target_entropy: float,
        lam_rate: float,
        lam_min: float,
        lam_max: float,
        # lexical-rate control for β
        base_beta: float,
        target_corp_rate: float,
        beta_rate: float,
        beta_min: float,
        beta_max: float,
        window: int,
        # priors
        q_temperature: float,
        lexicon_floor: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eps = float(eps)

        self.current_lam = float(base_lam)
        self.target_entropy = float(target_entropy)
        self.lam_rate = float(lam_rate)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)

        self.current_beta = float(base_beta)
        self.target_corp_rate = float(target_corp_rate)
        self.beta_rate = float(beta_rate)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.window = int(window)

        self.entropy_pre: list[float] = []
        self.entropy_post: list[float] = []
        self.lambda_history: list[float] = []
        self.beta_history: list[float] = []
        self.proc_time_ms: list[float] = []

        # Precompute priors.
        self.log_q_corp_cpu = build_semantic_log_prior(
            model=model,
            tokenizer=tokenizer,
            concept_text=target_concept,
            q_temperature=q_temperature,
            lexicon_words=lexicon_words,
            lexicon_floor=lexicon_floor,
            eps=self.eps,
        )
        self.log_q_anchor_cpu = build_semantic_log_prior(
            model=model,
            tokenizer=tokenizer,
            concept_text=rendered_prompt,
            q_temperature=q_temperature,
            lexicon_words=None,
            lexicon_floor=lexicon_floor,
            eps=self.eps,
        )

        vocab_size = int(self.log_q_corp_cpu.shape[0])
        self.lexicon_token_ids = build_lexicon_token_id_set(tokenizer, vocab_size, lexicon_words)
        self.corp_window: deque[int] = deque(maxlen=self.window)
        self.prompt_len = int(tokenizer(rendered_prompt, return_tensors="pt")["input_ids"].shape[-1])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        t0 = time.perf_counter()
        if scores.shape[0] != 1:
            return scores

        # Track corporate token rate in a rolling window (only after generation starts).
        if input_ids.shape[-1] > self.prompt_len:
            last_id = int(input_ids[0, -1].item())
            self.corp_window.append(1 if last_id in self.lexicon_token_ids else 0)

        log_p = torch.log_softmax(scores, dim=-1)

        # Mix priors with current beta.
        log_q_c = self.log_q_corp_cpu.to(device=scores.device).unsqueeze(0)
        log_q_a = self.log_q_anchor_cpu.to(device=scores.device).unsqueeze(0)
        beta = float(max(self.beta_min, min(self.beta_max, self.current_beta)))
        prior_mix = beta * log_q_c + (1.0 - beta) * log_q_a

        H_pre = entropy_from_logits(scores[0])
        self.entropy_pre.append(H_pre)

        mixed = (1.0 - self.current_lam) * log_p + self.current_lam * prior_mix
        H_post = entropy_from_logits(mixed[0])
        self.entropy_post.append(H_post)

        # Update λ using post entropy.
        err_H = self.target_entropy - H_post
        self.current_lam = float(
            max(self.lam_min, min(self.lam_max, self.current_lam + (-err_H * self.lam_rate)))
        )
        self.lambda_history.append(self.current_lam)

        # Update β using recent corporate rate.
        if len(self.corp_window) > 0:
            corp_rate = sum(self.corp_window) / len(self.corp_window)
            # If corporate rate too high: decrease beta; if too low: increase beta.
            err_r = self.target_corp_rate - corp_rate
            self.current_beta = float(
                max(self.beta_min, min(self.beta_max, self.current_beta + (err_r * self.beta_rate)))
            )
        self.beta_history.append(self.current_beta)

        # Return distribution using updated λ,β for this step.
        beta2 = float(max(self.beta_min, min(self.beta_max, self.current_beta)))
        prior_mix2 = beta2 * log_q_c + (1.0 - beta2) * log_q_a
        out = (1.0 - self.current_lam) * log_p + self.current_lam * prior_mix2

        self.proc_time_ms.append((time.perf_counter() - t0) * 1000.0)
        return out


def generate(
    model,
    tokenizer,
    rendered_prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    logits_processor: LogitsProcessorList | None,
) -> str:
    set_seed(seed)
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: entropy-bounded adaptive semantic magnet.")
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument(
        "--prompt",
        default="Write a fast-paced action scene about a medieval knight fighting a dragon.",
    )
    ap.add_argument(
        "--target",
        default="synergy leverage agile deliverables Q3 roadmap stakeholders bandwidth optimization strategy alignment",
    )
    ap.add_argument(
        "--lexicon",
        default="synergy,leverage,agile,deliverables,roadmap,stakeholders,bandwidth,optimization,strategy,alignment,pipeline,deadline,kpi,okr,roi,q3",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=170)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.10)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)

    # Static (for comparison)
    ap.add_argument("--static-lam", type=float, default=0.55)

    # Adaptive control parameters
    ap.add_argument("--base-lam", type=float, default=0.45)
    ap.add_argument("--target-entropy", type=float, default=2.5)
    ap.add_argument("--adaptation-rate", type=float, default=0.2)
    ap.add_argument("--lam-min", type=float, default=0.0)
    ap.add_argument("--lam-max", type=float, default=0.85)
    # Dual-prior controls (Phase 3+)
    ap.add_argument("--use-dual", action="store_true", default=True)
    ap.add_argument("--no-use-dual", dest="use_dual", action="store_false")
    ap.add_argument("--base-beta", type=float, default=0.45)
    ap.add_argument("--target-corp-rate", type=float, default=0.18)
    ap.add_argument("--beta-rate", type=float, default=0.35)
    ap.add_argument("--beta-min", type=float, default=0.05)
    ap.add_argument("--beta-max", type=float, default=0.75)
    ap.add_argument("--corp-window", type=int, default=30)

    # Manifold prior params
    ap.add_argument("--q-temperature", type=float, default=0.25)
    ap.add_argument("--lexicon-floor", type=float, default=12.0)

    args = ap.parse_args()

    lexicon_words = [w.strip() for w in args.lexicon.split(",") if w.strip()]
    leakage_keywords = list(lexicon_words)

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Target manifold: {args.target}")
    print(f"Decoding: temp={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}")
    print(f"Static λ: {args.static_lam}")
    print(
        f"Adaptive: base_λ={args.base_lam}, target_entropy={args.target_entropy}, "
        f"rate={args.adaptation_rate}, clamp=[{args.lam_min},{args.lam_max}]"
    )
    print()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to("cpu")
    model.eval()
    rendered = render_prompt(tokenizer, args.prompt)

    def timed_generate(label: str, logits_processor: LogitsProcessorList | None):
        t_start = time.perf_counter()
        text = generate(
            model,
            tokenizer,
            rendered,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            logits_processor=logits_processor,
        )
        t_end = time.perf_counter()
        # token accounting
        in_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
        out_ids = tokenizer(render_prompt(tokenizer, args.prompt) + text, return_tensors="pt")["input_ids"]
        gen_tokens = int(max(0, out_ids.shape[-1] - in_ids.shape[-1]))
        sec = float(t_end - t_start)
        tps = (gen_tokens / sec) if sec > 0 else float("nan")
        return text, {"label": label, "seconds": sec, "gen_tokens": gen_tokens, "tokens_per_sec": tps}

    # Baseline
    baseline, baseline_t = timed_generate("BASELINE", None)

    # Static constraint
    static_proc = StaticPoESemanticMagnet(
        model,
        tokenizer,
        args.target,
        args.static_lam,
        q_temperature=args.q_temperature,
        lexicon_words=lexicon_words,
        lexicon_floor=args.lexicon_floor,
    )
    static, static_t = timed_generate("STATIC", LogitsProcessorList([static_proc]))

    if args.use_dual:
        adaptive_proc = DualPriorAdaptiveMagnet(
            model,
            tokenizer,
            rendered_prompt=rendered,
            target_concept=args.target,
            lexicon_words=lexicon_words,
            base_lam=args.base_lam,
            target_entropy=args.target_entropy,
            lam_rate=args.adaptation_rate,
            lam_min=args.lam_min,
            lam_max=args.lam_max,
            base_beta=args.base_beta,
            target_corp_rate=args.target_corp_rate,
            beta_rate=args.beta_rate,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            window=args.corp_window,
            q_temperature=args.q_temperature,
            lexicon_floor=args.lexicon_floor,
        )
    else:
        adaptive_proc = AdaptivePoESemanticMagnet(
            model,
            tokenizer,
            args.target,
            base_lam=args.base_lam,
            target_entropy=args.target_entropy,
            adaptation_rate=args.adaptation_rate,
            lam_min=args.lam_min,
            lam_max=args.lam_max,
            q_temperature=args.q_temperature,
            lexicon_words=lexicon_words,
            lexicon_floor=args.lexicon_floor,
        )

    adaptive, adaptive_t = timed_generate("ADAPTIVE", LogitsProcessorList([adaptive_proc]))

    dt = time.time() - t0

    def summarize(label: str, text: str) -> None:
        hits = keyword_hits(text, leakage_keywords)
        total_hits = sum(hits.values())
        top_hits = sorted(((k, v) for k, v in hits.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))[:8]
        print(f"=== {label} ===")
        print(f"leakage_hits={total_hits}; top_keywords={top_hits}")
        print(text.strip() or "<EMPTY>")
        print()

    summarize("BASELINE", baseline)
    summarize(f"STATIC (λ={args.static_lam})", static)
    summarize("ADAPTIVE", adaptive)

    print("=== Inference time (critical) ===")
    for info in [baseline_t, static_t, adaptive_t]:
        print(
            f"{info['label']}: seconds={info['seconds']:.2f} gen_tokens={info['gen_tokens']} "
            f"tokens_per_sec={info['tokens_per_sec']:.2f}"
        )
    print()

    if hasattr(adaptive_proc, "lambda_history") and getattr(adaptive_proc, "lambda_history"):
        print("=== Adaptive dynamics ===")
        if isinstance(adaptive_proc, DualPriorAdaptiveMagnet):
            print(f"steps: {len(adaptive_proc.lambda_history)}")
            print(
                f"entropy_post: avg={statistics.mean(adaptive_proc.entropy_post):.2f} "
                f"min={min(adaptive_proc.entropy_post):.2f} max={max(adaptive_proc.entropy_post):.2f}"
            )
            print(
                f"lambda: avg={statistics.mean(adaptive_proc.lambda_history):.2f} "
                f"min={min(adaptive_proc.lambda_history):.2f} max={max(adaptive_proc.lambda_history):.2f}"
            )
            print(
                f"beta:  avg={statistics.mean(adaptive_proc.beta_history):.2f} "
                f"min={min(adaptive_proc.beta_history):.2f} max={max(adaptive_proc.beta_history):.2f}"
            )
            if adaptive_proc.proc_time_ms:
                print(
                    f"processor_overhead: avg_ms={statistics.mean(adaptive_proc.proc_time_ms):.2f} "
                    f"p95_ms={sorted(adaptive_proc.proc_time_ms)[int(0.95*len(adaptive_proc.proc_time_ms))-1]:.2f}"
                )
            pairs = list(zip(adaptive_proc.entropy_post, adaptive_proc.lambda_history, adaptive_proc.beta_history))[:10]
            print("first_steps (post_entropy, lambda, beta): " + ", ".join(f"({h:.2f},{l:.2f},{b:.2f})" for h, l, b in pairs))
            print()
        else:
            print(f"steps: {len(adaptive_proc.lambda_history)}")
            print(
                f"entropy_post: avg={statistics.mean(adaptive_proc.post_entropy_history):.2f} "
                f"min={min(adaptive_proc.post_entropy_history):.2f} max={max(adaptive_proc.post_entropy_history):.2f}"
            )
            print(
                f"lambda:  avg={statistics.mean(adaptive_proc.lambda_history):.2f} "
                f"min={min(adaptive_proc.lambda_history):.2f} max={max(adaptive_proc.lambda_history):.2f}"
            )
            pairs = list(
                zip(adaptive_proc.post_entropy_history, adaptive_proc.lambda_history, strict=False)
            )[:12]
            print("first_steps (post_entropy, lambda): " + ", ".join(f"({h:.2f},{l:.2f})" for h, l in pairs))
            print()

    print(f"Wall time: {dt:.1f}s")


if __name__ == "__main__":
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    main()

