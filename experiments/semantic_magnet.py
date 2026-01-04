import argparse
import os
import re
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)


def shannon_entropy_from_logits_row(logits_row: torch.Tensor) -> float:
    """Entropy in nats for a single [vocab] logits row."""
    probs = torch.softmax(logits_row, dim=-1).clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


def render_prompt(tokenizer, user_prompt: str) -> str:
    # Prefer chat templates for instruct models (Qwen uses these).
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


def get_input_embeddings_weight(model) -> torch.Tensor:
    """
    Returns the token embedding matrix [vocab, d] for common CausalLM architectures.
    """
    # Preferred: official API
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight
    # Fallbacks (older conventions)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight
    raise RuntimeError("Could not locate input embedding matrix for this model.")

def get_output_embeddings_weight(model) -> torch.Tensor:
    """
    Returns the output embedding / LM head matrix [vocab, d] if available.
    This is often more directly aligned with logits than the input embedding matrix.
    """
    out = model.get_output_embeddings()
    if out is not None and hasattr(out, "weight"):
        return out.weight
    # Many models tie weights; fall back to input.
    return get_input_embeddings_weight(model)


@dataclass
class EntropyTrace:
    pre: list[float]
    post: list[float]

    def avg_pre(self) -> float:
        return sum(self.pre) / max(1, len(self.pre))

    def avg_post(self) -> float:
        return sum(self.post) / max(1, len(self.post))


class SemanticMagnetLogitsProcessor(LogitsProcessor):
    """
    "Soft constraint": add a bias vector to logits proportional to embedding-space similarity
    between each token and a target concept center.

    This does NOT mask tokens; it warps the distribution by shifting probability mass.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_concept: str,
        strength: float,
        *,
        mode: str = "poe",
        q_temperature: float = 0.25,
        ascii_only_q: bool = True,
        require_leading_space: bool = True,
        lexicon_words: list[str] | None = None,
        lexicon_floor: float = 12.0,
        sim_threshold: float = 0.35,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.strength = float(strength)
        self.mode = mode
        self.q_temperature = float(q_temperature)
        self.ascii_only_q = bool(ascii_only_q)
        self.require_leading_space = bool(require_leading_space)
        self.lexicon_words = [w.lower() for w in (lexicon_words or [])]
        self.lexicon_floor = float(lexicon_floor)
        self.sim_threshold = float(sim_threshold)
        self.eps = float(eps)
        self.trace = EntropyTrace(pre=[], post=[])

        # Compute target "center" in the model's embedding space.
        target_ids = tokenizer.encode(target_concept, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Use LM head space if available.
            emb_w = get_output_embeddings_weight(model)  # [vocab, d]
            target_vec = emb_w[target_ids[0]].mean(dim=0, keepdim=False)  # [d]
            target_norm = target_vec.norm().clamp_min(self.eps)
            target_vec = target_vec / target_norm  # unit

            # Precompute cosine similarity to every vocab token ONCE:
            # sim_i = <e_i, t> / ||e_i|| since ||t||=1.
            vocab_norms = emb_w.norm(dim=1).clamp_min(self.eps)  # [vocab]
            dots = emb_w @ target_vec  # [vocab]
            similarity = dots / vocab_norms  # [vocab]

        # Store on CPU; moved to device on call.
        self.similarity_cpu = similarity.detach().to(dtype=torch.float32, device="cpu")
        vocab_size = int(self.similarity_cpu.shape[0])

        # Optional: token-level lexicon filter to keep the manifold prior "on-topic" (corporate words),
        # while still remaining soft by assigning a small floor probability outside the lexicon.
        lexicon_mask = None
        if self.lexicon_words:
            lex = set(self.lexicon_words)
            lm = torch.zeros(vocab_size, dtype=torch.bool)
            for tok_id in range(vocab_size):
                piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                if not piece.startswith(" "):
                    continue
                if piece.strip().lower() in lex:
                    lm[tok_id] = True
            lexicon_mask = lm
        # For additive mode, precompute a non-negative boost vector (only "very similar" tokens get boosted).
        boost = torch.clamp(self.similarity_cpu - self.sim_threshold, min=0.0)
        if lexicon_mask is not None:
            boost = torch.where(lexicon_mask, boost, torch.zeros_like(boost))
        elif self.ascii_only_q:
            # Reuse the same english-ish heuristic for add mode as well.
            mask2 = torch.zeros_like(boost, dtype=torch.bool)
            for tok_id in range(vocab_size):
                piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                if is_ascii_englishish(piece, require_leading_space=self.require_leading_space):
                    mask2[tok_id] = True
            boost = torch.where(mask2, boost, torch.zeros_like(boost))
        self.boost_cpu = boost.detach().to(dtype=torch.float32, device="cpu")
        # Precompute a target distribution q over vocab for "product-of-experts" mixing.
        # This is a soft semantic "manifold prior" (higher probability for more similar tokens).
        q_logits = (self.similarity_cpu / max(self.q_temperature, self.eps)).to(dtype=torch.float32)
        if lexicon_mask is not None:
            floor = torch.tensor(-abs(self.lexicon_floor), dtype=q_logits.dtype)
            q_logits = torch.where(lexicon_mask, q_logits, floor)
        elif self.ascii_only_q:
            # Filter q to ascii-ish word tokens to avoid weird unicode symbol collapse.
            mask = torch.zeros_like(q_logits, dtype=torch.bool)
            for tok_id in range(vocab_size):
                piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                if is_ascii_englishish(piece, require_leading_space=self.require_leading_space):
                    mask[tok_id] = True
            q_logits = torch.where(mask, q_logits, torch.tensor(-torch.inf, dtype=q_logits.dtype))

        self.log_q_cpu = torch.log_softmax(q_logits, dim=-1).detach().to(dtype=torch.float32, device="cpu")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab]
        if scores.numel() == 0:
            return scores
        if scores.shape[0] != 1:
            # keep it simple: single sample
            return scores

        pre_ent = shannon_entropy_from_logits_row(scores[0])
        self.trace.pre.append(pre_ent)

        if self.strength == 0.0:
            self.trace.post.append(pre_ent)
            return scores

        if self.mode == "add":
            boost = self.boost_cpu.to(device=scores.device)
            biased = scores + (boost.unsqueeze(0) * self.strength)
        elif self.mode == "poe":
            # log p' = (1-λ) log p + λ log q, where λ in [0,1].
            lam = float(max(0.0, min(1.0, self.strength)))
            log_p = torch.log_softmax(scores, dim=-1)
            log_q = self.log_q_cpu.to(device=scores.device).unsqueeze(0)
            biased = (1.0 - lam) * log_p + lam * log_q
        else:
            raise RuntimeError(f"Unknown mode: {self.mode!r}. Use 'add' or 'poe'.")

        post_ent = shannon_entropy_from_logits_row(biased[0])
        self.trace.post.append(post_ent)
        return biased


def keyword_hits(text: str, keywords: list[str]) -> dict[str, int]:
    t = text.lower()
    hits: dict[str, int] = {}
    for kw in keywords:
        hits[kw] = len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", t))
    return hits


def generate(
    model,
    tokenizer,
    rendered_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    logits_processor: LogitsProcessorList | None,
    repetition_penalty: float | None,
    no_repeat_ngram_size: int | None,
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
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def show_top_similar_tokens(tokenizer, similarity_cpu: torch.Tensor, *, k: int = 20) -> list[str]:
    vals, idxs = torch.topk(similarity_cpu, k=k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist(), strict=False):
        piece = tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)
        out.append(f"{i}:{piece!r}:{v:.3f}")
    return out

def show_top_weighted_tokens(tokenizer, weights_cpu: torch.Tensor, *, k: int = 20) -> list[str]:
    vals, idxs = torch.topk(weights_cpu, k=k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist(), strict=False):
        piece = tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)
        out.append(f"{i}:{piece!r}:{v:.3f}")
    return out

def is_ascii_englishish(token_piece: str, *, require_leading_space: bool) -> bool:
    # Heuristic: keep tokens that are ascii and contain at least one letter.
    # Optionally require a leading space (helps focus on word-start tokens in BPE tokenizers).
    if not token_piece:
        return False
    if require_leading_space and not token_piece.startswith(" "):
        return False
    for ch in token_piece:
        if ord(ch) > 127:
            return False
    return any(("a" <= ch.lower() <= "z") for ch in token_piece)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2: soft constraint distributional shaping (semantic magnet).")
    p.add_argument("--model", default=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument(
        "--prompt",
        default="Write a fast-paced action scene about a medieval knight fighting a dragon.",
    )
    p.add_argument(
        "--target",
        default="synergy leverage agile deliverables Q3 roadmap stakeholders bandwidth optimization",
    )
    p.add_argument("--mode", default="poe", choices=["poe", "add"], help="How to mix the manifold prior.")
    p.add_argument("--q-temperature", type=float, default=0.25, help="Softness of target manifold prior q.")
    p.add_argument(
        "--ascii-only-q",
        action="store_true",
        default=True,
        help="Restrict manifold prior q to ascii-ish word tokens (recommended).",
    )
    p.add_argument("--no-ascii-only-q", dest="ascii_only_q", action="store_false")
    p.add_argument(
        "--require-leading-space",
        action="store_true",
        default=True,
        help="Restrict prior/boost to tokens that start a new word (recommended for BPE).",
    )
    p.add_argument("--no-require-leading-space", dest="require_leading_space", action="store_false")
    p.add_argument(
        "--lexicon",
        default="",
        help="Comma-separated lexicon words to keep manifold prior on-topic (soft floor outside lexicon).",
    )
    p.add_argument("--lexicon-floor", type=float, default=12.0, help="Logit floor for tokens outside lexicon.")
    p.add_argument(
        "--sim-threshold",
        type=float,
        default=0.35,
        help="(add mode) Only boost tokens with similarity > this threshold.",
    )
    p.add_argument("--strengths", default="0,0.2,0.4,0.6", help="Comma-separated values. In poe mode: λ in [0,1].")
    p.add_argument("--max-new-tokens", type=int, default=140)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    strengths = [float(x.strip()) for x in args.strengths.split(",") if x.strip()]
    if not strengths:
        raise SystemExit("No strengths provided.")

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Target manifold: {args.target}")
    print(
        f"Mode: {args.mode} (q_temperature={args.q_temperature}, ascii_only_q={args.ascii_only_q}, "
        f"require_leading_space={args.require_leading_space}, sim_threshold={args.sim_threshold}, "
        f"lexicon={'ON' if bool(args.lexicon.strip()) else 'OFF'})"
    )
    print(f"Strengths: {strengths}")
    print()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to("cpu")
    model.eval()

    rendered = render_prompt(tokenizer, args.prompt)

    # Keywords for "leakage" measurement (small, interpretable proxy metric).
    leakage_keywords = [
        "synergy",
        "leverage",
        "agile",
        "deliverables",
        "roadmap",
        "stakeholders",
        "bandwidth",
        "optimization",
        "kpi",
        "roi",
        "quarter",
        "q3",
        "alignment",
        "strategy",
        "pipeline",
        "deadline",
    ]

    lexicon_words = [w.strip() for w in args.lexicon.split(",") if w.strip()]
    runs = []
    for alpha in strengths:
        proc = SemanticMagnetLogitsProcessor(
            model,
            tokenizer,
            args.target,
            alpha,
            mode=args.mode,
            q_temperature=args.q_temperature,
            ascii_only_q=args.ascii_only_q,
            require_leading_space=args.require_leading_space,
            lexicon_words=lexicon_words or None,
            lexicon_floor=args.lexicon_floor,
            sim_threshold=args.sim_threshold,
        )
        lp = LogitsProcessorList([proc])
        text = generate(
            model,
            tokenizer,
            rendered,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            logits_processor=lp,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        hits = keyword_hits(text, leakage_keywords)
        runs.append((alpha, text, proc.trace, hits))
        if alpha == strengths[0]:
            # Print diagnostics once (using first run's manifold prior).
            print("Top tokens by similarity to target manifold (diagnostic):")
            for line in show_top_similar_tokens(tokenizer, proc.similarity_cpu, k=20):
                print("  " + line)
            print("Top boosted tokens (add-mode boost vector; diagnostic):")
            for line in show_top_weighted_tokens(tokenizer, proc.boost_cpu, k=20):
                print("  " + line)
            print()

    dt = time.time() - t0
    print("=== Summary ===")
    print(f"Wall time: {dt:.1f}s")
    for alpha, _text, trace, hits in runs:
        total_hits = sum(hits.values())
        print(
            f"- alpha={alpha:g}: avg_entropy_pre={trace.avg_pre():.3f}, "
            f"avg_entropy_post={trace.avg_post():.3f}, leakage_hits={total_hits}"
        )
    print()

    for alpha, text, trace, hits in runs:
        total_hits = sum(hits.values())
        top_hits = sorted(((k, v) for k, v in hits.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))[:8]
        print(f"--- alpha={alpha:g} ---")
        print(f"avg_entropy_pre={trace.avg_pre():.3f}, avg_entropy_post={trace.avg_post():.3f}")
        print(f"leakage_hits={total_hits}; top_keywords={top_hits}")
        print(text.strip() or "<EMPTY>")
        print()


if __name__ == "__main__":
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    main()

