import argparse
import math
import os
import random
import time
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)


def shannon_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    logits: shape [vocab]
    returns entropy in nats
    """
    probs = torch.softmax(logits, dim=-1)
    probs = probs.clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


class ForbiddenSubstrLogitsProcessor(LogitsProcessor):
    """
    Sets logits to -inf for any token whose decoded text contains a forbidden substring.
    This is a "hard" constraint: those tokens become impossible to sample.
    """

    def __init__(self, bad_token_ids: torch.Tensor):
        super().__init__()
        # store on CPU; moved to device inside __call__
        self.bad_token_ids_cpu = bad_token_ids.to(dtype=torch.long, device="cpu")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab]
        if scores.numel() == 0:
            return scores
        bad = self.bad_token_ids_cpu.to(device=scores.device)
        scores[:, bad] = -torch.inf
        return scores


class EntropyGateLogitsProcessor(LogitsProcessor):
    """
    Not a mask, but an inference-time "quality gate".
    If next-token entropy exceeds a threshold, force EOS (or equivalently stop).
    """

    def __init__(self, entropy_threshold_nats: float, eos_token_id: int):
        super().__init__()
        self.entropy_threshold_nats = float(entropy_threshold_nats)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab]
        if scores.shape[0] != 1:
            # keep implementation simple/deterministic
            return scores
        ent = shannon_entropy_from_logits(scores[0])
        if ent > self.entropy_threshold_nats:
            forced = torch.full_like(scores, -torch.inf)
            forced[:, self.eos_token_id] = 0.0
            return forced
        return scores


@dataclass(frozen=True)
class RunResult:
    prompt: str
    baseline: str
    constrained: str
    baseline_ok: bool
    constrained_ok: bool


def build_bad_token_ids(tokenizer, forbidden: str, case_insensitive: bool) -> torch.Tensor:
    forbidden_cmp = forbidden.lower() if case_insensitive else forbidden

    bad_ids = []
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    for tok_id in range(vocab_size):
        piece = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        piece_cmp = piece.lower() if case_insensitive else piece
        if forbidden_cmp in piece_cmp:
            bad_ids.append(tok_id)

    if not bad_ids:
        raise RuntimeError(
            f"No bad token ids found for forbidden={forbidden!r}. "
            "This likely indicates a tokenizer decode mismatch."
        )
    return torch.tensor(sorted(set(bad_ids)), dtype=torch.long)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    logits_processor: LogitsProcessorList | None,
) -> str:
    set_seed(seed)

    # Prefer the model's chat template (instruct models often require it).
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def contains_forbidden(text: str, forbidden: str, case_insensitive: bool) -> bool:
    t = text.lower() if case_insensitive else text
    f = forbidden.lower() if case_insensitive else forbidden
    return f in t


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-time stochastic constraint validation (logit masking).")
    parser.add_argument(
        "--model",
        default=os.environ.get("HF_MODEL", "HuggingFaceTB/SmolLM-135M-Instruct"),
        help="HF model id (keep small for CPU). Default: HuggingFaceTB/SmolLM-135M-Instruct",
    )
    parser.add_argument("--forbidden", default="o", help="Forbidden substring (e.g., 'o' or 'e' or 'the').")
    parser.add_argument("--case-insensitive", action="store_true", default=True, help="Case-insensitive matching.")
    parser.add_argument("--no-case-insensitive", dest="case_insensitive", action="store_false")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--entropy-gate", type=float, default=None, help="If set, entropy threshold (nats) to gate.")
    args = parser.parse_args()

    # Determinism for prompt ordering
    random.seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Forbidden: {args.forbidden!r} (case_insensitive={args.case_insensitive})")
    print(f"Decoding: temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}")
    if args.entropy_gate is not None:
        print(f"Entropy gate: enabled (threshold={args.entropy_gate} nats)")
    else:
        print("Entropy gate: disabled")
    print()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    model.to("cpu")

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; cannot run safely.")

    bad_ids = build_bad_token_ids(tokenizer, args.forbidden, args.case_insensitive)
    print(f"Bad token ids: {bad_ids.numel()} / vocab_size={len(tokenizer)}")

    lp = LogitsProcessorList()
    lp.append(ForbiddenSubstrLogitsProcessor(bad_ids))
    if args.entropy_gate is not None:
        lp.append(EntropyGateLogitsProcessor(args.entropy_gate, tokenizer.eos_token_id))

    # These prompts are intentionally "adversarial" for the forbidden-letter test:
    # they strongly bias the baseline to emit the forbidden substring, making the
    # control-vs-constraint contrast obvious.
    prompts = [
        "Write the word 'ocean' three times, each on its own line.",
        "Write a 4-line poem about the ocean.",
        "In one paragraph, describe the ocean and the moon.",
        "Answer with exactly 2 sentences: Why do oceans have waves?",
        "Give a short list of 8 single words related to oceans.",
        "Write a cheerful slogan about going to the ocean.",
        "Write a tiny story about a dolphin.",
        "Write a JSON object with keys ocean, color, mood (and short string values).",
    ]
    random.shuffle(prompts)
    prompts = prompts[: max(1, min(args.trials, len(prompts)))]

    results: list[RunResult] = []
    for i, prompt in enumerate(prompts):
        # Use per-trial seeds so baseline/constrained are comparable and reproducible.
        trial_seed = args.seed + i
        baseline = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=trial_seed,
            logits_processor=None,
        )
        constrained = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=trial_seed,
            logits_processor=lp,
        )

        baseline_ok = not contains_forbidden(baseline, args.forbidden, args.case_insensitive)
        constrained_ok = not contains_forbidden(constrained, args.forbidden, args.case_insensitive)
        results.append(
            RunResult(
                prompt=prompt,
                baseline=baseline,
                constrained=constrained,
                baseline_ok=baseline_ok,
                constrained_ok=constrained_ok,
            )
        )

    dt = time.time() - t0
    b_ok = sum(r.baseline_ok for r in results)
    c_ok = sum(r.constrained_ok for r in results)
    print()
    print("=== Results ===")
    print(f"Trials: {len(results)}")
    print(f"Baseline adherence:   {b_ok}/{len(results)}")
    print(f"Constrained adherence:{c_ok}/{len(results)}")
    print(f"Wall time: {dt:.1f}s")
    print()

    for idx, r in enumerate(results, start=1):
        print(f"--- Trial {idx} ---")
        print(f"Prompt: {r.prompt}")
        print(f"Baseline ok={r.baseline_ok}:")
        print(r.baseline.strip() or "<EMPTY>")
        print()
        print(f"Constrained ok={r.constrained_ok}:")
        print(r.constrained.strip() or "<EMPTY>")
        print()


if __name__ == "__main__":
    # Avoid any accidental nondeterminism from threading
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    main()

