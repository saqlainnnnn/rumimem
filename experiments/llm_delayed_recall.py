import random
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rumimem.models.hf_integration import RumiMemCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"

ENTITIES = [
    ("red fox", "code"),
    ("blue whale", "code"),
    ("green owl", "code"),
    ("silver wolf", "code"),
    ("golden eagle", "code"),
    ("black panther", "code"),
    ("white tiger", "code"),
    ("brown bear", "code"),
    ("orange cat", "code"),
    ("purple raven", "code"),
    ("yellow crane", "code"),
    ("gray dolphin", "code"),
]

def score_candidates(forward, tokenizer, prompt, candidates, device):
    """Score candidate continuations by summed conditional log-likelihood."""
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]

    scores = []
    for candidate in candidates:
        continuation_ids = tokenizer(
            candidate, add_special_tokens=False
        )["input_ids"]

        if not prompt_ids or not continuation_ids:
            raise ValueError("Prompt and candidate must tokenize to nonempty IDs.")

        ids = torch.tensor(
            [prompt_ids + continuation_ids], device=device
        )

        with torch.no_grad():
            logits = forward(ids).logits[0].float()
            log_probs = logits.log_softmax(dim=-1)

        start = len(prompt_ids) - 1
        positions = torch.arange(
            start, start + len(continuation_ids), device=device
        )
        targets = torch.tensor(continuation_ids, device=device)
        score = log_probs[positions, targets].sum().item()
        scores.append(score)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device)
    base.eval()

    rumimem = RumiMemCausalLM(
        base_model=base,
        layer_index=15,
        key_dim=128,
        memory_hidden_dim=128,
        learning_rate=0.01,
    ).to(device)
    rumimem.eval()

    baseline_correct = 0
    memory_correct = 0
    margins = []

    for episode in range(args.episodes):
        entities = random.sample(
            ENTITIES, k=min(args.distractors + 1, len(ENTITIES))
        )
        target_entity, relation = entities[0]
        codes = random.sample(range(1000, 9999), k=len(entities))
        facts = [
            f"The secret {rel} for the {entity} is {code}."
            for (entity, rel), code in zip(entities, codes)
        ]
        random.shuffle(facts)
        support_text = " ".join(facts)

        correct_code = str(codes[0])
        wrong_codes = set()
        while len(wrong_codes) < 3:
            candidate = str(random.randint(1000, 9998))
            if candidate != correct_code:
                wrong_codes.add(candidate)

        candidates = [f" {correct_code}"] + [
            f" {code}" for code in wrong_codes
        ]
        random.shuffle(candidates)

        prompt = (
            f"Question: What is the secret {relation} for the "
            f"{target_entity}? Answer:"
        )

        support_ids = tokenizer(
            support_text, return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(device)

        state = rumimem.initial_memory_state()
        state = rumimem.write_memory(
            state, support_ids, chunk_size=16
        )

        base_scores = score_candidates(
            lambda ids: base(input_ids=ids, use_cache=False),
            tokenizer, prompt, candidates, device
        )
        memory_scores = score_candidates(
            lambda ids: rumimem(
                input_ids=ids, memory_state=state,
                use_memory=True
            ),
            tokenizer, prompt, candidates, device
        )

        correct_candidate = f" {correct_code}"
        base_choice = candidates[max(
            range(len(candidates)), key=lambda i: base_scores[i]
        )]
        memory_choice = candidates[max(
            range(len(candidates)), key=lambda i: memory_scores[i]
        )]

        baseline_correct += base_choice == correct_candidate
        memory_correct += memory_choice == correct_candidate

        correct_idx = candidates.index(correct_candidate)
        best_wrong_score = max(
            score for i, score in enumerate(memory_scores)
            if i != correct_idx
        )
        margins.append(memory_scores[correct_idx] - best_wrong_score)

        print(
            f"Episode {episode + 1:02d}/{args.episodes}: "
            f"base={'✓' if base_choice == correct_candidate else '✗'} "
            f"RumiMem={'✓' if memory_choice == correct_candidate else '✗'} "
            f"margin={margins[-1]:+.3f}"
        )

    print("\n=== DELAYED RECALL RESULTS ===")
    print(f"Episodes: {args.episodes}")
    print(f"Distractor facts requested: {args.distractors}")
    print(f"Base accuracy:    {baseline_correct / args.episodes:.1%}")
    print(f"RumiMem accuracy: {memory_correct / args.episodes:.1%}")
    print(
        "Mean RumiMem correct-vs-best-wrong log-likelihood margin: "
        f"{sum(margins) / len(margins):+.4f}"
    )


if __name__ == "__main__":
    main()
