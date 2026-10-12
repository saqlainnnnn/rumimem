import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rumimem.models.hf_integration import RumiMemCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"

ENTITIES = [
    "red fox", "blue whale", "green owl", "silver wolf",
    "golden eagle", "black panther", "white tiger", "brown bear",
    "orange cat", "purple raven", "yellow crane", "gray dolphin",
]

def score_candidates(forward, tokenizer, prompt, candidates, device):
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]
    results = []

    for candidate in candidates:
        continuation_ids = tokenizer(
            candidate, add_special_tokens=False
        )["input_ids"]
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
        results.append(log_probs[positions, targets].sum().item())

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    base.eval()

    model = RumiMemCausalLM(
        base_model=base,
        layer_index=15,
        key_dim=128,
        memory_hidden_dim=128,
        learning_rate=0.01,
    ).to(device)
    model.eval()

    totals = {
        "base": {"correct": 0, "margin": []},
        "empty": {"correct": 0, "margin": []},
        "written": {"correct": 0, "margin": []},
    }

    for episode in range(args.episodes):
        count = min(args.distractors + 1, len(ENTITIES))
        selected = random.sample(ENTITIES, count)
        target = selected[0]
        codes = random.sample(range(1000, 9999), count)
        facts = [
            f"The secret code for the {entity} is {code}."
            for entity, code in zip(selected, codes)
        ]
        support_text = " ".join(facts)

        correct = str(codes[0])
        wrong = set()
        while len(wrong) < 3:
            value = str(random.randint(1000, 9998))
            if value != correct:
                wrong.add(value)

        candidates = [f" {correct}"] + [f" {x}" for x in wrong]
        random.shuffle(candidates)
        correct_candidate = f" {correct}"
        correct_index = candidates.index(correct_candidate)

        prompt = (
            f"Question: What is the secret code for the {target}? Answer:"
        )
        support_ids = tokenizer(
            support_text, return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(device)
        query_ids = tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(device)

        state = model.initial_memory_state()
        written_state = model.write_memory(
            state, support_ids, chunk_size=16
        )

        score_sets = {
            "base": score_candidates(
                lambda ids: base(input_ids=ids, use_cache=False),
                tokenizer, prompt, candidates, device
            ),
            "empty": score_candidates(
                lambda ids: model(
                    input_ids=ids, memory_state=state,
                    use_memory=True
                ),
                tokenizer, prompt, candidates, device
            ),
            "written": score_candidates(
                lambda ids: model(
                    input_ids=ids, memory_state=written_state,
                    use_memory=True
                ),
                tokenizer, prompt, candidates, device
            ),
        }

        for name, scores in score_sets.items():
            ranking = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            totals[name]["correct"] += ranking[0] == correct_index
            best_wrong = max(
                score for i, score in enumerate(scores)
                if i != correct_index
            )
            totals[name]["margin"].append(scores[correct_index] - best_wrong)

        print(
            f"{episode + 1:02d}/{args.episodes} "
            + " ".join(
                f"{name}={'✓' if score_sets[name].index(max(score_sets[name])) == correct_index else '✗'}"
                for name in ("base", "empty", "written")
            )
        )

    print("\n=== RESULTS ===")
    for name, result in totals.items():
        print(
            f"{name:>7}: accuracy={result['correct'] / args.episodes:.1%}, "
            f"mean margin={sum(result['margin']) / len(result['margin']):+.4f}"
        )

if __name__ == "__main__":
    main()
