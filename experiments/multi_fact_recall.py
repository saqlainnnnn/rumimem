"""Multi-seed balanced recall benchmark."""

import statistics
import torch

from rumimem.memory.neural_memory import ProjectedNeuralMemory

MODEL_DIM = 32
KEY_DIM = 16
HIDDEN_DIM = 64
LEARNING_RATE = 0.1
NUM_EPOCHS = 50
NUM_FACTS = [1, 4, 8, 16]
SEEDS = [1, 7, 21, 42, 123]


def run(num_facts: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)

    memory = ProjectedNeuralMemory(
        model_dim=MODEL_DIM,
        key_dim=KEY_DIM,
        value_dim=MODEL_DIM,
        memory_hidden_dim=HIDDEN_DIM,
        learning_rate=LEARNING_RATE,
    )
    state = memory.initial_state()

    keys = torch.randn(num_facts, MODEL_DIM)
    values = torch.randn(num_facts, MODEL_DIM)

    for _ in range(NUM_EPOCHS):
        for index in torch.randperm(num_facts).tolist():
            state = memory.condition(
                state,
                keys[index:index + 1],
                values[index:index + 1],
            )

    with torch.no_grad():
        queries = memory.query_proj(keys)
        targets = memory.value_proj(values)

        predictions = torch.cat([
            memory.memory.read(state.memory, queries[i:i + 1])
            for i in range(num_facts)
        ], dim=0)

        mse = (predictions - targets).square().mean().item()
        baseline = targets.square().mean().item()

    return mse, baseline


def main() -> None:
    print("Multi-seed balanced recall benchmark")
    print(f"LR={LEARNING_RATE}; writes/fact={NUM_EPOCHS}; seeds={len(SEEDS)}")
    print(f"{'Facts':>6} {'MSE mean':>12} {'MSE std':>12} "
          f"{'Baseline':>12} {'Improvement':>13}")

    for count in NUM_FACTS:
        results = [run(count, seed) for seed in SEEDS]
        errors = [result[0] for result in results]
        baselines = [result[1] for result in results]

        mean_error = statistics.mean(errors)
        std_error = statistics.stdev(errors)
        mean_baseline = statistics.mean(baselines)
        improvement = 1.0 - mean_error / mean_baseline

        print(
            f"{count:>6} {mean_error:>12.5f} {std_error:>12.5f} "
            f"{mean_baseline:>12.5f} {improvement:>12.1%}"
        )


if __name__ == "__main__":
    main()
