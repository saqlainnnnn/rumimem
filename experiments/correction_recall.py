"""Sweep correction strength against retention of unrelated facts."""

import statistics
import torch

from rumimem.memory.neural_memory import ProjectedNeuralMemory

FACTS = 8
DIM = 32
EPOCHS = 50
CORRECTION_WRITES = [1, 2, 5, 10, 25, 50]
LR = 0.1
SEEDS = [1, 7, 21, 42, 123]


def run(seed):
    torch.manual_seed(seed)

    model = ProjectedNeuralMemory(
        model_dim=DIM,
        key_dim=16,
        value_dim=DIM,
        memory_hidden_dim=64,
        learning_rate=LR,
    )
    state = model.initial_state()
    keys = torch.randn(FACTS, DIM)
    values = torch.randn(FACTS, DIM)
    corrected = torch.randn(1, DIM)

    for _ in range(EPOCHS):
        for i in torch.randperm(FACTS).tolist():
            state = model.condition(
                state, keys[i:i + 1], values[i:i + 1]
            )

    queries = model.query_proj(keys)
    old_targets = model.value_proj(values).detach()
    corrected_target = model.value_proj(corrected).detach()[0]

    def predict(current_state):
        with torch.no_grad():
            return torch.cat([
                model.memory.read(
                    current_state.memory, queries[i:i + 1]
                )
                for i in range(FACTS)
            ])

    before = predict(state)
    retention_before = (
        before[1:] - old_targets[1:]
    ).square().mean().item()

    results = []
    for writes in CORRECTION_WRITES:
        # Branch from the same pre-correction state for a fair comparison.
        corrected_state = state
        for _ in range(writes):
            corrected_state = model.condition(
                corrected_state, keys[:1], corrected
            )

        after = predict(corrected_state)
        correction_mse = (
            after[0] - corrected_target
        ).square().mean().item()
        retention_after = (
            after[1:] - old_targets[1:]
        ).square().mean().item()

        results.append((writes, correction_mse, retention_after))

    return retention_before, results


def main():
    runs = [run(seed) for seed in SEEDS]
    print("Correction-write sweep (mean +/- sample standard deviation)")
    print(f"{'Writes':>7} {'Corrected MSE':>20} {'Other facts MSE':>20}")

    for index, writes in enumerate(CORRECTION_WRITES):
        correction = [row[1][index][1] for row in runs]
        retention = [row[1][index][2] for row in runs]

        print(
            f"{writes:>7} "
            f"{statistics.mean(correction):.5f} +/- "
            f"{statistics.stdev(correction):.5f} "
            f"{statistics.mean(retention):.5f} +/- "
            f"{statistics.stdev(retention):.5f}"
        )

    before = [row[0] for row in runs]
    print(
        "Other facts before correction: "
        f"{statistics.mean(before):.5f} +/- "
        f"{statistics.stdev(before):.5f}"
    )


if __name__ == "__main__":
    main()
