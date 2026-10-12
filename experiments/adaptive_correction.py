"""Compare fixed-count and error-controlled fact correction."""

import statistics
import torch

from rumimem.memory.neural_memory import ProjectedNeuralMemory

FACTS = 8
DIM = 32
PRETRAIN_WRITES = 50
LR = 0.1
MAX_CORRECTION_WRITES = 100
THRESHOLDS = [0.10, 0.05, 0.02]
SEEDS = [1, 7, 21, 42, 123]


def run(seed, threshold):
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

    for _ in range(PRETRAIN_WRITES):
        for i in torch.randperm(FACTS).tolist():
            state = model.condition(
                state, keys[i:i + 1], values[i:i + 1]
            )

    queries = model.query_proj(keys)
    old_targets = model.value_proj(values).detach()
    corrected_target = model.value_proj(corrected).detach()

    def read_all(current_state):
        with torch.no_grad():
            return torch.cat([
                model.memory.read(
                    current_state.memory, queries[i:i + 1]
                )
                for i in range(FACTS)
            ])

    before = read_all(state)
    retention_before = (
        before[1:] - old_targets[1:]
    ).square().mean().item()

    writes = 0
    while writes < MAX_CORRECTION_WRITES:
        state = model.condition(state, keys[:1], corrected)
        writes += 1

        prediction = model.memory.read(
            state.memory, queries[:1]
        )
        error = (
            prediction.detach() - corrected_target
        ).square().mean().item()

        if error <= threshold:
            break

    after = read_all(state)
    correction_error = (
        after[0] - corrected_target[0]
    ).square().mean().item()
    retention_after = (
        after[1:] - old_targets[1:]
    ).square().mean().item()

    return writes, correction_error, retention_before, retention_after


def main():
    print("Error-controlled correction")
    print(f"Pretraining: {PRETRAIN_WRITES} writes/fact; LR={LR}")
    print(f"{'Threshold':>10} {'Writes mean':>12} "
          f"{'Correction MSE':>16} {'Other facts before':>19} "
          f"{'Other facts after':>18}")

    for threshold in THRESHOLDS:
        results = [run(seed, threshold) for seed in SEEDS]
        print(
            f"{threshold:>10.3f} "
            f"{statistics.mean(r[0] for r in results):>12.1f} "
            f"{statistics.mean(r[1] for r in results):>16.5f} "
            f"{statistics.mean(r[2] for r in results):>19.5f} "
            f"{statistics.mean(r[3] for r in results):>18.5f}"
        )


if __name__ == "__main__":
    main()
