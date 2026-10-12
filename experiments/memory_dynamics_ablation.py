"""Controlled ablation of neural-memory update dynamics."""
import statistics
import torch

from rumimem.memory.neural_memory import (
    NeuralMemory,
    GatedNeuralMemory,
)

FACTS = 8
KEY_DIM = 16
VALUE_DIM = 32
HIDDEN_DIM = 64
PRETRAIN_WRITES = 50
CORRECTION_BUDGETS = [25, 50]
SEEDS = [1, 7, 21, 42, 123]

VARIANTS = {
    "fixed": {"momentum": 0.0, "forgetting": 0.0},
    "momentum": {"momentum": 0.9, "forgetting": 0.0},
    "forgetting": {"momentum": 0.0, "forgetting": 0.02},
    "momentum+forgetting": {"momentum": 0.9, "forgetting": 0.02},
    "gated": {},
}


def make_model(name, seed):
    # Reset initialization so all variants share the same base network.
    torch.manual_seed(10_000 + seed)

    if name == "gated":
        return GatedNeuralMemory(
            key_dim=KEY_DIM,
            value_dim=VALUE_DIM,
            hidden_dim=HIDDEN_DIM,
            learning_rate=0.1,
        )

    return NeuralMemory(
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        hidden_dim=HIDDEN_DIM,
        learning_rate=0.1,
        **VARIANTS[name],
    )


def run_episode(seed, variant, correction_budget):
    torch.manual_seed(seed)

    keys = torch.randn(FACTS, KEY_DIM)
    values = torch.randn(FACTS, VALUE_DIM)
    corrected_value = torch.randn(1, VALUE_DIM)

    model = make_model(variant, seed)
    state = model.initial_state()

    # Balanced pretraining: each fact is written the same number of times.
    for _ in range(PRETRAIN_WRITES):
        for i in torch.randperm(FACTS).tolist():
            state = model.write(
                state, keys[i:i + 1], values[i:i + 1]
            )

    def predictions(current_state):
        with torch.no_grad():
            return model.read(current_state, keys)

    before = predictions(state)
    retention_before = (
        before[1:] - values[1:]
    ).square().mean().item()

    # Correct fact 0 without changing the other targets.
    for _ in range(correction_budget):
        state = model.write(
            state, keys[:1], corrected_value
        )

    after = predictions(state)

    correction_error = (
        after[0:1] - corrected_value
    ).square().mean().item()

    retention_after = (
        after[1:] - values[1:]
    ).square().mean().item()

    return correction_error, retention_before, retention_after


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    print("Memory dynamics ablation")
    print(
        f"Facts={FACTS}, pretraining={PRETRAIN_WRITES} writes/fact, "
        f"seeds={len(SEEDS)}"
    )
    print("Gated controller is NOT outer-loop trained.")
    print()

    header = (
        f"{'Variant':<22} {'Writes':>6} "
        f"{'Correction MSE':>20} {'Retention before':>20} "
        f"{'Retention after':>20} {'Retention delta':>17}"
    )
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        for budget in CORRECTION_BUDGETS:
            results = [
                run_episode(seed, variant, budget)
                for seed in SEEDS
            ]

            correction = [r[0] for r in results]
            before = [r[1] for r in results]
            after = [r[2] for r in results]
            deltas = [a - b for a, b in zip(after, before)]

            print(
                f"{variant:<22} {budget:>6} "
                f"{mean_std(correction):>20} "
                f"{mean_std(before):>20} "
                f"{mean_std(after):>20} "
                f"{statistics.mean(deltas):>+17.4f}"
            )


if __name__ == "__main__":
    main()
