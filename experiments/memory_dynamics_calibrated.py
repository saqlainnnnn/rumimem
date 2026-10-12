"""Calibrated ablation of neural-memory update dynamics."""
import math
import statistics
import torch

from rumimem.memory.neural_memory import NeuralMemory, GatedNeuralMemory

FACTS = 8
KEY_DIM = 16
VALUE_DIM = 32
HIDDEN_DIM = 64
PRETRAIN_WRITES = 50
CORRECTION_BUDGETS = [25, 50]
SEEDS = [1, 7, 21, 42, 123]
FORGETTING = 0.0005
MOMENTUM = 0.5

VARIANTS = {
    "fixed": {},
    "momentum": {"momentum": MOMENTUM},
    "forgetting": {"forgetting": FORGETTING},
    "momentum+forgetting": {
        "momentum": MOMENTUM,
        "forgetting": FORGETTING,
    },
    "gated": {},
}


def make_model(name, seed):
    torch.manual_seed(10_000 + seed)

    if name == "gated":
        model = GatedNeuralMemory(
            key_dim=KEY_DIM,
            value_dim=VALUE_DIM,
            hidden_dim=HIDDEN_DIM,
            learning_rate=0.1,
        )
        # Calibrate the untrained controller's initial operating point.
        with torch.no_grad():
            model.update_controller.bias[0] = math.log(
                FORGETTING / (1.0 - FORGETTING)
            )
            model.update_controller.bias[1] = math.log(
                MOMENTUM / (1.0 - MOMENTUM)
            )
        return model

    return NeuralMemory(
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        hidden_dim=HIDDEN_DIM,
        learning_rate=0.1,
        **VARIANTS[name],
    )


def run_episode(seed, variant, budget):
    torch.manual_seed(seed)
    keys = torch.randn(FACTS, KEY_DIM)
    values = torch.randn(FACTS, VALUE_DIM)
    corrected_value = torch.randn(1, VALUE_DIM)

    model = make_model(variant, seed)
    state = model.initial_state()

    for _ in range(PRETRAIN_WRITES):
        for i in torch.randperm(FACTS).tolist():
            state = model.write(state, keys[i:i + 1], values[i:i + 1])

    def read_all(s):
        with torch.no_grad():
            return model.read(s, keys)

    before = read_all(state)
    retention_before = (before[1:] - values[1:]).square().mean().item()

    for _ in range(budget):
        state = model.write(state, keys[:1], corrected_value)

    after = read_all(state)
    correction_error = (after[:1] - corrected_value).square().mean().item()
    retention_after = (after[1:] - values[1:]).square().mean().item()

    return correction_error, retention_before, retention_after


def summarize(xs):
    return f"{statistics.mean(xs):.4f} +/- {statistics.stdev(xs):.4f}"


def main():
    print("Calibrated memory dynamics ablation")
    print(
        f"forgetting={FORGETTING}, momentum={MOMENTUM}, "
        f"seeds={len(SEEDS)}; gated controller is untrained"
    )
    print(
        f"{'Variant':<22} {'Writes':>6} {'Correction MSE':>20} "
        f"{'Retention before':>20} {'Retention after':>20} {'Delta':>12}"
    )

    for variant in VARIANTS:
        for budget in CORRECTION_BUDGETS:
            rows = [
                run_episode(seed, variant, budget)
                for seed in SEEDS
            ]
            correction = [r[0] for r in rows]
            before = [r[1] for r in rows]
            after = [r[2] for r in rows]
            delta = [a - b for a, b in zip(after, before)]

            print(
                f"{variant:<22} {budget:>6} "
                f"{summarize(correction):>20} "
                f"{summarize(before):>20} "
                f"{summarize(after):>20} "
                f"{statistics.mean(delta):>+12.4f}"
            )


if __name__ == "__main__":
    main()
