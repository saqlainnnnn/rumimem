"""Learning-curve and interference diagnostic for RumiMem."""

import torch
from rumimem.memory.neural_memory import ProjectedNeuralMemory

MODEL_DIM = 32
KEY_DIM = 16
HIDDEN_DIM = 64
LEARNING_RATES = [0.01, 0.05, 0.1]
WRITE_COUNTS = [1, 10, 50, 100]
DISTRACTORS = 16


def run(lr: float, writes: int) -> tuple[float, float, float]:
    torch.manual_seed(42)

    memory = ProjectedNeuralMemory(
        model_dim=MODEL_DIM,
        key_dim=KEY_DIM,
        value_dim=MODEL_DIM,
        memory_hidden_dim=HIDDEN_DIM,
        learning_rate=lr,
    )
    state = memory.initial_state()

    key = torch.randn(1, MODEL_DIM)
    value = torch.randn(1, MODEL_DIM)
    query = memory.query_proj(key)
    target = memory.value_proj(value).detach()

    for _ in range(writes):
        state = memory.condition(state, key, value)

    with torch.no_grad():
        immediate = memory.memory.read(state.memory, query)
        immediate_mse = (immediate - target).square().mean().item()
        baseline_mse = target.square().mean().item()

    # Interference from unrelated associations.
    for _ in range(DISTRACTORS):
        distractor_key = torch.randn(1, MODEL_DIM)
        distractor_value = torch.randn(1, MODEL_DIM)
        state = memory.condition(state, distractor_key, distractor_value)

    with torch.no_grad():
        delayed = memory.memory.read(state.memory, query)
        delayed_mse = (delayed - target).square().mean().item()

    return immediate_mse, delayed_mse, baseline_mse


def main() -> None:
    print("Learning curve and interference diagnostic")
    print("Lower MSE is better; zero baseline predicts zero for every query.")
    print(f"Distractor associations: {DISTRACTORS}\n")
    print(
        f"{'LR':>6} {'Writes':>7} {'Immediate':>12} "
        f"{'After distractors':>18} {'Zero baseline':>14}"
    )

    for lr in LEARNING_RATES:
        for writes in WRITE_COUNTS:
            immediate, delayed, baseline = run(lr, writes)
            print(
                f"{lr:>6.3f} {writes:>7} {immediate:>12.6f} "
                f"{delayed:>18.6f} {baseline:>14.6f}"
            )


if __name__ == "__main__":
    main()
