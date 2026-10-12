"""Replicate trained memory-controller results across independent runs."""
import math
import statistics
import torch

from rumimem.memory.neural_memory import NeuralMemory, GatedNeuralMemory

FACTS, KEY_DIM, VALUE_DIM, HIDDEN_DIM = 4, 16, 32, 64
ROUNDS, CORRECTION_WRITES = 5, 3
TRAIN_STEPS, BATCH_SIZE = 100, 4
TRAIN_SEEDS = [17, 37, 73]
EVAL_SEEDS = list(range(9001, 9021))
LR, MOMENTUM, FORGETTING = 0.1, 0.5, 0.0005


def make_gated(init_seed):
    torch.manual_seed(init_seed)
    model = GatedNeuralMemory(KEY_DIM, VALUE_DIM, HIDDEN_DIM, LR)
    with torch.no_grad():
        model.update_controller.bias[0] = math.log(
            FORGETTING / (1.0 - FORGETTING)
        )
        model.update_controller.bias[1] = 0.0
        model.update_controller.bias[2] = math.log(math.expm1(LR))
    return model


def make_fixed(init_seed):
    torch.manual_seed(init_seed)
    return NeuralMemory(
        KEY_DIM, VALUE_DIM, HIDDEN_DIM,
        learning_rate=LR, momentum=MOMENTUM, forgetting=FORGETTING,
    )


def episode(model, seed, differentiable=False):
    torch.manual_seed(seed)
    keys = torch.randn(FACTS, KEY_DIM)
    values = torch.randn(FACTS, VALUE_DIM)
    replacement = torch.randn(1, VALUE_DIM)
    target = int(torch.randint(FACTS, ()).item())
    state = model.initial_state()

    kwargs = {"differentiable": True} if differentiable else {}

    for _ in range(ROUNDS):
        for i in torch.randperm(FACTS).tolist():
            state = model.write(
                state, keys[i:i + 1], values[i:i + 1], **kwargs
            )

    for _ in range(CORRECTION_WRITES):
        state = model.write(
            state, keys[target:target + 1], replacement, **kwargs
        )

    prediction = model.read(state, keys)
    keep = [i for i in range(FACTS) if i != target]
    correction = (
        prediction[target:target + 1] - replacement
    ).square().mean()
    retention = (
        prediction[keep] - values[keep]
    ).square().mean()
    return correction + 0.5 * retention, correction, retention


def evaluate(model):
    results = []
    for seed in EVAL_SEEDS:
        with torch.enable_grad():
            _, correction, retention = episode(model, seed)
        results.append((correction.item(), retention.item()))
    return results


def summarize(values):
    return (
        f"{statistics.mean(values):.5f} +/- "
        f"{statistics.stdev(values):.5f}"
    )


def main():
    all_deltas = []

    for run_number, train_seed in enumerate(TRAIN_SEEDS, start=1):
        init_seed = 5000 + train_seed
        fixed = make_fixed(init_seed)
        gated = make_gated(init_seed)

        fixed_results = evaluate(fixed)
        optimizer = torch.optim.Adam(
            gated.update_controller.parameters(), lr=1e-3
        )

        for step in range(TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for j in range(BATCH_SIZE):
                seed = train_seed * 100_000 + step * BATCH_SIZE + j
                loss, _, _ = episode(gated, seed, differentiable=True)
                losses.append(loss)

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                gated.update_controller.parameters(), max_norm=1.0
            )
            optimizer.step()

        trained_results = evaluate(gated)

        fixed_c = [row[0] for row in fixed_results]
        fixed_r = [row[1] for row in fixed_results]
        trained_c = [row[0] for row in trained_results]
        trained_r = [row[1] for row in trained_results]

        dc = [a - b for a, b in zip(trained_c, fixed_c)]
        dr = [a - b for a, b in zip(trained_r, fixed_r)]
        all_deltas.append((statistics.mean(dc), statistics.mean(dr)))

        print(f"\nTraining run {run_number}/3 (seed={train_seed})")
        print(f"{'Variant':<18} {'Correction MSE':>23} {'Retention MSE':>23}")
        print(
            f"{'Fixed momentum':<18} {summarize(fixed_c):>23} "
            f"{summarize(fixed_r):>23}"
        )
        print(
            f"{'Trained gates':<18} {summarize(trained_c):>23} "
            f"{summarize(trained_r):>23}"
        )
        print(
            f"Paired mean difference (trained - fixed): "
            f"correction={statistics.mean(dc):+.5f}, "
            f"retention={statistics.mean(dr):+.5f}"
        )

    print("\nAcross-run summary of paired mean differences")
    print("Negative values favor trained gates.")
    print(f"Correction: {summarize([x[0] for x in all_deltas])}")
    print(f"Retention:  {summarize([x[1] for x in all_deltas])}")


if __name__ == "__main__":
    main()
