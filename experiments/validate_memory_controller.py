"""Validate outer-loop-trained gates against matched memory baselines."""
import math
import statistics
import torch

from rumimem.memory.neural_memory import NeuralMemory, GatedNeuralMemory

FACTS, KEY_DIM, VALUE_DIM = 4, 16, 32
ROUNDS, CORRECTION_WRITES = 5, 3
TRAIN_STEPS, BATCH_SIZE = 100, 4
EVAL_SEEDS = list(range(9001, 9021))
MOMENTUM, FORGETTING, LR = 0.5, 0.0005, 0.1


def make_gated():
    torch.manual_seed(77)
    model = GatedNeuralMemory(KEY_DIM, VALUE_DIM, 64, LR)
    with torch.no_grad():
        model.update_controller.bias[0] = math.log(
            FORGETTING / (1 - FORGETTING)
        )
        model.update_controller.bias[1] = 0.0  # momentum = 0.5
        model.update_controller.bias[2] = math.log(math.expm1(LR))
    return model


def run_episode(model, seed, differentiable=False):
    torch.manual_seed(seed)
    keys = torch.randn(FACTS, KEY_DIM)
    values = torch.randn(FACTS, VALUE_DIM)
    replacement = torch.randn(1, VALUE_DIM)
    target = int(torch.randint(FACTS, ()).item())
    state = model.initial_state()

    for _ in range(ROUNDS):
        for i in torch.randperm(FACTS).tolist():
            state = model.write(
                state, keys[i:i + 1], values[i:i + 1],
                **({"differentiable": True} if differentiable else {}),
            )

    for _ in range(CORRECTION_WRITES):
        state = model.write(
            state, keys[target:target + 1], replacement,
            **({"differentiable": True} if differentiable else {}),
        )

    pred = model.read(state, keys)
    keep = [i for i in range(FACTS) if i != target]
    correction = (pred[target:target + 1] - replacement).square().mean()
    retention = (pred[keep] - values[keep]).square().mean()
    return correction + 0.5 * retention, correction, retention


def evaluate(model, seeds):
    rows = []
    for seed in seeds:
        _, correction, retention = run_episode(model, seed)
        rows.append((correction.item(), retention.item()))
    return rows


def report(name, rows):
    corr = [r[0] for r in rows]
    retain = [r[1] for r in rows]
    print(
        f"{name:<24} "
        f"{statistics.mean(corr):.5f} +/- {statistics.stdev(corr):.5f}  "
        f"{statistics.mean(retain):.5f} +/- {statistics.stdev(retain):.5f}"
    )


def main():
    gated = make_gated()
    initial_controller = {
        n: p.detach().clone()
        for n, p in gated.update_controller.named_parameters()
    }

    fixed = NeuralMemory(
        KEY_DIM, VALUE_DIM, 64, learning_rate=LR,
        momentum=MOMENTUM, forgetting=FORGETTING,
    )
    # Match the fixed model's base-network initialization to the gated model.
    torch.manual_seed(77)
    fixed = NeuralMemory(
        KEY_DIM, VALUE_DIM, 64, learning_rate=LR,
        momentum=MOMENTUM, forgetting=FORGETTING,
    )

    print(f"Evaluation on {len(EVAL_SEEDS)} held-out seeds")
    print("Variant                  Correction MSE             Retention MSE")
    report("Fixed momentum", evaluate(fixed, EVAL_SEEDS))
    report("Gates before training", evaluate(gated, EVAL_SEEDS))

    optimizer = torch.optim.Adam(gated.update_controller.parameters(), lr=1e-3)
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for j in range(BATCH_SIZE):
            seed = 20_000 + step * BATCH_SIZE + j
            loss, _, _ = run_episode(gated, seed, differentiable=True)
            losses.append(loss)
        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            gated.update_controller.parameters(), max_norm=1.0
        )
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Training step {step + 1:3d}: "
                  f"batch outer loss={batch_loss.detach().item():.5f}")

    report("Gates after training", evaluate(gated, EVAL_SEEDS))


if __name__ == "__main__":
    main()
