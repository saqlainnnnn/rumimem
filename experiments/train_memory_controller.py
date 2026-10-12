"""Train memory update gates with a held-out outer-loop objective."""
import statistics
import torch
from rumimem.memory.neural_memory import GatedNeuralMemory

FACTS, KEY_DIM, VALUE_DIM = 4, 16, 32
PRETRAIN_ROUNDS, CORRECTION_WRITES = 5, 3
TRAIN_EPISODES, EVAL_SEEDS = 100, [101, 202, 303, 404, 505]


def episode(model, seed, training):
    torch.manual_seed(seed)
    keys = torch.randn(FACTS, KEY_DIM)
    values = torch.randn(FACTS, VALUE_DIM)
    corrected_value = torch.randn(1, VALUE_DIM)
    target_idx = int(torch.randint(FACTS, ()).item())
    state = model.initial_state()

    for _ in range(PRETRAIN_ROUNDS):
        for i in torch.randperm(FACTS).tolist():
            state = model.write(state, keys[i:i + 1], values[i:i + 1],
                                differentiable=training)

    for _ in range(CORRECTION_WRITES):
        state = model.write(
            state, keys[target_idx:target_idx + 1], corrected_value,
            differentiable=training,
        )

    predictions = model.read(state, keys)
    target_error = (
        predictions[target_idx:target_idx + 1] - corrected_value
    ).square().mean()
    keep = [i for i in range(FACTS) if i != target_idx]
    retention_error = (predictions[keep] - values[keep]).square().mean()
    return target_error + 0.5 * retention_error, target_error, retention_error


def evaluate(model):
    model.eval()
    rows = []
    with torch.enable_grad():
        for seed in EVAL_SEEDS:
            _, target, retention = episode(model, seed, training=False)
            rows.append((target.item(), retention.item()))
    return tuple(
        (statistics.mean(row[i] for row in rows),
         statistics.stdev(row[i] for row in rows))
        for i in range(2)
    )


def show(label, result):
    target, retention = result
    print(
        f"{label:<18} "
        f"correction MSE={target[0]:.5f} +/- {target[1]:.5f}  "
        f"retention MSE={retention[0]:.5f} +/- {retention[1]:.5f}"
    )


def main():
    torch.manual_seed(7)
    model = GatedNeuralMemory(
        key_dim=KEY_DIM, value_dim=VALUE_DIM,
        hidden_dim=64, learning_rate=0.1,
    )
    initial_controller = {
        name: p.detach().clone()
        for name, p in model.update_controller.named_parameters()
    }

    print("Held-out evaluation before outer-loop training")
    before = evaluate(model)
    show("Untrained gates", before)

    optimizer = torch.optim.Adam(model.update_controller.parameters(), lr=1e-3)
    model.train()

    for step in range(TRAIN_EPISODES):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = episode(model, seed=10_000 + step, training=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.update_controller.parameters(), max_norm=1.0
        )
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Episode {step + 1:3d}/{TRAIN_EPISODES}: "
                  f"outer loss={loss.detach().item():.5f}")

    trained_controller = {
        name: p.detach().clone()
        for name, p in model.update_controller.named_parameters()
    }

    with torch.no_grad():
        for name, p in model.update_controller.named_parameters():
            p.copy_(initial_controller[name])
    before = evaluate(model)

    with torch.no_grad():
        for name, p in model.update_controller.named_parameters():
            p.copy_(trained_controller[name])
    after = evaluate(model)

    print("\nFinal comparison on the same held-out episodes")
    show("Untrained gates", before)
    show("Trained gates", after)
    print("\nLower is better for both metrics. Compare both, not correction "
          "MSE alone.")


if __name__ == "__main__":
    main()
