import torch

from rumimem.memory.neural_memory import GatedNeuralMemory


def make_memory():
    torch.manual_seed(7)
    return GatedNeuralMemory(
        key_dim=3,
        value_dim=2,
        hidden_dim=16,
        learning_rate=0.05,
    )


def test_gates_are_in_valid_ranges():
    module = make_memory()
    keys = torch.randn(5, 3)

    gates = module.update_controller(keys)
    decay = torch.sigmoid(gates[:, 0])
    momentum = torch.sigmoid(gates[:, 1])
    strength = torch.nn.functional.softplus(gates[:, 2])

    assert torch.all((decay > 0) & (decay < 1))
    assert torch.all((momentum > 0) & (momentum < 1))
    assert torch.all(strength > 0)


def test_gated_write_updates_memory():
    module = make_memory()
    state = module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0]])

    updated = module.write(state, keys, values)

    assert updated.steps == 1
    assert any(
        not torch.equal(updated.params[name], state.params[name])
        for name in state.params
    )


def test_separate_gated_states_remain_independent():
    module = make_memory()
    state_a = module.initial_state()
    state_b = module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0]])

    before_b = {
        name: value.detach().clone()
        for name, value in state_b.params.items()
    }

    state_a = module.write(state_a, keys, values)

    assert state_a.steps == 1
    assert state_b.steps == 0

    for name in state_b.params:
        assert torch.equal(state_b.params[name], before_b[name])


def test_outer_loss_can_train_update_controller():
    module = make_memory()
    state = module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0]])

    updated = module.write(
        state, keys, values, differentiable=True
    )
    prediction = module.read(updated, keys)
    outer_loss = (prediction - values).square().mean()
    outer_loss.backward()

    gradient = module.update_controller.weight.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0


def test_gated_write_rejects_mismatched_values():
    module = make_memory()
    state = module.initial_state()
    keys = torch.randn(2, 3)
    values = torch.randn(1, 2)

    try:
        module.write(state, keys, values)
    except ValueError as exc:
        assert "values must match" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched values.")
