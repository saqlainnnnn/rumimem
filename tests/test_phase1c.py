import pytest
import torch

from rumimem.memory.neural_memory import (
    NeuralMemory,
    ProjectedNeuralMemory,
)


def make_memory():
    torch.manual_seed(17)
    return NeuralMemory(
        key_dim=3,
        value_dim=2,
        hidden_dim=16,
        learning_rate=0.05,
    )


def test_multiple_associations_can_be_learned():
    module = make_memory()
    state = module.initial_state()

    keys = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    values = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    for _ in range(150):
        state = module.write(state, keys, values)

    predictions = module.read(state, keys)
    mse = (predictions - values).square().mean().item()

    assert state.steps == 150
    assert mse < 0.02


def test_correction_replaces_an_old_association():
    module = make_memory()
    state = module.initial_state()

    key = torch.tensor([[1.0, 0.0, 0.0]])
    old_value = torch.tensor([[1.0, 0.0]])
    new_value = torch.tensor([[0.0, 1.0]])

    for _ in range(100):
        state = module.write(state, key, old_value)

    old_error_before = (
        module.read(state, key) - old_value
    ).square().mean().item()

    for _ in range(200):
        state = module.write(state, key, new_value)

    prediction = module.read(state, key)
    new_error = (prediction - new_value).square().mean().item()
    old_error_after = (
        prediction - old_value
    ).square().mean().item()

    assert new_error < old_error_before
    assert new_error < old_error_after
    assert state.steps == 300


def test_condition_matches_manual_chunk_writes():
    module = ProjectedNeuralMemory(
        model_dim=8,
        key_dim=4,
        value_dim=3,
        memory_hidden_dim=8,
        learning_rate=0.02,
    )
    state_a = module.initial_state()
    state_b = module.initial_state()

    torch.manual_seed(23)
    hidden_keys = torch.randn(11, 8)
    hidden_values = torch.randn(11, 8)
    chunk_size = 4

    state_a = module.condition(
        state_a, hidden_keys, hidden_values, chunk_size=chunk_size
    )

    for start in range(0, len(hidden_keys), chunk_size):
        end = start + chunk_size
        state_b = module.write(
            state_b,
            hidden_keys[start:end],
            hidden_values[start:end],
        )

    assert state_a.memory.steps == 3
    assert state_b.memory.steps == 3

    for name in state_a.memory.params:
        torch.testing.assert_close(
            state_a.memory.params[name],
            state_b.memory.params[name],
        )
        torch.testing.assert_close(
            state_a.memory.surprise[name],
            state_b.memory.surprise[name],
        )


def test_condition_rejects_invalid_chunk_size():
    module = ProjectedNeuralMemory(
        model_dim=8,
        key_dim=4,
        value_dim=3,
        memory_hidden_dim=8,
    )
    state = module.initial_state()
    hidden = torch.randn(4, 8)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        module.condition(state, hidden, hidden, chunk_size=0)


def test_condition_rejects_mismatched_sequence_lengths():
    module = ProjectedNeuralMemory(
        model_dim=8,
        key_dim=4,
        value_dim=3,
        memory_hidden_dim=8,
    )
    state = module.initial_state()

    with pytest.raises(ValueError, match="sequence lengths must match"):
        module.condition(
            state,
            torch.randn(4, 8),
            torch.randn(3, 8),
            chunk_size=2,
        )


def test_writing_one_projected_stream_does_not_mutate_another():
    module = ProjectedNeuralMemory(
        model_dim=8,
        key_dim=4,
        value_dim=3,
        memory_hidden_dim=8,
    )
    state_a = module.initial_state()
    state_b = module.initial_state()
    hidden = torch.randn(3, 8)

    before_b = {
        name: parameter.detach().clone()
        for name, parameter in state_b.memory.params.items()
    }

    state_a = module.write(state_a, hidden, hidden)

    assert state_a.memory.steps == 1
    assert state_b.memory.steps == 0

    for name, parameter in state_b.memory.params.items():
        assert torch.equal(parameter, before_b[name])

    assert any(
        not torch.equal(state_a.memory.params[name], before_b[name])
        for name in before_b
    )
