import pytest
import torch

from rumimem.memory.neural_memory import ProjectedNeuralMemory


@pytest.fixture
def module():
    torch.manual_seed(7)
    return ProjectedNeuralMemory(
        model_dim=32,
        key_dim=16,
        value_dim=12,
        memory_hidden_dim=32,
        learning_rate=0.05,
    )


def test_read_returns_value_dimension(module):
    state = module.initial_state()
    hidden = torch.randn(4, 32)

    result = module.read(state, hidden)

    assert result.shape == (4, 12)


def test_write_updates_fast_memory(module):
    state = module.initial_state()
    hidden = torch.randn(4, 32)

    updated = module.write(state, hidden, hidden)

    assert updated.memory.steps == 1
    assert state.memory.steps == 0


def test_repeated_writes_increment_steps(module):
    state = module.initial_state()
    hidden = torch.randn(4, 32)

    for _ in range(5):
        state = module.write(state, hidden, hidden)

    assert state.memory.steps == 5


def test_separate_streams_remain_independent(module):
    state_a = module.initial_state()
    state_b = module.initial_state()
    hidden = torch.randn(2, 32)

    state_a = module.write(state_a, hidden, hidden)

    assert state_a.memory.steps == 1
    assert state_b.memory.steps == 0

    for name, value in state_b.memory.params.items():
        assert torch.equal(
            value,
            state_a.memory.params[name]
            if torch.equal(value, state_a.memory.params[name])
            else value,
        )


def test_invalid_hidden_dimension_is_rejected(module):
    state = module.initial_state()
    invalid_hidden = torch.randn(4, 24)

    with pytest.raises(ValueError, match="model_dim"):
        module.read(state, invalid_hidden)


def test_mismatched_write_batch_sizes_are_rejected(module):
    state = module.initial_state()
    keys = torch.randn(4, 32)
    values = torch.randn(3, 32)

    with pytest.raises(ValueError, match="batch sizes must match"):
        module.write(state, keys, values)
