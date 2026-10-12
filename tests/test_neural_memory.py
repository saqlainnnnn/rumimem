import pytest
import torch

from rumimem.memory.neural_memory import NeuralMemory


@pytest.fixture
def module():
    torch.manual_seed(7)
    return NeuralMemory(
        key_dim=3,
        value_dim=2,
        hidden_dim=16,
        learning_rate=0.05,
    )


def test_initial_state_predicts_zeros(module):
    state = module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])

    prediction = module.read(state, keys)

    assert prediction.shape == (1, 2)
    assert torch.allclose(prediction, torch.zeros(1, 2))


def test_repeated_writes_improve_retrieval(module):
    state = module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0]])

    initial_error = (module.read(state, keys) - values).square().mean()

    for _ in range(100):
        state = module.write(state, keys, values)

    final_error = (module.read(state, keys) - values).square().mean()

    assert state.steps == 100
    assert final_error < initial_error
    assert final_error.item() < 0.01


def test_read_does_not_mutate_state(module):
    state = module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])

    params_before = {
        name: value.detach().clone()
        for name, value in state.params.items()
    }
    surprise_before = {
        name: value.clone()
        for name, value in state.surprise.items()
    }

    module.read(state, keys)

    assert state.steps == 0
    for name in state.params:
        assert torch.equal(state.params[name], params_before[name])
        assert torch.equal(state.surprise[name], surprise_before[name])


def test_memory_states_are_independent(module):
    state_a = module.initial_state()
    state_b = module.initial_state()

    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0]])

    state_b_before = {
        name: value.detach().clone()
        for name, value in state_b.params.items()
    }

    state_a = module.write(state_a, keys, values)

    assert state_a.steps == 1
    assert state_b.steps == 0

    # The untouched state must remain unchanged.
    for name in state_b.params:
        assert torch.equal(state_b.params[name], state_b_before[name])

    # At least one parameter in the written state should have changed.
    assert any(
        not torch.equal(state_a.params[name], state_b_before[name])
        for name in state_a.params
    )

def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        NeuralMemory(key_dim=0, value_dim=2)


def test_invalid_momentum_is_rejected():
    with pytest.raises(ValueError, match="momentum"):
        NeuralMemory(key_dim=3, value_dim=2, momentum=1.0)


def test_invalid_key_shape_is_rejected(module):
    state = module.initial_state()
    invalid_keys = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="keys/queries must have shape"):
        module.read(state, invalid_keys)
