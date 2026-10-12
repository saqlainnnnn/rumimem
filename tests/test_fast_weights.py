import pytest
import torch

from rumimem.memory.fast_weights import LinearFastWeightMemory


@pytest.fixture
def memory_module():
    return LinearFastWeightMemory(
        key_dim=3,
        value_dim=2,
        learning_rate=0.1,
    )


def test_initial_memory_reads_zeros(memory_module):
    memory = memory_module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])

    prediction = memory_module.read(memory, keys)

    assert prediction.shape == (1, 2)
    assert torch.equal(prediction, torch.zeros(1, 2))


def test_write_improves_retrieval(memory_module):
    memory = memory_module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[2.0, 3.0]])

    initial_error = (memory_module.read(memory, keys) - values).square().mean()

    for _ in range(100):
        memory = memory_module.write(memory, keys, values)

    final_error = (memory_module.read(memory, keys) - values).square().mean()

    assert final_error < initial_error
    assert final_error.item() < 0.01


def test_write_returns_updated_memory_without_mutating_original(memory_module):
    memory = memory_module.initial_state()
    original_memory = memory.detach().clone()

    keys = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[2.0, 3.0]])

    updated_memory = memory_module.write(memory, keys, values)

    assert torch.equal(memory.detach(), original_memory)
    assert not torch.equal(updated_memory, original_memory)


def test_read_rejects_invalid_key_dimensions(memory_module):
    memory = memory_module.initial_state()
    invalid_keys = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="keys must have shape"):
        memory_module.read(memory, invalid_keys)


def test_write_rejects_mismatched_value_dimensions(memory_module):
    memory = memory_module.initial_state()
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    invalid_values = torch.tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="values must match"):
        memory_module.write(memory, keys, invalid_values)


def test_joint_writes_learn_multiple_associations(memory_module):
    memory = memory_module.initial_state()
    keys = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
    ])
    values = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    initial_error = (memory_module.read(memory, keys) - values).square().mean()

    for _ in range(500):
        memory = memory_module.write(memory, keys, values)

    predictions = memory_module.read(memory, keys)
    final_error = (predictions - values).square().mean()

    assert final_error < initial_error
    assert final_error.item() < 0.01
