import pytest
import torch

from rumimem.models.mac import MACBlock


def test_mac_processes_multiple_chunks():
    torch.manual_seed(42)
    model = MACBlock(model_dim=32, num_heads=4, memory_key_dim=16)
    state = model.initial_state()

    out1, state = model(torch.randn(1, 8, 32), state)
    out2, state = model(torch.randn(1, 5, 32), state)

    assert out1.shape == (1, 8, 32)
    assert out2.shape == (1, 5, 32)
    assert state.steps == 2
    assert state.memory.memory.steps == 2
    assert state.context.shape == (1, 13, 32)


def test_mac_supports_independent_batch_streams():
    torch.manual_seed(42)
    model = MACBlock(model_dim=16, num_heads=4, memory_key_dim=8)
    state = model.initial_state(batch_size=2)

    _, updated = model(torch.randn(2, 4, 16), state)

    assert len(updated.memories) == 2
    assert updated.memories[0] is not updated.memories[1]
    assert updated.memories[0].memory.params is not updated.memories[1].memory.params
    assert updated.context.shape == (2, 4, 16)


def test_stream_memories_are_independent():
    torch.manual_seed(42)
    model = MACBlock(model_dim=16, num_heads=4, memory_key_dim=8)
    state = model.initial_state(batch_size=2)

    stream1_before = {
        name: value.clone()
        for name, value in state.memories[1].memory.params.items()
    }

    # Update stream 0 only, using a distinct sequence.
    updated_stream0 = model.long_term_memory.condition(
        state.memories[0],
        torch.randn(4, 16),
        torch.randn(4, 16),
        chunk_size=2,
    )

    # Stream 1's state must remain untouched.
    for name, value in state.memories[1].memory.params.items():
        assert torch.equal(value, stream1_before[name])

    # Updating stream 0 must produce a new state without mutating its input.
    assert updated_stream0 is not state.memories[0]
    assert updated_stream0.memory.steps == 2
    assert state.memories[0].memory.steps == 0

def test_context_cache_is_bounded():
    model = MACBlock(
        model_dim=16,
        num_heads=4,
        memory_key_dim=8,
        context_cache_size=6,
    )
    state = model.initial_state()

    _, state = model(torch.randn(1, 4, 16), state)
    assert state.context.shape[1] == 4

    _, state = model(torch.randn(1, 4, 16), state)
    assert state.context.shape == (1, 6, 16)


def test_future_tokens_do_not_change_earlier_outputs():
    torch.manual_seed(42)
    model = MACBlock(model_dim=16, num_heads=4, memory_key_dim=8).eval()
    state = model.initial_state()

    prefix = torch.randn(1, 3, 16)
    first = torch.cat([prefix, torch.randn(1, 1, 16)], dim=1)
    second = torch.cat([prefix, torch.randn(1, 1, 16) * 10], dim=1)

    with torch.no_grad():
        out1, _ = model(first, state, update_memory=False)
        out2, _ = model(second, state, update_memory=False)

    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)


def test_mac_can_skip_memory_write():
    model = MACBlock(model_dim=16, num_heads=4, memory_key_dim=8)
    state = model.initial_state()

    _, updated = model(
        torch.randn(1, 4, 16), state, update_memory=False
    )

    assert updated.steps == 1
    assert updated.memory.memory.steps == 0


def test_mac_rejects_empty_sequence():
    model = MACBlock(model_dim=16, num_heads=4, memory_key_dim=8)
    state = model.initial_state()

    with pytest.raises(ValueError, match="sequence length"):
        model(torch.randn(1, 0, 16), state)


def test_mac_rejects_mismatched_batch_size():
    model = MACBlock(model_dim=16, num_heads=4, memory_key_dim=8)
    state = model.initial_state(batch_size=1)

    with pytest.raises(ValueError, match="batch size"):
        model(torch.randn(2, 4, 16), state)
