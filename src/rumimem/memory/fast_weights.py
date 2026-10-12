"""Linear fast-weight associative memory."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn


class LinearFastWeightMemory(nn.Module):
    """Associative memory updated through gradient-based writes."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()

        if key_dim <= 0 or value_dim <= 0:
            raise ValueError("key_dim and value_dim must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.learning_rate = learning_rate

    def initial_state(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Create a fresh memory matrix for one independent stream."""
        return torch.zeros(
            self.value_dim,
            self.key_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

    def read(self, memory: Tensor, keys: Tensor) -> Tensor:
        """Retrieve values associated with a batch of keys."""
        self._validate_keys(keys)
        self._validate_memory(memory)

        if keys.device != memory.device or keys.dtype != memory.dtype:
            raise ValueError("keys and memory must share device and dtype.")

        return keys @ memory.T

    def write(
        self,
        memory: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Update memory to reduce key-value prediction error."""
        self._validate_keys(keys)
        self._validate_memory(memory)

        if values.ndim != 2:
            raise ValueError("values must have shape [batch, value_dim].")
        if values.shape != (keys.shape[0], self.value_dim):
            raise ValueError("values must match the key batch and value_dim.")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("keys and values must share device and dtype.")
        if memory.device != keys.device or memory.dtype != keys.dtype:
            raise ValueError("memory, keys, and values must share device and dtype.")

        state = memory if memory.requires_grad else memory.detach().requires_grad_(True)

        with torch.enable_grad():
            predictions = keys @ state.T
            loss = 0.5 * (predictions - values).square().mean()

            (gradient,) = torch.autograd.grad(
                loss,
                state,
                create_graph=create_graph,
            )

            updated_memory = state - self.learning_rate * gradient

        if not create_graph:
            updated_memory = updated_memory.detach()

        return updated_memory

    def _validate_keys(self, keys: Tensor) -> None:
        if keys.ndim != 2 or keys.shape[1] != self.key_dim:
            raise ValueError("keys must have shape [batch, key_dim].")
        if not keys.is_floating_point():
            raise ValueError("keys must use a floating-point dtype.")

    def _validate_memory(self, memory: Tensor) -> None:
        if memory.shape != (self.value_dim, self.key_dim):
            raise ValueError("memory must have shape [value_dim, key_dim].")
        if not memory.is_floating_point():
            raise ValueError("memory must use a floating-point dtype.")


@dataclass
class LinearMemoryState:
    """Per-stream state for a momentum-based linear memory."""

    memory: Tensor
    surprise: Tensor
    steps: int = 0


class MomentumLinearFastWeightMemory(nn.Module):
    """Linear fast-weight memory with momentum and explicit forgetting."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        forgetting: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if not 0.0 <= forgetting <= 1.0:
            raise ValueError("forgetting must be in [0, 1].")

        self.base = LinearFastWeightMemory(
            key_dim=key_dim,
            value_dim=value_dim,
            learning_rate=learning_rate,
        )
        self.momentum = momentum
        self.forgetting = forgetting

    def initial_state(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> LinearMemoryState:
        """Create independent memory and surprise tensors for one stream."""
        memory = self.base.initial_state(device=device, dtype=dtype).detach()
        surprise = torch.zeros_like(memory)

        return LinearMemoryState(memory=memory, surprise=surprise)

    def read(self, state: LinearMemoryState, keys: Tensor) -> Tensor:
        """Read from the current memory without modifying its state."""
        self._validate_state(state)
        return self.base.read(state.memory, keys)

    def write(
        self,
        state: LinearMemoryState,
        keys: Tensor,
        values: Tensor,
    ) -> LinearMemoryState:
        """Apply one gradient-based write with momentum and forgetting."""
        self._validate_state(state)
        self.base._validate_keys(keys)

        if values.ndim != 2:
            raise ValueError("values must have shape [batch, value_dim].")
        if values.shape != (keys.shape[0], self.base.value_dim):
            raise ValueError("values must match the key batch and value_dim.")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("keys and values must share device and dtype.")
        if state.memory.device != keys.device or state.memory.dtype != keys.dtype:
            raise ValueError("memory, keys, and values must share device and dtype.")

        # Compute the gradient against the current memory.
        with torch.enable_grad():
            memory = state.memory.detach().requires_grad_(True)
            predictions = self.base.read(memory, keys)
            loss = 0.5 * (predictions - values).square().mean()
            (gradient,) = torch.autograd.grad(loss, memory)

        # Accumulate the surprise/update signal, then decay old memory.
        surprise = (
            self.momentum * state.surprise
            - self.base.learning_rate * gradient.detach()
        )
        updated_memory = (1.0 - self.forgetting) * state.memory + surprise

        return LinearMemoryState(
            memory=updated_memory.detach(),
            surprise=surprise.detach(),
            steps=state.steps + 1,
        )

    def _validate_state(self, state: LinearMemoryState) -> None:
        self.base._validate_memory(state.memory)

        if state.surprise.shape != state.memory.shape:
            raise ValueError("surprise must have the same shape as memory.")
        if (
            state.surprise.device != state.memory.device
            or state.surprise.dtype != state.memory.dtype
        ):
            raise ValueError("surprise and memory must share device and dtype.")
        if state.steps < 0:
            raise ValueError("steps cannot be negative.")
