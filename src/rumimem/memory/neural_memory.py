"""Neural fast-weight memory with gradient-based updates."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.func import functional_call


@dataclass
class NeuralMemoryState:
    """Independent fast-weight state for one memory stream."""

    params: dict[str, Tensor]
    surprise: dict[str, Tensor]
    steps: int = 0


class NeuralMemory(nn.Module):
    """MLP-based associative memory with momentum and forgetting."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        forgetting: float = 0.0,
    ) -> None:
        super().__init__()

        if key_dim <= 0 or value_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if not 0.0 <= forgetting <= 1.0:
            raise ValueError("forgetting must be in [0, 1].")

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.forgetting = forgetting

        self.network = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, value_dim),
        )

        # Start with zero predictions while retaining useful hidden features.
        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def initial_state(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NeuralMemoryState:
        """Create an independent set of fast weights and surprise buffers."""
        reference = next(self.network.parameters())
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype

        params = {
            name: parameter.detach()
            .to(device=device, dtype=dtype)
            .clone()
            .requires_grad_(True)
            for name, parameter in self.network.named_parameters()
        }
        surprise = {name: torch.zeros_like(parameter) for name, parameter in params.items()}

        return NeuralMemoryState(params=params, surprise=surprise)

    def read(self, state: NeuralMemoryState, queries: Tensor) -> Tensor:
        """Retrieve values for queries without modifying the memory state."""
        self._validate_state(state)
        self._validate_keys(queries)

        self._validate_state_input(state, queries)

        return functional_call(self.network, state.params, (queries,))

    def write(
        self,
        state: NeuralMemoryState,
        keys: Tensor,
        values: Tensor,
    ) -> NeuralMemoryState:
        """Update fast weights to reduce associative prediction error."""
        self._validate_state(state)
        self._validate_keys(keys)

        if values.ndim != 2:
            raise ValueError("values must have shape [batch, value_dim].")
        if values.shape != (keys.shape[0], self.value_dim):
            raise ValueError("values must match the key batch and value_dim.")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("keys and values must share device and dtype.")

        self._validate_state_input(state, keys)

        with torch.enable_grad():
            predictions = functional_call(self.network, state.params, (keys,))
            loss = 0.5 * (predictions - values).square().mean()

            names = tuple(state.params)
            gradients = torch.autograd.grad(
                loss,
                tuple(state.params[name] for name in names),
            )

        new_params: dict[str, Tensor] = {}
        new_surprise: dict[str, Tensor] = {}

        for name, gradient in zip(names, gradients, strict=True):
            surprise = (
                self.momentum * state.surprise[name]
                - self.learning_rate * gradient.detach()
            )
            parameter = (
                (1.0 - self.forgetting) * state.params[name] + surprise
            )

            new_surprise[name] = surprise.detach()
            new_params[name] = parameter.detach().requires_grad_(True)

        return NeuralMemoryState(
            params=new_params,
            surprise=new_surprise,
            steps=state.steps + 1,
        )

    def _validate_keys(self, keys: Tensor) -> None:
        if keys.ndim != 2 or keys.shape[1] != self.key_dim:
            raise ValueError("keys/queries must have shape [batch, key_dim].")
        if keys.shape[0] == 0:
            raise ValueError("Batch must not be empty.")
        if not keys.is_floating_point():
            raise ValueError("keys/queries must use a floating-point dtype.")

    def _validate_state_input(
        self,
        state: NeuralMemoryState,
        inputs: Tensor,
    ) -> None:
        reference = next(iter(state.params.values()))
        if inputs.device != reference.device or inputs.dtype != reference.dtype:
            raise ValueError("state and inputs must share device and dtype.")

    def _validate_state(self, state: NeuralMemoryState) -> None:
        expected = dict(self.network.named_parameters())

        if state.steps < 0:
            raise ValueError("steps cannot be negative.")
        if state.params.keys() != expected.keys():
            raise ValueError("state parameters do not match the memory network.")
        if state.surprise.keys() != expected.keys():
            raise ValueError("surprise buffers do not match the memory network.")

        for name, parameter in state.params.items():
            if parameter.shape != expected[name].shape:
                raise ValueError(f"Invalid shape for state parameter {name}.")
            if not parameter.is_floating_point():
                raise ValueError("State parameters must be floating point.")

            surprise = state.surprise[name]
            if surprise.shape != parameter.shape:
                raise ValueError(f"Invalid shape for surprise buffer {name}.")
            if surprise.device != parameter.device or surprise.dtype != parameter.dtype:
                raise ValueError("Parameters and surprise buffers must match device and dtype.")


@dataclass
class ProjectedMemoryState:
    """Fast memory state for projected language-model representations."""

    memory: NeuralMemoryState


class ProjectedNeuralMemory(nn.Module):
    """Project hidden states into keys, values, and queries for neural memory."""

    def __init__(
        self,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        memory_hidden_dim: int = 64,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        forgetting: float = 0.0,
    ) -> None:
        super().__init__()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")

        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.key_proj = nn.Linear(model_dim, key_dim)
        self.value_proj = nn.Linear(model_dim, value_dim)
        self.query_proj = nn.Linear(model_dim, key_dim)

        # Start reads and writes with compatible addressing projections.
        with torch.no_grad():
            self.query_proj.weight.copy_(self.key_proj.weight)
            self.query_proj.bias.copy_(self.key_proj.bias)

        self.memory = NeuralMemory(
            key_dim=key_dim,
            value_dim=value_dim,
            hidden_dim=memory_hidden_dim,
            learning_rate=learning_rate,
            momentum=momentum,
            forgetting=forgetting,
        )

    def initial_state(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ProjectedMemoryState:
        """Create a fresh fast-memory state for one stream."""
        return ProjectedMemoryState(memory=self.memory.initial_state(device, dtype))

    def read(
        self,
        state: ProjectedMemoryState,
        hidden_queries: Tensor,
    ) -> Tensor:
        """Project hidden states to queries and retrieve memory values."""
        self._validate_hidden(hidden_queries, "hidden_queries")
        query = self.query_proj(hidden_queries)
        return self.memory.read(state.memory, query)

    def write(
        self,
        state: ProjectedMemoryState,
        hidden_keys: Tensor,
        hidden_values: Tensor,
    ) -> ProjectedMemoryState:
        """Project hidden states and update the stream's fast memory."""
        self._validate_hidden(hidden_keys, "hidden_keys")
        self._validate_hidden(hidden_values, "hidden_values")

        if hidden_keys.shape[0] != hidden_values.shape[0]:
            raise ValueError("hidden_keys and hidden_values batch sizes must match.")
        if hidden_keys.device != hidden_values.device:
            raise ValueError("hidden_keys and hidden_values must share a device.")
        if hidden_keys.dtype != hidden_values.dtype:
            raise ValueError("hidden_keys and hidden_values must share a dtype.")

        keys = self.key_proj(hidden_keys)
        values = self.value_proj(hidden_values)

        updated_memory = self.memory.write(state.memory, keys, values)
        return ProjectedMemoryState(memory=updated_memory)

    def condition(
        self,
        state: ProjectedMemoryState,
        hidden_keys: Tensor,
        hidden_values: Tensor,
        chunk_size: int = 16,
    ) -> ProjectedMemoryState:
        """Update memory sequentially from chunks of a hidden-state sequence."""
        self._validate_hidden(hidden_keys, "hidden_keys")
        self._validate_hidden(hidden_values, "hidden_values")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if hidden_keys.shape[0] != hidden_values.shape[0]:
            raise ValueError("hidden_keys and hidden_values sequence lengths must match.")
        if hidden_keys.device != hidden_values.device:
            raise ValueError("hidden_keys and hidden_values must share a device.")
        if hidden_keys.dtype != hidden_values.dtype:
            raise ValueError("hidden_keys and hidden_values must share a dtype.")

        for start in range(0, hidden_keys.shape[0], chunk_size):
            end = start + chunk_size
            state = self.write(
                state,
                hidden_keys[start:end],
                hidden_values[start:end],
            )

        return state

    def _validate_hidden(self, hidden: Tensor, name: str) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != self.model_dim:
            raise ValueError(
                f"{name} must have shape [batch, model_dim]."
            )
        if hidden.shape[0] == 0:
            raise ValueError(f"{name} batch must not be empty.")
        if not hidden.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype.")

        reference = self.key_proj.weight
        if hidden.device != reference.device or hidden.dtype != reference.dtype:
            raise ValueError("hidden states must match projection device and dtype.")


class GatedNeuralMemory(NeuralMemory):
    """Neural memory with input-dependent learning, momentum, and decay gates."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(
            key_dim=key_dim,
            value_dim=value_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            momentum=0.0,
            forgetting=0.0,
        )

        # Outputs control decay, momentum, and write strength, respectively.
        self.update_controller = nn.Linear(key_dim, 3)

        with torch.no_grad():
            nn.init.normal_(self.update_controller.weight, mean=0.0, std=0.01)
            self.update_controller.bias[0] = math.log(0.02 / 0.98)
            self.update_controller.bias[1] = math.log(0.9 / 0.1)
            self.update_controller.bias[2] = math.log(
                math.expm1(learning_rate)
            )

    def write(
        self,
        state: NeuralMemoryState,
        keys: Tensor,
        values: Tensor,
        *,
        differentiable: bool = False,
    ) -> NeuralMemoryState:
        """Write each association sequentially using input-dependent gates.

        Set differentiable=True when training an outer objective through
        the memory updates. This retains computation graphs and uses more RAM.
        """
        self._validate_state(state)
        self._validate_keys(keys)

        if values.ndim != 2 or values.shape != (
            keys.shape[0], self.value_dim
        ):
            raise ValueError("values must match the key batch and value_dim.")
        if values.device != keys.device or values.dtype != keys.dtype:
            raise ValueError("keys and values must share device and dtype.")

        self._validate_state_input(state, keys)

        params = state.params
        surprise_state = state.surprise

        # A sequence/batch is processed as ordered individual writes.
        for index in range(keys.shape[0]):
            key = keys[index:index + 1]
            value = values[index:index + 1]

            with torch.enable_grad():
                prediction = functional_call(self.network, params, (key,))
                loss = 0.5 * (prediction - value).square().mean()

                names = tuple(params)
                gradients = torch.autograd.grad(
                    loss,
                    tuple(params[name] for name in names),
                    create_graph=differentiable,
                )

            decay_logit, momentum_logit, strength_logit = (
                self.update_controller(key).squeeze(0)
            )
            decay = torch.sigmoid(decay_logit)
            momentum = torch.sigmoid(momentum_logit)
            strength = torch.nn.functional.softplus(strength_logit)

            next_params = {}
            next_surprise = {}

            for name, gradient in zip(names, gradients, strict=True):
                updated_surprise = (
                    momentum * surprise_state[name]
                    - strength * gradient
                )
                updated_param = (
                    (1.0 - decay) * params[name] + updated_surprise
                )

                if not differentiable:
                    updated_surprise = updated_surprise.detach()
                    updated_param = updated_param.detach().requires_grad_(True)

                next_surprise[name] = updated_surprise
                next_params[name] = updated_param

            params = next_params
            surprise_state = next_surprise

        return NeuralMemoryState(
            params=params,
            surprise=surprise_state,
            steps=state.steps + 1,
        )
