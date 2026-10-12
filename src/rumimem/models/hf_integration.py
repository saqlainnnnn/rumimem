"""Minimal Hugging Face causal-LM integration for RumiMem."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from rumimem.memory.neural_memory import (
    ProjectedMemoryState,
    ProjectedNeuralMemory,
)


class RumiMemCausalLM(nn.Module):
    """Inject retrieved RumiMem values after a chosen decoder layer.

    This prototype supports one independent memory stream per forward pass
    (batch size 1). Memory writes are explicit and separate from generation.
    The pretrained language model remains frozen.
    """

    def __init__(
        self,
        base_model: nn.Module,
        layer_index: int = 15,
        key_dim: int = 128,
        memory_hidden_dim: int = 128,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()

        if not hasattr(base_model, "model") or not hasattr(
            base_model.model, "layers"
        ):
            raise ValueError(
                "Expected a Hugging Face causal LM with model.layers."
            )

        layers = base_model.model.layers
        if not 0 <= layer_index < len(layers):
            raise ValueError("layer_index is outside the decoder layer range.")

        self.base_model = base_model
        self.layer_index = layer_index
        self.hidden_size = base_model.config.hidden_size

        self.memory = ProjectedNeuralMemory(
            model_dim=self.hidden_size,
            key_dim=key_dim,
            value_dim=self.hidden_size,
            memory_hidden_dim=memory_hidden_dim,
            learning_rate=learning_rate,
        )

        self.fusion = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.normal_(self.fusion.weight, mean=0.0, std=1e-3)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True) -> RumiMemCausalLM:
        """Train memory components while keeping the backbone in eval mode."""
        super().train(mode)
        self.base_model.eval()
        return self

    def initial_memory_state(self) -> ProjectedMemoryState:
        """Create an empty memory state for one conversation stream."""
        return self.memory.initial_state()

    def _capture_hook(self, captured: list[Tensor]):
        def hook(_module: nn.Module, _inputs: tuple, output: Any):
            if not isinstance(output, Tensor):
                raise TypeError(
                    "Expected decoder layer output to be a Tensor."
                )
            captured.append(output.detach())
            return output

        return hook

    @torch.no_grad()
    def encode_for_memory(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Extract layer activations for explicit memory writes.

        Returns [sequence, hidden_size] for a single input stream.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("This prototype requires input_ids shape [1, seq].")

        captured: list[Tensor] = []
        layer = self.base_model.model.layers[self.layer_index]
        handle = layer.register_forward_hook(self._capture_hook(captured))
        try:
            self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("The selected decoder layer was not executed.")

        return captured[0][0]

    def write_memory(
        self,
        state: ProjectedMemoryState,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        chunk_size: int = 16,
    ) -> ProjectedMemoryState:
        """Explicitly write a support passage into the persistent memory."""
        hidden = self.encode_for_memory(input_ids, attention_mask)
        return self.memory.condition(
            state,
            hidden_keys=hidden,
            hidden_values=hidden,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        memory_state: ProjectedMemoryState | None = None,
        use_memory: bool = True,
        **kwargs: Any,
    ):
        """Run the causal LM, optionally injecting retrieved memory."""
        if not use_memory or memory_state is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("This prototype requires input_ids shape [1, seq].")

        def inject(
            _module: nn.Module,
            _inputs: tuple,
            output: Any,
        ):
            if not isinstance(output, Tensor):
                raise TypeError(
                    "Expected decoder layer output to be a Tensor."
                )

            # One memory stream; each token acts as a read query.
            retrieved = self.memory.read(memory_state, output[0])
            residual = self.fusion(retrieved).unsqueeze(0)
            return output + torch.sigmoid(self.gate_logit) * residual

        layer = self.base_model.model.layers[self.layer_index]
        handle = layer.register_forward_hook(inject)
        try:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
        finally:
            handle.remove()
