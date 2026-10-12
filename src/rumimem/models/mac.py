"""MAC: short-term attention plus persistent neural memory.

Research prototype inspired by memory-augmented architectures.
This is not a full reproduction of the Titans architecture.
Each batch item maintains an independent long-term memory state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rumimem.memory.neural_memory import (
    ProjectedMemoryState,
    ProjectedNeuralMemory,
)


@dataclass
class MACState:
    """State for a batch of independent streams."""

    memory: ProjectedMemoryState
    memories: list[ProjectedMemoryState]
    persistent_tokens: Tensor
    context: Tensor
    steps: int = 0


class MACBlock(nn.Module):
    """Fuse causal short-term context and long-term neural memory."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 4,
        memory_key_dim: int = 64,
        memory_value_dim: int | None = None,
        memory_hidden_dim: int = 64,
        num_persistent_tokens: int = 4,
        memory_learning_rate: float = 0.01,
        context_cache_size: int = 128,
    ) -> None:
        super().__init__()

        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_persistent_tokens <= 0:
            raise ValueError("num_persistent_tokens must be positive")
        if context_cache_size <= 0:
            raise ValueError("context_cache_size must be positive")

        self.model_dim = model_dim
        self.num_persistent_tokens = num_persistent_tokens
        self.context_cache_size = context_cache_size

        self.persistent_tokens = nn.Parameter(
            torch.randn(1, num_persistent_tokens, model_dim)
            * model_dim**-0.5
        )

        self.short_term_norm = nn.LayerNorm(model_dim)
        self.short_term_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            batch_first=True,
        )

        value_dim = (
            memory_value_dim
            if memory_value_dim is not None
            else model_dim
        )

        self.long_term_memory = ProjectedNeuralMemory(
            model_dim=model_dim,
            key_dim=memory_key_dim,
            value_dim=value_dim,
            memory_hidden_dim=memory_hidden_dim,
            learning_rate=memory_learning_rate,
        )

        self.memory_output = nn.Linear(value_dim, model_dim)
        self.fusion_gate = nn.Linear(model_dim * 3, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )

    def initial_state(self, batch_size: int = 1) -> MACState:
        """Create independent memory states for each batch stream."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        memories = [
            self.long_term_memory.initial_state()
            for _ in range(batch_size)
        ]

        persistent = self.persistent_tokens.expand(
            batch_size, -1, -1
        ).clone()

        context = self.persistent_tokens.new_empty(
            batch_size, 0, self.model_dim
        )

        return MACState(
            memory=memories[0],
            memories=memories,
            persistent_tokens=persistent,
            context=context,
        )

    def forward(
        self,
        hidden: Tensor,
        state: MACState,
        *,
        update_memory: bool = True,
        memory_chunk_size: int = 16,
    ) -> tuple[Tensor, MACState]:
        """Process a chunk shaped [batch, sequence, model_dim]."""

        if hidden.ndim != 3 or hidden.shape[-1] != self.model_dim:
            raise ValueError(
                "hidden must have shape [batch, sequence, model_dim]"
            )
        if hidden.shape[0] != len(state.memories):
            raise ValueError("batch size does not match MAC state")
        if hidden.shape[0] != state.persistent_tokens.shape[0]:
            raise ValueError(
                "batch size does not match persistent tokens"
            )
        if hidden.shape[1] == 0:
            raise ValueError("sequence length must be positive")
        if memory_chunk_size <= 0:
            raise ValueError("memory_chunk_size must be positive")
        if hidden.device != self.persistent_tokens.device:
            raise ValueError("hidden and model must share a device")
        if hidden.dtype != self.persistent_tokens.dtype:
            raise ValueError("hidden and model must share a dtype")
        if state.context.device != hidden.device:
            raise ValueError("context state must match hidden device")
        if state.context.dtype != hidden.dtype:
            raise ValueError("context state must match hidden dtype")
        if state.context.shape[0] != hidden.shape[0]:
            raise ValueError("context batch size does not match hidden")

        batch_size, seq_len, _ = hidden.shape
        cached_len = state.context.shape[1]

        # The cache contains only prior tokens. Current tokens are appended
        # for attention but will be added to the cache only after processing.
        context = torch.cat([state.context, hidden], dim=1)
        key_values = torch.cat(
            [state.persistent_tokens, context],
            dim=1,
        )

        query_norm = self.short_term_norm(hidden)
        key_values_norm = self.short_term_norm(key_values)

        # Keys are ordered as:
        # persistent tokens | cached tokens | current chunk.
        #
        # Current token i can attend to all persistent and cached tokens,
        # plus current-chunk positions <= i. Persistent tokens are keys only
        # in this attention operation, so they cannot read future tokens.
        key_len = key_values.shape[1]
        mask = torch.zeros(
            seq_len,
            key_len,
            dtype=torch.bool,
            device=hidden.device,
        )

        current_start = self.num_persistent_tokens + cached_len
        mask[:, current_start:] = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=hidden.device,
            ),
            diagonal=1,
        )

        short_term, _ = self.short_term_attention(
            query_norm,
            key_values_norm,
            key_values_norm,
            attn_mask=mask,
            need_weights=False,
        )

        # Retrieve independently from each stream's long-term memory.
        long_term_outputs = []

        for index in range(batch_size):
            queries = self.long_term_memory.query_proj(hidden[index])

            retrieved = self.long_term_memory.memory.read(
                state.memories[index].memory,
                queries,
            )

            long_term_outputs.append(self.memory_output(retrieved))

        long_term = torch.stack(long_term_outputs, dim=0)

        # Fuse current hidden states, short-term context, and long-term recall.
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat(
                    [hidden, short_term, long_term],
                    dim=-1,
                )
            )
        )

        fused = gate * short_term + (1.0 - gate) * long_term
        output = self.output_norm(hidden + fused)
        output = output + self.feed_forward(output)

        # Write only new tokens. Each stream has its own fast-weight state.
        # condition() accepts [sequence, model_dim] and projects internally.
        new_memories = state.memories

        if update_memory:
            new_memories = [
                self.long_term_memory.condition(
                    state.memories[index],
                    hidden[index],
                    hidden[index],
                    chunk_size=memory_chunk_size,
                )
                for index in range(batch_size)
            ]

        # Bound context length and avoid retaining the entire prior autograd
        # graph across inference chunks.
        new_context = context[:, -self.context_cache_size :].detach()

        new_state = MACState(
            memory=new_memories[0],
            memories=new_memories,
            persistent_tokens=state.persistent_tokens,
            context=new_context,
            steps=state.steps + 1,
        )

        return output, new_state
