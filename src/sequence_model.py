"""Arquitectura GRU pequeña para Modelo B."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceGRU(nn.Module):
    """Combina historia GRU y operación actual, sin agregados de Modelo A."""

    def __init__(self, merchant_vocab_size: int, channel_vocab_size: int, *, hidden_size: int,
                 num_layers: int, dense_size: int, dropout: float,
                 merchant_embedding_dim: int = 6, channel_embedding_dim: int = 3) -> None:
        super().__init__()
        self.merchant_embedding = nn.Embedding(merchant_vocab_size, merchant_embedding_dim, padding_idx=0)
        self.channel_embedding = nn.Embedding(channel_vocab_size, channel_embedding_dim, padding_idx=0)
        event_size = 8 + merchant_embedding_dim + channel_embedding_dim
        self.gru = nn.GRU(event_size, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=False)
        current_size = 8 + merchant_embedding_dim + channel_embedding_dim
        self.current_projection = nn.Sequential(nn.Linear(current_size, dense_size), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(hidden_size + dense_size, dense_size), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(dense_size, 1))

    @staticmethod
    def compact_left_padded(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mueve eventos válidos al inicio para que packing ignore el left padding."""
        batch, width = values.shape[:2]
        positions = torch.arange(width, device=values.device).expand(batch, width)
        source = positions + (width - lengths).unsqueeze(1)
        source = source.clamp(max=width - 1)
        gather_shape = [batch, width] + [1] * (values.ndim - 2)
        gathered = values.gather(1, source.view(*gather_shape).expand_as(values))
        valid = positions < lengths.unsqueeze(1)
        while valid.ndim < gathered.ndim:
            valid = valid.unsqueeze(-1)
        return torch.where(valid, gathered, torch.zeros_like(gathered))

    def forward(self, sequence_numeric: torch.Tensor, sequence_categorical: torch.Tensor,
                lengths: torch.Tensor, current_numeric: torch.Tensor,
                current_categorical: torch.Tensor) -> torch.Tensor:
        numeric = self.compact_left_padded(sequence_numeric, lengths)
        categorical = self.compact_left_padded(sequence_categorical, lengths)
        events = torch.cat((numeric, self.merchant_embedding(categorical[:, :, 0]),
                            self.channel_embedding(categorical[:, :, 1])), dim=-1)
        packed = pack_padded_sequence(events, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        # hidden[-1] corresponde al último evento válido; packing excluye PAD.
        history_representation = hidden[-1]
        current = torch.cat((current_numeric, self.merchant_embedding(current_categorical[:, 0]),
                             self.channel_embedding(current_categorical[:, 1])), dim=-1)
        current_representation = self.current_projection(current)
        return self.classifier(torch.cat((history_representation, current_representation), dim=-1)).squeeze(1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
