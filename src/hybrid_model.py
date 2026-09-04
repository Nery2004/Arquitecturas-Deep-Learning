"""Arquitectura híbrida C: GRU, operación actual y agregados históricos."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .sequence_model import SequenceGRU


class HybridGRU(nn.Module):
    """Mantiene las ramas history/current de B y añade agregados sin duplicar current."""

    def __init__(self, merchant_vocab_size: int, channel_vocab_size: int, *, aggregate_size: int,
                 aggregate_hidden: int, fusion_hidden: int, dropout: float,
                 merchant_embedding_dim: int = 6, channel_embedding_dim: int = 3,
                 gru_hidden_size: int = 64, current_hidden_size: int = 64) -> None:
        super().__init__()
        self.merchant_embedding = nn.Embedding(merchant_vocab_size, merchant_embedding_dim, padding_idx=0)
        self.channel_embedding = nn.Embedding(channel_vocab_size, channel_embedding_dim, padding_idx=0)
        event_size = 8 + merchant_embedding_dim + channel_embedding_dim
        self.gru = nn.GRU(event_size, gru_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        current_size = 8 + merchant_embedding_dim + channel_embedding_dim
        self.current_projection = nn.Sequential(nn.Linear(current_size, current_hidden_size), nn.ReLU())
        self.aggregate_projection = nn.Sequential(nn.Linear(aggregate_size, aggregate_hidden), nn.ReLU(),
                                                  nn.Dropout(dropout / 2))
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size + current_hidden_size + aggregate_hidden, fusion_hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_hidden, 1))

    def forward(self, sequence_numeric: torch.Tensor, sequence_categorical: torch.Tensor,
                lengths: torch.Tensor, current_numeric: torch.Tensor,
                current_categorical: torch.Tensor, aggregates: torch.Tensor) -> torch.Tensor:
        numeric = SequenceGRU.compact_left_padded(sequence_numeric, lengths)
        categorical = SequenceGRU.compact_left_padded(sequence_categorical, lengths)
        events = torch.cat((numeric, self.merchant_embedding(categorical[:, :, 0]),
                            self.channel_embedding(categorical[:, :, 1])), dim=-1)
        packed = pack_padded_sequence(events, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        current = torch.cat((current_numeric, self.merchant_embedding(current_categorical[:, 0]),
                             self.channel_embedding(current_categorical[:, 1])), dim=-1)
        representations = (hidden[-1], self.current_projection(current), self.aggregate_projection(aggregates))
        return self.classifier(torch.cat(representations, dim=-1)).squeeze(1)
