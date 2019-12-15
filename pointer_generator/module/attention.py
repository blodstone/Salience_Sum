from typing import Any, Tuple

import torch
from allennlp.common import Registrable
from torch.nn import Module, Sequential, Linear, Tanh, Softmax


class Attention(Module, Registrable):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden = hidden_size
        self._alignment = Sequential(
            Linear(hidden_size * 4 + 1, hidden_size * 2, bias=True),
            Tanh(),
            Linear(hidden_size * 2, 1, bias=False),
            Softmax(dim=1)
        )
        self._reduce = Linear(hidden_size * 2, hidden_size, bias=False)

    def score(self, query: torch.Tensor, keys: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        length = keys.size(1)
        dim_size = query.size(2)
        expanded_query = query.expand(batch_size, length, dim_size)
        alignments = self._alignment(torch.cat((expanded_query, keys, coverage), dim=2))
        return alignments

    def forward(self, query: torch.Tensor, keys: torch.Tensor, coverage: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculating the attention using Bahdanau approach

        :param query: The state of target (B, 1, H)
        :param keys: The memory states of encoder (B, L, 2*H) in bidirectional format
        :return: The weighted context (B, 1, H)
        """
        # (B, L, 1)
        attention = self.score(query, keys, coverage)
        assert attention.size(2) == 1
        # (B, L, H)
        weighted_keys = attention.transpose(1, 2).matmul(self._reduce(keys))
        # (B, 1, H)
        context = torch.sum(weighted_keys, dim=1).unsqueeze(1)
        return context, attention
