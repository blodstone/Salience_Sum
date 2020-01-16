from typing import Any, Tuple

import torch
from allennlp.common import Registrable
from torch.nn import Module, Sequential, Linear, Tanh, Softmax


class Attention(Module, Registrable):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden = hidden_size
        self._linear_query = Linear(hidden_size, hidden_size * 2, bias=True)
        self._linear_source = Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self._v = Linear(hidden_size * 2, 1, bias=False)
        self._softmax = Softmax(dim=2)
        self._context = Linear(hidden_size * 3, hidden_size, bias=True)

    def score(self, query: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Bahdanau attention
        :param coverage: The coverage of all the previous steps (dim: H)
        :param query: The decoder states (dim: H)
        :param states: The encoder states (dim: 2H)
        :return: The alignment between query and source un-normalized
        """
        batch_size = states.size(0)
        tgt_length = query.size(1)
        src_length = states.size(1)
        dim = states.size(2)
        # The formula: v.tanh(Wh*hi + Ws*st + b)
        # (B x L_tgt x H) -> (B x L_tgt x 2H)
        query_features = self._linear_query(query)
        # (B x L_tgt x 2H) -> (B x L_tgt x 1 x 2H)
        query_features = query_features.unsqueeze(2)
        # (B x L_tgt x 1 x 2H) -> (B x L_tgt x L_src x 2H)
        query_features = query_features.expand(batch_size, tgt_length, src_length, dim)

        # (B x L_src x 2H) -> (B x L_src x 2H)
        states_features = self._linear_source(states)
        # (B x L_src x 2H) -> (B x 1 x L_src x 2H)
        states_features = states_features.unsqueeze(1)
        # (B x 1 x L_src x 2H) -> (B x L_tgt x L_src x 2H)
        states_features = states_features.expand(batch_size, tgt_length, src_length, dim)

        # (B x L_tgt x L_src x 1)
        alignments = self._v(torch.tanh(query_features + states_features))
        # (B x L_tgt x L_src)
        alignments = alignments.squeeze(3)
        return alignments

    def forward(self,
                query: torch.Tensor,
                states: torch.Tensor,
                source_mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculating the attention using Bahdanau approach

        :param coverage: The previous coverage (B, L_src, 1)
        :param source_mask: Mask for source (B, L_src, 1)
        :param query: The state of target (B, L_tgt, H)
        :param states: The memory states of encoder (B, L_src, 2*H)
        :return: The weighted context (B, 1, H)
        """
        # (B, L_tgt, L_src)
        alignments = self.score(query, states)
        # Set padding to zero
        alignments = alignments.masked_fill(~source_mask.bool().unsqueeze(1), float('-inf'))
        attentions = self._softmax(alignments)
        # (B, L_tgt, L_src) X (B, L_src, 2*H) = (B, L_tgt, 2*H)
        context = torch.bmm(attentions, states)
        concat_context = torch.cat((context, query), dim=2)
        # (B, L_tgt, H)
        attention_hidden = self._context(concat_context)
        # (B, L_src, 1)
        attentions = attentions.transpose(1, 2).contiguous()
        return context, attention_hidden, attentions
