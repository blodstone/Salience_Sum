from typing import Any, Tuple

import torch
from allennlp.common import Registrable
from torch.nn import Module, Sequential, Linear, Tanh, Softmax, ReLU


class Attention(Module, Registrable):
    def __init__(self, hidden_size: int, bidirectional: bool) -> None:
        super().__init__()
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.hidden = hidden_size
        # The query is cat[emb, attn_hidden]
        self._linear_query = Linear(hidden_size * self.num_directions, hidden_size * self.num_directions, bias=True)
        self._linear_source = Linear(hidden_size * self.num_directions, hidden_size * self.num_directions, bias=False)
        self._linear_context = Sequential(
            Linear(hidden_size * self.num_directions + hidden_size * self.num_directions, hidden_size * self.num_directions, bias=True),
            ReLU(),
            Linear(hidden_size * self.num_directions, hidden_size * self.num_directions),
        )
        self._linear_coverage = Linear(1, hidden_size * self.num_directions, bias=False)
        self._v = Linear(hidden_size * self.num_directions, 1, bias=False)
        self._softmax = Softmax(dim=2)

    def score(self, query: torch.Tensor, states: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
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
        dim = query.size(2)
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

        # (B x L_src x 1) -> (B x L_src x 2H)
        coverages_features = self._linear_coverage(coverage)
        # (B x L_src x 1) -> (B x 1 x L_src x 1)
        coverages_features = coverages_features.unsqueeze(1)
        # (B x 1 x L_src x 1) -> (B x L_tgt x L_src x 1)
        coverages_features = coverages_features.expand(batch_size, tgt_length, src_length, dim)
        # (B x L_tgt x L_src x 1)
        alignments = self._v(torch.tanh(query_features + states_features + coverages_features))
        # (B x L_tgt x L_src)
        alignments = alignments.squeeze(3)
        return alignments

    def forward(self,
                query: torch.Tensor,
                states: torch.Tensor,
                source_mask: torch.Tensor,
                coverage: torch.Tensor) \
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
        alignments = self.score(query, states, coverage)
        # Set padding to zero
        alignments = alignments.masked_fill(~source_mask.bool().unsqueeze(1), float('-inf'))
        attentions = self._softmax(alignments)
        # (B, L_tgt, L_src) X (B, L_src, 2*H) = (B, L_tgt, 2*H)
        hidden_context = torch.bmm(attentions, states)
        # (B, L_tgt, H)
        hidden_context = self._linear_context(torch.cat((hidden_context, query), dim=2))
        # attention_hidden = self._context2(attention_hidden)
        # (B, L_src, 1)
        attentions = attentions.transpose(1, 2).contiguous()
        new_coverage = coverage + attentions
        return hidden_context, new_coverage, attentions
