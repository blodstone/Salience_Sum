from typing import Tuple

import torch
from allennlp.common import Registrable
from torch.nn import Module, Linear, Softmax


class Attention(Module, Registrable):
    def __init__(self, hidden_size: int, bidirectional: bool) -> None:
        super().__init__()
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.hidden = hidden_size
        # The query is cat[emb, attn_hidden]
        self._linear_query = Linear(2 * hidden_size, 2 * hidden_size)
        self._linear_coverage = Linear(1, 2 * hidden_size, bias=False)
        self._v = Linear(2 * hidden_size, 1, bias=False)
        self._softmax = Softmax(dim=1)

    def score(self, query: torch.Tensor,
              states: torch.Tensor,
              states_features: torch.Tensor,
              coverage: torch.Tensor,
              is_coverage: bool,
              emb_salience_feature: torch.Tensor,
              is_emb_attention: bool,
              emb_attention_mode: str
              ) -> torch.Tensor:
        """
        Bahdanau attention
        :param emb_attention_mode:
        :param emb_salience_feature:
        :param is_emb_attention:
        :param is_coverage:
        :param states_features:
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
        query_features = self._linear_query(query)
        # (B x 1 x 2H) -> (B x L_src x 2H)
        query_features = query_features.expand(batch_size, src_length, dim)

        if is_coverage:
            # (B x L_src x 2H) -> (B x L_src x 2H)
            coverages_features = self._linear_coverage(coverage)
            total_features = query_features + states_features + coverages_features
        else:
            if is_emb_attention:
                total_features = query_features + states_features + emb_salience_feature
            else:
                total_features = query_features + states_features
        # (B x L_src x 2H)
        alignments = self._v(torch.tanh(total_features))
        # (B x L_src x 1)
        return alignments

    def forward(self,
                query: torch.Tensor,
                states: torch.Tensor,
                states_features: torch.Tensor,
                source_mask: torch.Tensor,
                coverage: torch.Tensor,
                is_coverage: bool,
                emb_salience_feature: torch.Tensor = None,
                is_emb_attention: bool = False,
                emb_attention_mode: str = 'mlp',
                ) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculating the attention using Bahdanau approach

        :param emb_attention_mode: The mode of embedding (mlp or bilinear)
        :param is_emb_attention: To use salience embedding in attention or not
        :param emb_salience_feature: The embedding of salience value ()
        :param is_coverage: To use coverage or not
        :param states_features: Precalculated states features
        :param coverage: The previous coverage (B, L_src, 1)
        :param source_mask: Mask for source (B, L_src, 1)
        :param query: The state of target (B, L_tgt, H)
        :param states: The memory states of encoder (B, L_src, 2*H)
        :return: The weighted context (B, 1, H)
        """
        # (B, L_tgt, L_src)
        alignments = self.score(query, states, states_features,
                                coverage, is_coverage,
                                emb_salience_feature, is_emb_attention, emb_attention_mode
                                )
        # Set padding to zero
        alignments = alignments.masked_fill(~source_mask.bool().unsqueeze(2), float('-inf'))
        align_vectors = self._softmax(alignments)
        # (B, L_tgt, L_src) X (B, L_src, 2*H) = (B, L_tgt, 2*H)
        attn_h = torch.bmm(align_vectors.transpose(1, 2), states)
        # (B, L_src, 1)
        # attentions = attentions.transpose(1, 2).contiguous()
        new_coverage = coverage + align_vectors
        return attn_h, new_coverage, align_vectors
