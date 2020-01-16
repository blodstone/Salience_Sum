from typing import Dict, Tuple, List, Any

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from torch.nn import LSTM, Module, Sequential, Linear, Tanh, Sigmoid, Softmax

from salience_sum.module.copy_attention import CopyAttention


class Decoder(Module, Registrable):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 attention: CopyAttention,
                 num_layers: int,
                 training: bool = True) -> None:
        super().__init__()
        self.vocab = None
        self.training = training
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self._attention = attention
        self._rnn = LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         batch_first=True)
        self._p_gen = Sequential(
            Linear(2 * self.hidden_size + self.hidden_size + self.input_size, 1, bias=True),
            Sigmoid()
        )
        self._gen_vocab_dist = None

    def add_vocab(self, vocab: Vocabulary):
        self.vocab = vocab
        self._gen_vocab_dist = Sequential(
            Linear(self.hidden_size, self.vocab.get_vocab_size()),
            Softmax(dim=2)
        )

    def get_output_dim(self) -> int:
        return self.hidden_size

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

    def forward(self, source_tokens: Dict[str, torch.Tensor],
                embedded_tgt: torch.Tensor,
                state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        states = state['encoder_states']
        source_mask = state['source_mask']
        # Initial decoder state
        final_state = (
            state['encoder_final_state'].transpose(0, 1).contiguous(),
            state['encoder_context'].transpose(0, 1).contiguous()
        )
        dec_states = []
        p_gens = []
        attentions = []
        class_log_probs = []
        coverages = []
        # (B, L_src, 1)
        coverage = states.new_zeros((states.size(0), states.size(1), 1))
        for step, emb in enumerate(embedded_tgt.split(1, 1)):
            dec_state, final_state = self._rnn(
                emb,
                final_state
            )

            context, attention_hidden, coverage, attention = self._attention(
                dec_state, states, source_mask, coverage)

            # (B x 1 x 1)
            p_gen = self._p_gen(torch.cat((context, dec_state, emb), dim=2))
            # (B x Vocab)
            vocab_dist = (p_gen * self._gen_vocab_dist(attention_hidden)).squeeze(1)
            # (B x L_src)
            attn_dist = ((1 - p_gen) * attention).squeeze(2)
            # (B x 1 x Vocab)
            final_dist = vocab_dist.scatter_add(1, source_tokens['tokens'], attn_dist).unsqueeze(1)
            class_log_prob = (final_dist + 1e-20).log()
            class_log_probs.append(class_log_prob)

            dec_state = attention_hidden
            dec_states.append(dec_state)

            attentions.append(attention)
            coverages.append(coverage)
            p_gens.append(p_gen)
        state['decoder_states'] = torch.stack(dec_states, dim=1).squeeze(2)
        state['class_log_probs'] = torch.stack(class_log_probs, dim=1).squeeze(2)
        meta_state = {
            'p_gens': torch.stack(p_gens, dim=1).squeeze(2),
            'attentions': torch.stack(attentions, dim=1).squeeze(2),
            'coverages': torch.stack(coverages, dim=1).squeeze(2),
        }
        return state, meta_state

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass
