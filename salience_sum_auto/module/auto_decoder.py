from typing import Dict, Tuple, List, Any

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from torch.nn import LSTM, Module, Sequential, Linear, Tanh, Sigmoid, Softmax, ReLU

from salience_sum_auto.module.attention import Attention
from salience_sum_auto.module.copy_attention import CopyAttention


class AutoDecoder(Module, Registrable):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 attention: Attention,
                 num_layers: int,
                 training: bool = True) -> None:
        super().__init__()
        self.vocab = None
        self._attention = attention
        self.training = training
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self._rnn = LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         batch_first=True)
        self._score_MLP = Sequential(
            Linear(self.hidden_size, 1, bias=True),
            ReLU()
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

    def forward(self, embedded_src: torch.Tensor,
                state: Dict[str, torch.Tensor]) \
            -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        states = state['encoder_states']
        source_mask = state['source_mask']
        # Initial decoder state
        final_state = (
            state['encoder_final_state'].transpose(0, 1).contiguous(),
            state['encoder_context'].transpose(0, 1).contiguous()
        )
        # (B, L_src, 1)
        dec_state, final_state = self._rnn(
            embedded_src,
            final_state
        )
        context, attention_hidden, attention = self._attention(
            dec_state, states, source_mask)
        vocab_dist = self._gen_vocab_dist(attention_hidden).squeeze(1)
        final_dist = vocab_dist
        scores = self._score_MLP(attention_hidden)
        class_log_probs = (final_dist + 1e-20).log()
        state['class_log_probs'] = class_log_probs
        meta_state = {
            'scores': scores
        }
        return state, meta_state

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass
