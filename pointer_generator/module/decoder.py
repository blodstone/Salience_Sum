from typing import Dict, Tuple, List

import torch
from allennlp.common import Registrable
from torch.nn import LSTM, Module, Sequential, Linear, Tanh, Sigmoid

from pointer_generator.module.attention import Attention


class Decoder(Module, Registrable):

    def __init__(self, input_size: int,
                 hidden_size: int,
                 attention: Attention,
                 num_layers: int,
                 training: bool = True) -> None:
        super().__init__()
        self.training = training
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self._attention = attention
        self._rnn = LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         batch_first=True)
        self._context = Sequential(
            Linear(self.hidden_size * 2, self.hidden_size, bias=True),
            Linear(self.hidden_size, self.hidden_size, bias=True)
        )
        self._p_gen = Sequential(
            Linear(self.hidden_size + self.hidden_size + self.input_size, 1, bias=True),
            Sigmoid()
        )
        # self._context = Sequential(
        #     Linear(self.input_size + self.hidden_size, self.input_size),
        #     Tanh()
        # )

    def get_output_dim(self) -> int:
        return self.hidden_size

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

    def forward(self, embedded_tgt: torch.Tensor,
                final_state: Tuple[torch.Tensor, torch.Tensor],
                states: torch.Tensor) -> Tuple[List[torch.Tensor],
                                               List[torch.Tensor],
                                               List[torch.Tensor]]:
        # A static context c for all time-steps
        final_state = (
            final_state[0].transpose(0, 1).contiguous(),
            final_state[1].transpose(0, 1).contiguous()
        )
        dec_states = []
        p_gens = []
        attentions = []
        coverages = []
        coverage = torch.zeros(states.size(0), states.size(1), 1)
        for step, emb in enumerate(embedded_tgt.split(1, 1)):
            dec_state, final_state = self._rnn(
                emb,
                final_state
            )
            context, attention = self._attention(
                torch.cat((dec_state, dec_state), dim=2), states, coverage)
            p_gen = self._p_gen(torch.cat((context, dec_state, emb), dim=2))
            dec_state = self._context(torch.cat((dec_state, context), dim=2))
            attentions.append(attention)
            coverage = sum(attentions)
            coverages.append(coverage)
            dec_states.append(dec_state)
            p_gens.append(p_gen)
        return dec_states, p_gens, attentions, coverages

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass
