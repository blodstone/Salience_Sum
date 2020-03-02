import torch
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from torch import Tensor
from torch.nn import LSTM, Linear, Sequential, ReLU
from typing import Dict, Tuple

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(Seq2SeqEncoder):
    """
    A standard LSTM encoder that supports bidirectional. If bidirectional is True, we split
    the hidden layer and then concatenate the two directions in the resulting encoder states.
    Everything is on first batch basis.
    """

    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 stateful: bool = False) -> None:
        super().__init__(stateful)
        self.hidden_size = hidden_size
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.num_layers = num_layers
        self.input_size = input_size
        self.rnn = LSTM(input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bidirectional=bidirectional,
                        batch_first=True)
        self._linear_source = Linear(self.hidden_size * self.num_directions,
                                     self.hidden_size * self.num_directions, bias=False)
        self._reduce_h = Sequential(
            Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=True),
            ReLU()
        )
        self._reduce_c = Sequential(
            Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=True),
            ReLU()
        )

    def forward(self, embedded_src: torch.Tensor,
                source_lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packed_src = pack_padded_sequence(embedded_src, source_lengths,
                                          batch_first=True, enforce_sorted=False)
        # noinspection PyTypeChecker
        packed_states, final = self.rnn(packed_src)
        # states = (B x L X num_dir*D_h) , last layer
        # noinspection PyTypeChecker
        states, length = pad_packed_sequence(packed_states, batch_first=True)
        final_state = self._reduce_h(self.get_final_layer(final[0]))
        final_context_state = self._reduce_c(self.get_final_layer(final[1]))
        # Save times for attention mechanism
        states_features = self._linear_source(states)
        assert states.size(2) == self.num_directions * self.hidden_size
        return states_features, states, final_state, final_context_state

    @staticmethod
    def get_final_layer(state):
        # (B, Num_dir * D_h)
        state = torch.cat([state[0:state.size(0):2],
                           state[1:state.size(0):2]], 2)
        return state[-1].unsqueeze(0)

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self) -> int:
        return 2 * self.hidden_size

    def is_bidirectional(self) -> bool:
        return self.bidirectional
