import torch
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from torch.nn import LSTM, Linear, Sequential, ReLU
from typing import Dict, Tuple

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@Seq2SeqEncoder.register('salience_encoder')
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
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.input_size = input_size
        self._rnn = LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         bidirectional=self.bidirectional,
                         batch_first=True)
        self._reduce = Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, embedded_src: torch.Tensor, source_mask: torch.Tensor) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        source_lengths = get_lengths_from_binary_sequence_mask(source_mask)
        packed_src = pack_padded_sequence(embedded_src, source_lengths,
                                          batch_first=True, enforce_sorted=False)
        # states = (B x L X 2*H)
        packed_states, final = self._rnn(packed_src)
        states, _ = pad_packed_sequence(packed_states, batch_first=True)
        batch_size = states.size(0)
        # final_states and context = (B x 2*num_layer x H)
        final_state, context = final
        # Reducing the dual hidden size to one hidden size
        if self.bidirectional:
            final_state = self._reduce(final_state.view(batch_size, self.num_layers, -1))
            context = self._reduce(context.view(batch_size, self.num_layers, -1))
        return states, (final_state, context)

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self) -> int:
        return self.hidden_size

    def is_bidirectional(self) -> bool:
        return self.bidirectional
