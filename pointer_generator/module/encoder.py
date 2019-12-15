import torch
from allennlp.modules import Seq2SeqEncoder
from torch.nn import LSTM, Linear, Sequential, ReLU
from typing import Dict, Tuple


@Seq2SeqEncoder.register('my_encoder')
class Encoder(Seq2SeqEncoder):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 stateful: bool = False) -> None:
        super().__init__(stateful)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self._rnn = LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         bidirectional=self.bidirectional,
                         batch_first=True)
        self._reduce = Sequential(
            Linear(2 * self.hidden_size, self.hidden_size, bias=True),
            ReLU()
        )

    def forward(self, embedded_src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        states, final = self._rnn(embedded_src)
        batch_size = states.size(0)
        # B x 2 * num_layer x Hidden
        state, context = final
        if self.bidirectional:
            state = self._reduce(state.view(batch_size, self.num_layers, -1))
            context = self._reduce(context.view(batch_size, self.num_layers, -1))
        return states, (state, context)

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self) -> int:
        return self.hidden_size

    def is_bidirectional(self) -> bool:
        return self.bidirectional
