import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class DenoisingEncoder(nn.Module):

    def __init__(self, bidirectional, num_layers,
                 input_size, hidden_size, dropout=0.0,
                 use_bridge=False):
        super(DenoisingEncoder, self).__init__()
        self.hidden_size = hidden_size

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.no_pack_padded_seq = False
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=True)
        self.dropout = nn.Dropout()
        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(hidden_size, num_layers)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        src, src_length = unpack(src, batch_first=True)
        src = pack(self.dropout(src), src_length, batch_first=True)
        packed_src = src
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_src = pack(src, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_src)

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return memory_bank, encoder_final

    def _initialize_bridge(self, hidden_size, num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
