import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Registrable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from overrides import overrides
import torch
from torch.nn.utils.rnn import pad_packed_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class CustomEncoder(nn.Module, Registrable):
    default_implementation = 'denoising_encoder'


@Seq2SeqEncoder.register('seq2seqwrapper')
class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.

    In order to be wrapped with this wrapper, a class must have the following members:

        - ``self.input_size: int``
        - ``self.hidden_size: int``
        - ``def forward(inputs: PackedSequence, hidden_state: torch.Tensor) ->
          Tuple[PackedSequence, torch.Tensor]``.
        - ``self.bidirectional: bool`` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass a binary mask of shape (batch_size, sequence_length)
    when you call this module, to avoid subtle bugs around masking.  If you already have a
    ``PackedSequence`` you can pass ``None`` as the second parameter.

    We support stateful RNNs where the final state from each batch is used as the initial
    state for the subsequent batch by passing ``stateful=True`` to the constructor.
    """
    def __init__(self, module: CustomEncoder, stateful: bool = False) -> None:
        super(PytorchSeq2SeqWrapper, self).__init__(stateful)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    @overrides
    def get_input_dim(self) -> int:
        return self._module.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    @overrides
    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:

        if self.stateful and mask is None:
            raise ValueError("Always pass a mask with stateful RNNs.")
        if self.stateful and hidden_state is not None:
            raise ValueError("Stateful RNNs provide their own initial hidden_state.")

        if mask is None:
            return self._module(inputs, hidden_state)[0]

        batch_size, total_sequence_length = mask.size()

        packed_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        num_valid = unpacked_sequence_tensor.size(0)
        # Some RNNs (GRUs) only return one state as a Tensor.  Others (LSTMs) return two.
        # If one state, use a single element list to handle in a consistent manner below.
        if not isinstance(final_states, (list, tuple)) and self.stateful:
            final_states = [final_states]

        # Add back invalid rows.
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

            # The states also need to have invalid rows added back.
            if self.stateful:
                new_states = []
                for state in final_states:
                    num_layers, _, state_dim = state.size()
                    zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                    new_states.append(torch.cat([state, zeros], 1))
                final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(batch_size,
                                                       sequence_length_difference,
                                                       unpacked_sequence_tensor.size(-1))
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        if self.stateful:
            self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor.index_select(0, restoration_indices)


@CustomEncoder.register('denoising_encoder')
class DenoisingEncoder(CustomEncoder):

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
