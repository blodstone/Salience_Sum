import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from torch.nn import Module, LSTM, Sequential, Linear, Sigmoid, Softmax, ReLU, Dropout
from typing import Dict, Tuple, Union

from pointer_generator_salience.module.attention import Attention


class Decoder(Module, Registrable):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 attention: Attention = None,
                 training: bool = True) -> None:
        super().__init__()
        self.vocab = None
        self.training = training
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = LSTM(input_size=self.input_size,
                        hidden_size=2 * self.hidden_size,
                        num_layers=self.num_layers,
                        batch_first=True)
        self.input_context = Linear(
            2*hidden_size + input_size, input_size)
        if attention is None:
            self.is_attention = False
        else:
            self.is_attention = True
            self.attention = attention
            self._p_gen = Sequential(
                Linear(2*self.hidden_size + 2*self.hidden_size + self.input_size, 1, bias=True),
                Sigmoid()
            )
        self.gen_vocab_dist = None

    def add_vocab(self, vocab: Vocabulary):
        self.vocab = vocab
        self.gen_vocab_dist = Sequential(
            Linear(4*self.hidden_size, 2*self.hidden_size, bias=True),
            Linear(2*self.hidden_size, self.vocab.get_vocab_size(), bias=True)
        )

    def get_output_dim(self) -> int:
        return self.hidden_size

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

    def forward(self,
                input_emb: torch.Tensor,
                state: Dict[str, Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]]]):
        source_ids = state['source_ids']
        max_oov = state['max_oov']
        states = state['encoder_states']
        states_features = state['states_features']
        hidden = state['hidden']
        context = state['context']
        dec_state = state['dec_state']
        source_mask = state['source_mask']
        coverage = state['coverage']
        if len(input_emb.size()) == 2:
            input_emb = input_emb.unsqueeze(1)
        batch_size = input_emb.size(0)
        hidden_context = None
        attention = None
        if self.is_attention:
            # Dec_state (s_{j-1} is initialized with the last encoder hidden state (h_n))
            hidden_context, coverage, attention = self.attention(
                dec_state, states, states_features, source_mask, coverage)
        dec_state, final = self.rnn(
            self.input_context(
                torch.cat((hidden_context, input_emb), dim=2)),
            (hidden.view(-1, batch_size, 2*self.hidden_size),
             context.view(-1, batch_size, 2*self.hidden_size))
        )
        if self.is_attention:
            state['class_logits'] = self._build_class_logits(
                attention, hidden_context, dec_state, input_emb, source_ids, max_oov)
        else:
            state['class_logits'] = self._build_class_logits_no_attn(dec_state)
        state['dec_state'] = dec_state
        state['coverage'] = coverage
        state['attention'] = attention
        state['hidden'] = final[0].view(batch_size, -1, self.hidden_size)
        state['context'] = final[1].view(batch_size, -1, self.hidden_size)
        return state

    def _build_class_logits(self,
                            attention: torch.Tensor,
                            hidden_context: torch.Tensor,
                            dec_state: torch.Tensor,
                            input_emb: torch.Tensor,
                            source_ids: torch.Tensor,
                            max_oov: torch.Tensor
                            ) -> torch.Tensor:
        p_gen = self._p_gen(torch.cat((hidden_context, dec_state, input_emb), dim=2))
        vocab_dist = (p_gen * self.gen_vocab_dist(torch.cat((hidden_context, dec_state), dim=2))).squeeze(1)
        if (max_oov.max()+1).item() > self.vocab.get_vocab_size():
            extended_vocab = vocab_dist.new_zeros([vocab_dist.size(0), max_oov.max()+1])
            extended_vocab[:, :vocab_dist.size(1)] = vocab_dist
        else:
            extended_vocab = vocab_dist
        attn_dist = ((1 - p_gen) * attention).squeeze(2)
        final_dist = extended_vocab.scatter_add(1, source_ids, attn_dist).unsqueeze(1)
        # some logits might zero

        class_logits = final_dist + 1e-13
        return class_logits

    def _build_class_logits_no_attn(self,
                                    dec_state: torch.Tensor,
                                    ) -> torch.Tensor:
        class_logits = self.gen_vocab_dist(dec_state).squeeze(1)
        return class_logits
