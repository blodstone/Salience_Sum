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
                 attention: Attention = None,
                 training: bool = True) -> None:
        super().__init__()
        self.vocab = None
        self.training = training
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = LSTM(input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=1,
                        batch_first=True)
        self.input_context = Linear(
            hidden_size + input_size, input_size)
        if attention is None:
            self.is_attention = False
        else:
            self.is_attention = True
            self.attention = attention
            self._p_gen = Sequential(
                Linear(self.hidden_size, 1, bias=True),
                Sigmoid()
            )
        self.gen_vocab_dist = None

    def add_vocab(self, vocab: Vocabulary):
        self.vocab = vocab
        self.gen_vocab_dist = Sequential(
            Linear(self.hidden_size, self.vocab.get_vocab_size()),
            Softmax(dim=-1)
        )

    def get_output_dim(self) -> int:
        return self.hidden_size

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

    def forward(self,
                input_emb: torch.Tensor,
                state: Dict[str, Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]]],
                is_coverage: bool,
                is_training: bool = True,
                is_first_step: bool = False,
                ):
        input_feed = state['input_feed']
        source_ids = state['source_ids']
        max_oov = state['max_oov']
        states = state['encoder_states']
        states_features = state['states_features']
        hidden = state['hidden']
        context = state['context']
        source_mask = state['source_mask']
        coverage = state['coverage']
        if len(input_emb.size()) == 2:
            input_emb = input_emb.unsqueeze(1)
        batch_size = input_emb.size(0)
        attention = None
        x = self.input_context(torch.cat((input_emb, input_feed), dim=2))
        rnn_output, final = self.rnn(x, (hidden, context))
        if self.is_attention:
            # Attention step 0 to calculate coverage step 1
            decoder_output, coverage, attention = self.attention(
                rnn_output, states, states_features, source_mask, coverage, is_coverage)
            input_feed = decoder_output
        else:
            input_feed = rnn_output

        if self.is_attention:
            state['class_probs'], state['p_gen'] = self._build_class_logits(
                attention, decoder_output, rnn_output, x, source_ids, max_oov)
        else:
            state['class_probs'], state['p_gen'] = self._build_class_logits_no_attn(decoder_output)
        state['coverage'] = coverage
        state['attention'] = attention
        state['input_feed'] = input_feed
        state['hidden'] = final[0].view(1, batch_size, self.hidden_size)
        state['context'] = final[1].view(1, batch_size, self.hidden_size)
        return state

    def _build_class_logits(self,
                            attention: torch.Tensor,
                            decoder_output: torch.Tensor,
                            rnn_output: torch.Tensor,
                            x: torch.Tensor,
                            source_ids: torch.Tensor,
                            max_oov: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        p_gen = self._p_gen(decoder_output)
        vocab_dist = (p_gen * self.gen_vocab_dist(decoder_output)).squeeze(1)
        if (max_oov.max() + 1).item() > self.vocab.get_vocab_size():
            extended_vocab = vocab_dist.new_zeros([vocab_dist.size(0), max_oov.max() + 1])
            extended_vocab[:, :vocab_dist.size(1)] = vocab_dist
        else:
            extended_vocab = vocab_dist

        attn_dist = ((p_gen.new_ones(p_gen.size()) - p_gen) * attention).squeeze(2)
        final_dist = extended_vocab.scatter_add(1, source_ids, attn_dist).unsqueeze(1)
        return final_dist, p_gen

    def _build_class_logits_no_attn(self,
                                    dec_state: torch.Tensor,
                                    ) -> torch.Tensor:
        final_dist = self.gen_vocab_dist(dec_state).squeeze(1)
        return final_dist
