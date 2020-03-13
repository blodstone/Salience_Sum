import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from torch.nn import Module, LSTM, Sequential, Linear, Sigmoid, Softmax, ReLU, Dropout
from typing import Dict, Tuple, Union

from pg_salience_feature.module.attention import Attention


class Decoder(Module, Registrable):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 is_emb_attention: bool = False,
                 use_copy_mechanism: bool = True,
                 emb_attention_mode: str = 'mlp',
                 attention: Attention = None,
                 training: bool = True) -> None:
        super().__init__()
        self.use_copy_mechanism = use_copy_mechanism
        self.emb_attention_mode = emb_attention_mode
        self.is_emb_attention = is_emb_attention
        self.vocab = None
        self.training = training
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = LSTM(input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        num_layers=1,
                        batch_first=True)
        self.input_context = Linear(
            2 * hidden_size + input_size, input_size)
        if attention is None:
            self.is_attention = False
        else:
            self.is_attention = True
            self.attention = attention
            self._p_gen = Sequential(
                Linear(self.hidden_size * 4 + input_size, 1, bias=True),
                Sigmoid()
            )
        self.gen_vocab_dist = None

    def add_vocab(self, vocab: Vocabulary):
        self.vocab = vocab
        self.gen_vocab_dist = Sequential(
            Linear(self.hidden_size * 3, self.hidden_size),
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
        if 'emb_salience_feature' in state.keys():
            emb_salience_feature = state['emb_salience_feature']
        else:
            emb_salience_feature = None
        hidden = state['hidden']
        context = state['context']
        source_mask = state['source_mask']
        coverage = state['coverage']
        if len(input_emb.size()) == 2:
            input_emb = input_emb.unsqueeze(1)
        batch_size = input_emb.size(0)
        attention = None
        x = self.input_context(torch.cat((input_emb, input_feed), dim=2))
        rnn_output, final = self.rnn(x,
                                     (hidden.view(1, batch_size, hidden.size(2)),
                                      context.view(1, batch_size, hidden.size(2))))
        final_hidden, final_context = final
        final_hat = torch.cat((
            final_hidden.view(batch_size, 1, hidden.size(2)),
            final_context.view(batch_size, 1, hidden.size(2))), dim=2)
        if self.is_attention:
            # Attention step 0 to calculate coverage step 1
            decoder_output, coverage, attention = \
                self.attention(
                    final_hat, states, states_features,
                    source_mask, coverage, is_coverage,
                    emb_salience_feature, self.is_emb_attention,
                    self.emb_attention_mode
                )
            input_feed = decoder_output
        else:
            input_feed = rnn_output

        if self.is_attention:
            if self.use_copy_mechanism:
                state['class_probs'], state['p_gen'] = self._build_class_logits(
                    attention, decoder_output, final_hat, rnn_output, x, source_ids, max_oov)
            else:
                state['class_probs'], state['p_gen'] = self._build_class_logits_no_copy(decoder_output, rnn_output)
        else:
            state['class_probs'], state['p_gen'] = self._build_class_logits_no_attn(decoder_output)
        state['coverage'] = coverage
        state['attention'] = attention
        state['input_feed'] = input_feed
        state['hidden'] = final[0].view(batch_size, 1, self.hidden_size)
        state['context'] = final[1].view(batch_size, 1, self.hidden_size)
        return state

    def _build_class_logits_no_copy(self,
                                    decoder_output: torch.Tensor,
                                    rnn_output: torch.Tensor):
        out = torch.cat((decoder_output, rnn_output), dim=2)
        vocab_dist = self.gen_vocab_dist(out)
        p_gen = decoder_output.new_zeros((decoder_output.size(0), 1, 1))
        return vocab_dist, p_gen

    def _build_class_logits(self,
                            attention: torch.Tensor,
                            decoder_output: torch.Tensor,
                            final_hat: torch.Tensor,
                            rnn_output: torch.Tensor,
                            x: torch.Tensor,
                            source_ids: torch.Tensor,
                            max_oov: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        p_gen = self._p_gen(torch.cat((decoder_output, final_hat, x), dim=2))
        out = torch.cat((decoder_output, rnn_output), dim=2)
        vocab_dist = (p_gen * self.gen_vocab_dist(out)).squeeze(1)
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
