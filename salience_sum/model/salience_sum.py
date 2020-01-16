from typing import Dict, Tuple, List

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, InputVariationalDropout
from allennlp.nn import RegularizerApplicator, util
from allennlp.nn.beam_search import BeamSearch
from torch.nn import Sequential, Linear, LogSoftmax, CrossEntropyLoss, NLLLoss

from salience_sum.module.auto_decoder import AutoDecoder
from salience_sum.module.summ_decoder import Decoder
from salience_sum.module.encoder import Encoder


@Model.register("salience_sum")
class EncoderDecoder(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Encoder,
                 decoder: AutoDecoder,
                 dropout: float,
                 hidden_size: int,
                 max_target_size: int,
                 beam_size: int,
                 vocab: Vocabulary,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        torch.autograd.set_detect_anomaly(True)
        self._max_target_size = max_target_size
        self._beam_size = beam_size
        self.hidden_size = hidden_size
        self._embedder = embedder
        self._encoder = encoder
        self._decoder = decoder
        self._dropout = InputVariationalDropout(dropout)
        self._decoder.add_vocab(self.vocab)
        # self._beam_search = BeamSearch(self._end_index, max_steps=self._max_target_size, beam_size=self._beam_size)
        padding_idx = self.vocab.get_token_to_index_vocabulary()[self.vocab._padding_token]
        self._criterion = NLLLoss(
            reduction='sum', ignore_index=padding_idx)

    def forward(self, source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor],
                salience_values: torch.Tensor) \
            -> Dict[str, torch.Tensor]:
        """
        The forward function of the encoder and decoder model

        :param source_tokens: Indexes of states tokens
        :param target_tokens: Indexes of target tokens
        :return: The loss and prediction of the model
        """
        # TODO: make _make_predictions
        state = self._encode(source_tokens)
        # state, meta_state = self._summ_decode(source_tokens, target_tokens, state)
        state, meta_state = self._auto_decode(source_tokens, state)
        loss = self._compute_loss(source_tokens, salience_values, state, meta_state)
        if not self.training:
            pass
            # state = {
            #     'source_mask': None,
            #     'encoder_outputs': ,
            #     'decoder_hidden': None,
            #     'decoder_context': None
            # }
            # predictions = self._make_predictions(dec_states, final_state)
        return {
            "loss": loss,
            # "predictions": predictions
        }

    # def take_step(self, last_predictions: torch.Tensor,
    #               state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    #     self._decode(last_predictions)
    #     class_log_probs = self._generator(dec_states)
    #     return class_log_probs, state

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode the states tokens

        :param source_tokens: The indexes of states tokens
        :return: All the states and the last state
        """
        # (Batch, Seq, Emb Dim)
        embedded_src = self._dropout(self._embedder(source_tokens))
        # Needed for building packed pad sequence
        source_mask = util.get_text_field_mask(source_tokens)
        # (Batch, Seq, Num Direction * Hidden), final_state = (last state, last context)
        states, final_state = self._encoder(embedded_src, source_mask)
        state = {
            'encoder_states': states,
            'encoder_final_state': final_state[0],
            'encoder_context': final_state[1],
            'source_mask': source_mask
        }
        return state

    def _auto_decode(self,
                     source_tokens: Dict[str, torch.Tensor],
                     state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        Decode the encoder state
        :param target_tokens: The indexes of target tokens
        :param enc_states: All the encoder states
        :param enc_state: The last encoder state
        :return: The output of decoder, attentions and last decoding state
        """
        embedded_src = self._embedder(source_tokens)
        state, meta_state = self._decoder(embedded_src, state)
        return state, meta_state

    def _summ_decode(self,
                     source_tokens: Dict[str, torch.Tensor],
                     target_tokens: Dict[str, torch.Tensor],
                     state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        Decode the encoder state
        :param target_tokens: The indexes of target tokens
        :param enc_states: All the encoder states
        :param enc_state: The last encoder state
        :return: The output of decoder, attentions and last decoding state
        """
        embedded_tgt = self._embedder(target_tokens)
        state, meta_state = self._decoder(source_tokens, embedded_tgt, state)
        return state, meta_state

    def _compute_loss(self,
                      source_tokens: Dict[str, torch.Tensor],
                      salience_values: torch.Tensor,
                      state: Dict[str, torch.Tensor],
                      meta_state: Dict[str, List[torch.Tensor]]):
        # (B, L, V)
        class_log_probs = state['class_log_probs']
        scores = meta_state['scores']
        # (B, L, 1)
        length = source_tokens['tokens'].size(1)
        batch_size = class_log_probs.size(0)
        loss = 0
        for b in range(batch_size):
            loss += self._criterion(class_log_probs[b], source_tokens['tokens'][b])
        salience_loss = 0
        for b in range(batch_size):
            # Ignore start and end index
            for l in range(1, length-1):
                salience_loss += (scores[b][l] - salience_values[b][l-1]) * \
                                 (scores[b][l] - salience_values[b][l-1])
        total_loss = loss + salience_loss
        return total_loss / batch_size

    def _make_predictions(self, dec_states, final_state):

        pass
