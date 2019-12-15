from typing import Dict, Tuple, List

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, util
from allennlp.nn.beam_search import BeamSearch
from torch.nn import Sequential, Linear, LogSoftmax, CrossEntropyLoss

from pointer_generator.module.decoder import Decoder
from pointer_generator.module.encoder import Encoder


@Model.register("encoder_decoder")
class EncoderDecoder(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Encoder,
                 decoder: Decoder,
                 hidden_size: int,
                 max_target_size: int,
                 beam_size: int,
                 vocab: Vocabulary,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self._max_target_size = max_target_size
        self._beam_size = beam_size
        self.hidden_size = hidden_size
        self._embedder = embedder
        self._encoder = encoder
        self._decoder = decoder
        # self._beam_search = BeamSearch(self._end_index, max_steps=self._max_target_size, beam_size=self._beam_size)
        self._generator = Sequential(
            Linear(self.hidden_size, vocab.get_vocab_size()),
            LogSoftmax(dim=2)
        )
        self._criterion = CrossEntropyLoss(reduction='sum')

    def forward(self, source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor]) \
            -> Dict[str, float]:
        """
        The forward function of the encoder and decoder model

        :param source_tokens: Indexes of source tokens
        :param target_tokens: Indexes of target tokens
        :return: The loss and prediction of the model
        """
        # TODO: make _make_predictions
        state = self._encode(source_tokens)
        state, meta_state = self._decode(target_tokens, state)
        loss = self._compute_loss(source_tokens, target_tokens, state, meta_state)
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
        Encode the source tokens

        :param source_tokens: The indexes of source tokens
        :return: All the states and the last state
        """
        # (Batch, Seq, Emb Dim)
        embedded_src = self._embedder(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens)
        # (Batch, Seq, Num Direction * Hidden), final_state = (last state, last context)
        states, final_state = self._encoder(embedded_src)
        state = {
            'encoder_states': states,
            'encoder_final_state': final_state[0],
            'encoder_context': final_state[1],
            'source_mask': source_mask
        }
        return state

    def _decode(self, target_tokens: Dict[str, torch.Tensor],
                state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        Decode the encoder state
        :param target_tokens: The indexes of target tokens
        :param enc_states: All the encoder states
        :param enc_state: The last encoder state
        :return: The output of decoder, attentions and last decoding state
        """
        embedded_tgt = self._embedder(target_tokens)
        state, meta_state = self._decoder(embedded_tgt, state)
        return state, meta_state

    def _compute_loss(self, source_tokens: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.Tensor],
                      state: Dict[str, torch.Tensor],
                      meta_state: Dict[str, List[torch.Tensor]]):
        tgt = target_tokens['tokens']
        src = source_tokens['tokens']
        dec_states = state['decoder_states']
        attentions = meta_state['attentions']
        coverages = meta_state['coverages']
        p_gens = meta_state['p_gens']
        # (B, L, V)
        class_log_probs = self._generator(dec_states)
        batch_size = class_log_probs.size(0)
        length = tgt.size(1)
        loss = 0.0
        num_item = 0
        for b in range(batch_size):
            for l in range(length):
                idxs = torch.nonzero(src[b] == tgt[b][l]).squeeze(1)
                gold_token = target_tokens['tokens'][b][l]
                loss_coverage = sum(torch.min(attentions[l][b], coverages[l][b]))
                loss_vocab = -class_log_probs[b][l][gold_token]
                loss_copy = -sum(attentions[l][b].index_select(0, idxs).log())
                loss += p_gens[l][b][0] * loss_vocab + \
                        (torch.tensor([1]) - p_gens[l][b][0]) * loss_copy + loss_coverage
                num_item += 1

        return loss / num_item

    def _make_predictions(self, dec_states, final_state):

        pass
