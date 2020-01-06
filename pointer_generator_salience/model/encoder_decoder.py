from typing import Dict, Tuple, Any

import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, util
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from torch.distributions import Categorical, Gumbel
from torch.nn import CrossEntropyLoss, Softmax, MSELoss, NLLLoss

from pointer_generator_salience.module.decoder import Decoder
from pointer_generator_salience.module.encoder import Encoder
from pointer_generator_salience.module.salience_predictor import SaliencePredictor


@Model.register("encoder_decoder_salience")
class EncoderDecoder(Model):

    def __init__(self,
                 source_embedder: TextFieldEmbedder,
                 target_embedder: TextFieldEmbedder,
                 salience_lambda: float,
                 coverage_lambda: float,
                 max_steps: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 salience_predictor: SaliencePredictor,
                 vocab: Vocabulary,
                 teacher_force_ratio: float,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        # TODO: Workon BeamSearch, try to switch to OpenNMT BeamSearch but implement our own beamsearch first
        self.coverage_lambda = coverage_lambda
        self.salience_lambda = salience_lambda
        self.max_steps = max_steps
        self.source_embedder = source_embedder
        self.target_embedder = target_embedder
        self.target_embedder._modules['token_embedder_tokens'].weight = \
            self.source_embedder._modules['token_embedder_tokens'].weight
        self.encoder = encoder
        self.hidden_size = self.encoder.get_output_dim()
        self.salience_predictor = salience_predictor
        self.decoder = decoder
        self.teacher_force_ratio = teacher_force_ratio
        self.decoder.add_vocab(self.vocab)
        self.padding_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self.start_idx = self.vocab.get_token_index(START_SYMBOL)
        self.end_idx = self.vocab.get_token_index(END_SYMBOL)
        self.unk_idx = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self.beam = BeamSearch(self.end_idx, max_steps=self.max_steps, beam_size=10, per_node_beam_size=5)
        self.criterion = NLLLoss(ignore_index=self.padding_idx)
        self.prediction_criterion = MSELoss()
        self.salience_MSE = 0.0
        self.coverage_loss = 0.0

    # noinspection PyMethodMayBeStatic
    def init_enc_state(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source_mask = util.get_text_field_mask(source_tokens)
        source_lengths = get_lengths_from_binary_sequence_mask(source_mask)
        state = {
            'source_mask': source_mask,  # (B, L)
            'source_lengths': source_lengths,  # (L)
            'source_tokens': source_tokens['tokens'],
        }
        return state

    def init_dec_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = state['encoder_states']
        batch_size = states.size(0)
        length = states.size(1)
        state['context'] = states.new_zeros((batch_size, 1, self.hidden_size))
        state['dec_state'] = state['hidden']
        state['coverage'] = states.new_zeros((batch_size, length, 1))  # (B, L, 1)
        state['hidden_context'] = state['hidden'].new_zeros(state['hidden'].size())
        return state

    def forward(self,
                source_tokens: Dict[str, torch.Tensor],
                source_text: Dict[str, Any],
                source_ids: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor] = None,
                target_ids: torch.Tensor = None,
                salience_values: torch.Tensor = None) \
            -> Dict[str, torch.Tensor]:
        """
        The forward function of the encoder and decoder model

        :param source_ids: The source ids that is unique to the document
        :param source_text: The raw text of source sequence
        :param salience_values: The saliency values for source tokens
        :param source_tokens: Indexes of states tokens
        :param target_tokens: Indexes of target tokens
        :param target_ids: Similar with target_tokens but mapped with extended vocabulary
        :return: The loss and prediction of the model
        """
        state = self._encode(source_tokens)
        if self.salience_lambda != 0.0:
            predicted_salience = self._predict_salience(state)
        else:
            predicted_salience = None
        output_dict = {}

        if target_tokens:
            state = self._decode(source_ids, target_tokens, state)
            output_dict['loss'] = self._compute_loss(target_tokens, target_ids,
                                                     salience_values, predicted_salience, state)

        if not self.training and not target_tokens:
            output_dict['results'] = self._forward_beam_search(state, source_ids)
        return output_dict

    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor],
                             source_ids: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        state = self.init_dec_state(state)
        state['source_ids'] = source_ids['ids']
        state['max_oov'] = source_ids['max_oov']
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self.start_idx)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self.beam.search(
            start_predictions, state, self.take_step)
        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        emb = self.target_embedder({
            'tokens': last_predictions
        })
        state = self.decoder(emb, state)
        return state['class_probs'].squeeze(1).log(), state

    def _predict_salience(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        predicted_salience = self.salience_predictor(state)
        return predicted_salience

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode the states tokens

        :param source_tokens: The indexes of states tokens
        :return: All the states and the last state
        """
        state = self.init_enc_state(source_tokens)
        # (Batch, Seq, Emb Dim)
        embedded_src = self.source_embedder(source_tokens)

        # final_state = (last state, last context)
        states_features, states, final_state = self.encoder(embedded_src, state['source_lengths'])
        state['encoder_states'] = states  # (B, L, Num Direction * D_h)
        state['hidden'] = final_state  # (B, L, Num Direction * D_h)
        state['states_features'] = states_features  # (B, L, Num Direction * D_h)
        assert state['encoder_states'].size(2) == self.hidden_size
        return state

    def _decode(self,
                source_ids: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor],
                state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decode the encoder state
        :param target_tokens: The indexes of target tokens
        :param enc_states: All the encoder states
        :param enc_state: The last encoder state
        :return: The output of decoder, attentions and last decoding state
        """
        state = self.init_dec_state(state)
        state['source_ids'] = source_ids['ids']
        state['max_oov'] = source_ids['max_oov']

        all_class_probs = []
        batch_size, length = state['source_mask'].size()
        all_coverages = [state['source_mask'].new_zeros(
            (batch_size, length, 1), dtype=torch.float)]
        all_attentions = []
        # Teacher Forcing
        if torch.rand(1).item() <= self.teacher_force_ratio:
            embedded_tgt = self.target_embedder(target_tokens)
            for step, emb in enumerate(embedded_tgt.split(1, dim=1)):
                state = self.decoder(emb, state)
                all_class_probs.append(state['class_probs'])
                all_coverages.append(state['coverage'])
                all_attentions.append(state['attention'])
        else:
            tokens = state["encoder_states"].new_full(
                (state["encoder_states"].size(0),), fill_value=self.start_idx, dtype=torch.long)
            emb = self.target_embedder({'tokens': tokens})
            for step in range(target_tokens['tokens'].size(1)):
                state = self.decoder(emb, state)
                all_class_probs.append(state['class_probs'])
                all_coverages.append(state['coverage'])
                all_attentions.append(state['attention'])
                class_prob = all_class_probs[-1].squeeze(1)
                # gumbel_sample = gumbel.rsample(class_logit.size()).squeeze(2)
                # _, tokens = (class_logit + gumbel_sample).topk(1)
                prob_dist = Categorical(class_prob)
                tokens = prob_dist.sample()
                # _, tokens = class_prob.topk(1)
                tokens[tokens >= self.vocab.get_vocab_size()] = self.unk_idx
                emb = self.target_embedder({'tokens': tokens})
            # print(predicted_tokens)
        state['all_class_probs'] = torch.cat(all_class_probs, dim=1)
        state['all_coverages'] = torch.cat(all_coverages[:-1], dim=2)
        state['all_attentions'] = torch.cat(all_attentions, dim=2)
        state.pop('class_probs', None)
        state.pop('coverage', None)
        state.pop('attention', None)
        return state

    def _compute_loss(self,
                      target_tokens: Dict[str, torch.Tensor],
                      target_ids: torch.Tensor,
                      salience_values: torch.Tensor,
                      predicted_salience: torch.Tensor,
                      state: Dict[str, torch.Tensor]):
        # (B, L, V)
        all_class_log_probs = state['all_class_probs']
        attentions = state['all_attentions']
        coverages = state['all_coverages']
        source_mask = state['source_mask']
        target_mask = util.get_text_field_mask(target_tokens)
        # (B, L, 1)
        length = all_class_log_probs.size(1)
        step_losses = []
        for i in range(length):
            target = target_ids[:, i]
            gold_probs = torch.gather(all_class_log_probs[:, i], 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs)
            step_coverage_loss = torch.sum(torch.min(attentions[:, i], coverages[:, i]), 1)
            step_loss = step_loss + self.coverage_lambda * step_coverage_loss
            step_loss = step_loss * target_mask[:, i]
            step_losses.append(step_loss.unsqueeze(1))

        sum_losses = torch.cat(step_losses, dim=1).sum(1)
        batch_avg_loss = sum_losses / util.get_lengths_from_binary_sequence_mask(target_mask)
        total_loss = torch.mean(batch_avg_loss)
        # loss = self.criterion(
        #     all_class_log_probs[:, :-1, :].transpose(1, 2),
        #     target_ids[:, 1:all_class_log_probs.size(1)])
        # coverage_loss = torch.min(attentions, coverages).sum(1).mean()
        if predicted_salience is not None:
            predicted_salience = source_mask * predicted_salience.squeeze(2)
            salience_values = source_mask * salience_values
            salience_loss = self.prediction_criterion(predicted_salience, salience_values)
            total_loss = total_loss + self.salience_lambda * salience_loss
            self.salience_MSE = salience_loss.item()
        # self.coverage_loss = coverage_loss.item()
        return total_loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            'salience_MSE': self.salience_MSE,
            'coverage_loss': self.coverage_loss
        }
        if reset:
            self.salience_MSE = 0.0
            self.coverage_loss = 0.0
        return metrics_to_return
    # def _predict(self, source_tokens: Dict[str, torch.Tensor], state: Dict[str, torch.Tensor]):
    #     states = state['encoder_states']
    #     source_lengths = state['source_lengths']
    #     source_mask = state['source_mask']
    #     fn_map_state, states, source_lengths, src_map = \
    #         self.beam.initialize(
    #             states.transpose(0, 1).contiguous(),
    #             source_lengths, device='cpu')
    #     states = states.transpose(0, 1).contiguous()
    #
    #     final_state = (
    #         state['final_state'].transpose(0, 1).contiguous(),
    #         state['final_context'].transpose(0, 1).contiguous()
    #     )
    #     final_state = (
    #         final_state[0].repeat(1, self.beam.beam_size, 1),
    #         final_state[1].repeat(1, self.beam.beam_size, 1)
    #     )
    #     source_mask = source_mask.repeat(self.beam.beam_size, 1)
    #     coverage = states.new_zeros((states.size(0), states.size(1), 1))
    #     for step in range(self.max_steps):
    #         generated_token = {
    #             'tokens': self.beam.current_predictions.view(-1, 1)
    #         }
    #         emb = self.source_embedder(generated_token)
    #         class_log_prob, final_state, coverage, attention = \
    #             self.decoder(source_tokens, emb, final_state, coverage, states, source_mask)
    #         self.beam.advance(
    #             class_log_prob.squeeze(1),
    #             attention.view(1, attention.size(0), -1))
    #         any_finished = self.beam.is_finished.any()
    #         if any_finished:
    #             self.beam.update_finished()
    #             if self.beam.done:
    #                 break
    #         select_indices = self.beam.select_indices
    #
    #         if any_finished:
    #             # Reorder states.
    #             if isinstance(states, tuple):
    #                 states = tuple(x.index_select(0, select_indices)
    #                                for x in states)
    #             else:
    #                 states = states.index_select(0, select_indices)
    #
    #             source_lengths = source_lengths.index_select(0, select_indices)
    #
    #             if src_map is not None:
    #                 src_map = src_map.index_select(0, select_indices)
    #
    #         if self.beam.beam_size > 1 or any_finished:
    #             coverage = coverage.index_select(0, select_indices)
    #             final_state = (
    #                 final_state[0].index_select(1, select_indices),
    #                 final_state[1].index_select(1, select_indices)
    #             )
    #             source_mask = source_mask.index_select(0, select_indices)
    #     return self.beam.predictions
