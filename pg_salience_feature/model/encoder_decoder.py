import math
from collections import Counter
from typing import Dict, Tuple, List

import torch
from nltk.util import ngrams
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import RegularizerApplicator, util
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from torch.distributions import Categorical
from torch.nn import NLLLoss

from pg_salience_feature.inference.common_beam_search import CommonBeamSearch
from pg_salience_feature.module.decoder import Decoder
from pg_salience_feature.module.encoder import Encoder


# noinspection DuplicatedCode
from pg_salience_feature.module.salience_src_mixer import SalienceSourceMixer


@Model.register("enc_dec_salience_feature")
class EncoderDecoder(Model):

    def __init__(self,
                 source_embedder: TextFieldEmbedder,
                 target_embedder: TextFieldEmbedder,
                 coverage_lambda: float,
                 max_steps: int,
                 is_result_rouge_ranked: bool,
                 encoder: Encoder,
                 decoder: Decoder,
                 vocab: Vocabulary,
                 teacher_force_ratio: float,
                 beam_search: CommonBeamSearch,
                 use_copy_mechanism: bool = True,
                 salience_source_mixer: SalienceSourceMixer = None,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self.is_result_rouge_ranked = is_result_rouge_ranked
        self.salience_source_mixer = salience_source_mixer
        self.coverage_lambda = coverage_lambda
        self.use_copy_mechanism = use_copy_mechanism
        if coverage_lambda == 0.0:
            self.is_coverage = False
        else:
            self.is_coverage = True
        # For end and start tokens
        self.max_steps = max_steps + 2
        self.source_embedder = source_embedder
        self.target_embedder = target_embedder
        self.target_embedder._modules['token_embedder_tokens'].weight = \
            self.source_embedder._modules['token_embedder_tokens'].weight
        self.encoder = encoder
        self.hidden_size = self.encoder.get_output_dim()
        self.decoder = decoder
        self.teacher_force_ratio = teacher_force_ratio
        self.decoder.add_vocab(self.vocab)
        self.padding_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self.start_idx = self.vocab.get_token_index(START_SYMBOL)
        self.end_idx = self.vocab.get_token_index(END_SYMBOL)
        self.unk_idx = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self.beam_search = beam_search
        self.beam_search.config_beam(end_index=self.end_idx,
                                     max_steps=self.max_steps, unk_index=self.unk_idx)
        self.criterion = NLLLoss(ignore_index=self.padding_idx)
        self.coverage_loss = 0.0
        self.p_gen = 0.0

    def update_constraint_idx(self, raw_constraints, source_text, source_ids):
        new_raw_constraint = []
        word_constraint = []
        token_to_index = self.vocab.get_token_to_index_vocabulary()
        for batch_idx, constraint in enumerate(raw_constraints):
            new_phrase_idx = []
            new_phrase_words = []
            for phrase in constraint:
                vocab_indexes = []
                for word_idx in phrase:
                    text = source_text[batch_idx][int(word_idx)].text
                    if text in token_to_index:
                        vocab_indexes.append(token_to_index[text])
                    else:
                        vocab_indexes.append(
                            source_ids['ids'][batch_idx][int(word_idx)].item())
                vocab_words = [source_text[batch_idx][int(word_idx)].text for word_idx in phrase]
                if self.unk_idx not in vocab_indexes:
                    new_phrase_idx.append(vocab_indexes)
                    new_phrase_words.append(vocab_words)
            if len(new_phrase_idx) != 0:
                new_raw_constraint.append(new_phrase_idx)
                word_constraint.append(new_phrase_words)
        return new_raw_constraint, word_constraint

    def init_enc_state(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source_mask = util.get_text_field_mask(source_tokens)
        source_lengths = get_lengths_from_binary_sequence_mask(source_mask)
        state = {
            'source_mask': source_mask,  # (B, L)
            'source_lengths': source_lengths,  # (L)
            'source_tokens': source_tokens['tokens'],
        }
        return state

    # def weighted_rouge(self):
    #
    #     def numH(w, H):
    #         result = 0
    #         H_group = H.groupby('result_id')
    #         highlights = {}
    #         for result_id, data in H_group:
    #             if result_id not in highlights.keys():
    #                 highlights[result_id] = data['indexes']
    #             else:
    #                 highlights[result_id].append(data['indexes'])
    #         for result_id in highlights.keys():
    #             h_words = list(chain(*[highlight for highlight in highlights[result_id]]))
    #             if w in h_words:
    #                 result += len(h_words) / MAX_LEN
    #         return result
    #
    #     def beta(n, g, w, H):
    #         numerator = 0
    #         denominator = 0
    #         m = len(w[0])
    #         for i in range(m - n + 1):
    #             total_NumH = 0
    #             for j in range(i, i + n):
    #                 if w[0][i:i + n] == list(g):
    #                     total_NumH += numH(w[1][j], H)
    #             total_NumH /= 10
    #             total_NumH /= n
    #             numerator += total_NumH
    #         for i in range(m - n + 1):
    #             if w[0][i:i + n] == list(g):
    #                 denominator += 1
    #         if denominator == 0 or numerator == 0:
    #             return 0
    #         return numerator / denominator
    #
    #     def R_rec(n, S, D, H):
    #         # stemD = [stemmer.stem(d) for d in D[0]]
    #         # stemS = [stemmer.stem(s) for s in S]
    #         n_gram_D = list(ngrams(D[0], n))
    #         count_n_gram_D = Counter(n_gram_D)
    #         n_gram_S = list(ngrams(S, n))
    #         count_n_gram_S = Counter(n_gram_S)
    #
    #         n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))
    #         numerator = 0
    #         for g in n_gram_DnS:
    #             # numerator += 1 * min(count_n_gram_D[g], count_n_gram_S[g])
    #             numerator += beta(n, g, D, H) * min(count_n_gram_D[g], count_n_gram_S[g])
    #         denominator = 0
    #         for g in set(n_gram_D):
    #             denominator += beta(n, g, D, H) * count_n_gram_D[g]
    #             # denominator += 1 * count_n_gram_D[g]
    #         return numerator / max(denominator, 1)
    #
    #     def R_prec(n, S, D, H):
    #         # stemD = [stemmer.stem(d) for d in D[0]]
    #         # stemS = [stemmer.stem(s) for s in S]
    #         n_gram_D = list(ngrams(D[0], n))
    #         count_n_gram_D = Counter(n_gram_D)
    #         n_gram_S = list(ngrams(S, n))
    #         count_n_gram_S = Counter(n_gram_S)
    #         n_gram_DnS = set(n_numHgram_S).intersection(set(n_gram_D))
    #         numerator = 0
    #         for g in n_gram_DnS:
    #             # numerator += 1 * min(count_n_gram_D[g], count_n_gram_S[g])
    #             numerator += beta(n, g, D, H) * min(count_n_gram_D[g], count_n_gram_S[g])
    #         denominator = 0
    #         for g in set(n_gram_S):
    #             # denominator += beta(n, g, D, H) * count_n_gram_S[g]
    #             denominator += 1 * count_n_gram_S[g]
    #         return numerator / max(denominator, 1)

    def init_dec_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = state['encoder_states']
        batch_size = states.size(0)
        length = states.size(1)
        state['input_feed'] = states.new_zeros((batch_size, 1, states.size(2)))
        state['coverage'] = states.new_zeros((batch_size, length, 1))  # (B, L, 1)
        state['hidden_context'] = state['input_feed'].new_zeros(state['input_feed'].size())
        return state

    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor],
                             salience_values: torch.Tensor,
                             source_ids: Dict[str, torch.Tensor],
                             source_text: List[List[str]],
                             raw_constraints: List) -> Dict[str, List]:
        """Make forward pass during prediction using a beam_search search."""
        state = self.init_dec_state(state)
        state['source_ids'] = source_ids['ids']
        state['max_oov'] = source_ids['max_oov']
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self.start_idx)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        n_system_idx = self.beam_search.search(
            start_predictions, state, self.take_step, raw_constraints)
        n_system_summaries = []
        for system_idx in n_system_idx:
            system_summaries = []
            for batch_idx, predict_idx in enumerate(system_idx):
                predict_tokens = []
                for idx in predict_idx:
                    if idx < self.vocab.get_vocab_size():
                        token = self.vocab.get_token_from_index(idx)
                    else:
                        real_id = (source_ids['ids'][batch_idx, :] == idx).nonzero().squeeze()
                        if len(real_id.size()) != 0:
                            real_id = real_id[0].item()
                        else:
                            real_id = real_id.item()
                        token = source_text[batch_idx][real_id]
                    if type(token) != str:
                        token = token.text
                    predict_tokens.append(token)
                system_summaries.append(predict_tokens)
            n_system_summaries.append(system_summaries)
        if self.is_result_rouge_ranked:
            scores = {}
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            max_score = {batch: -1 for batch in range(len(n_system_summaries[0]))}
            best_summary = {batch: '' for batch in range(len(n_system_summaries[0]))}
            for system_summaries in n_system_summaries:
                for batch, summary in enumerate(system_summaries):
                    if batch not in scores.keys():
                        scores[batch] = []
                    source = [token.text for token in source_text[batch]]
                    score = scorer.score(' '.join(source), ' '.join(summary))['rougeL'].fmeasure
                    scores[batch].append((score, summary))
                    if score > max_score[batch]:
                        max_score[batch] = score
                        best_summary[batch] = summary

        else:
            for system_summaries in n_system_summaries:
                pass
        output_dict = {
            "predictions": [summ for batch, summ in best_summary.items()]
        }
        return output_dict



    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor],
                  is_first_step: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam_search search class.

        Parameters
        ----------
        is_first_step : ``bool``
            Denoting that it is a first step
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``community intelligence is needed to tackle the migrant boat threat along the south coast , a national crime agency -lrb- nca -rrb- chief has said .
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
        tokens = last_predictions
        tokens[tokens >= self.vocab.get_vocab_size()] = self.unk_idx
        emb = self.target_embedder({
            'tokens': tokens
        })
        state = self.decoder(emb, state, self.is_coverage, self.training, is_first_step)
        state['class_probs'] = state['class_probs'] + 1e-9
        return -state['class_probs'].squeeze(1).log(), state

    def forward(self,
                source_tokens: Dict[str, torch.Tensor],
                source_text: List[List[str]],
                source_ids: Dict[str, torch.Tensor],
                raw_constraint: List[List[int]] = None,
                target_text: List[List[str]] = None,
                target_tokens: Dict[str, torch.Tensor] = None,
                target_ids: torch.Tensor = None,
                salience_values: torch.Tensor = None) \
            -> Dict[str, torch.Tensor]:
        """
        The forward function of the encoder and decoder model

        :param raw_constraint: Lexical constraint
        :param target_text: The raw text of target sequence
        :param source_ids: The source ids that is unique to the document
        :param source_text: The raw text of source sequence
        :param salience_values: The saliency values for source tokens
        :param source_tokens: Indexes of states tokens
        :param target_tokens: Indexes of target tokens
        :param target_ids: Similar with target_tokens but mapped with extended vocabulary
        :return: The loss and prediction of the model
        """
        state = self._encode(source_tokens, salience_values)
        output_dict = {}

        if target_tokens:
            state = self._decode(source_ids, target_tokens, state)
            if not self.use_copy_mechanism:
                target_ids = target_tokens['tokens']
            output_dict['loss'] = self._compute_loss(target_tokens, target_ids, state)

        if not self.training and not target_tokens:
            raw_constraints = []
            if raw_constraint:
                raw_constraints, word_constraints = \
                    self.update_constraint_idx(raw_constraint, source_text, source_ids)
                output_dict['word_constraints'] = word_constraints
            output_dict['results'] = self._forward_beam_search(state, salience_values, source_ids, source_text, raw_constraints)
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor], salience_values: torch.Tensor) \
            -> Dict[str, torch.Tensor]:
        """
        Encode the states tokens

        :param source_tokens: The indexes of states tokens
        :return: All the states and the last state
        """
        state = self.init_enc_state(source_tokens)
        embedded_src = self.source_embedder(source_tokens)
        emb_salience_feature = None
        if salience_values is not None and self.salience_source_mixer is not None:
            embedded_src, emb_salience_feature = self.salience_source_mixer(
                salience_values, embedded_src)
        # final_state = (last state, last context)
        states_features, states, final_state, final_context_state = \
            self.encoder(embedded_src, state['source_lengths'])
        batch_size = states.size(0)
        state['encoder_states'] = states  # (B, L, Num Direction * D_h)
        state['hidden'] = final_state.view(
            batch_size,
            final_state.size(0),
            final_state.size(2))  # (B, L, Num Direction * D_h)
        state['context'] = final_context_state.view(
            batch_size,
            final_context_state.size(0),
            final_context_state.size(2))  # (B, L, Num Direction * D_h)
        state['states_features'] = states_features  # (B, L, Num Direction * D_h)
        if emb_salience_feature is not None:
            state['emb_salience_feature'] = emb_salience_feature  # (B, L, EmbDim)
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
        all_pgens = []
        # Teacher Forcing
        if torch.rand(1).item() <= self.teacher_force_ratio:
            embedded_tgt = self.target_embedder(target_tokens)[:, :-1, :]
            for step, emb in enumerate(embedded_tgt.split(1, dim=1)):
                if step == 0:
                    is_first_step = True
                else:
                    is_first_step = False
                state = self.decoder(emb, state, self.is_coverage, self.training, is_first_step)
                all_class_probs.append(state['class_probs'])
                all_coverages.append(state['coverage'])
                all_attentions.append(state['attention'])
                all_pgens.append(state['p_gen'])
        else:
            tokens = state["encoder_states"].new_full(
                (state["encoder_states"].size(0),), fill_value=self.start_idx, dtype=torch.long)
            emb = self.target_embedder({'tokens': tokens})
            for step in range(min(target_tokens['tokens'].size(1) - 1, self.max_steps - 2)):
                if step == 0:
                    is_first_step = True
                else:
                    is_first_step = False
                state = self.decoder(emb, state, self.is_coverage, self.training, is_first_step)
                all_class_probs.append(state['class_probs'])
                all_coverages.append(state['coverage'])
                all_attentions.append(state['attention'])
                all_pgens.append(state['p_gen'])
                class_prob = all_class_probs[-1].squeeze(1)
                prob_dist = Categorical(class_prob)
                tokens = prob_dist.sample()
                # _, tokens = class_prob.topk(1)
                tokens[tokens >= self.vocab.get_vocab_size()] = self.unk_idx
                emb = self.target_embedder({'tokens': tokens})
            # print(predicted_tokens)
        state['all_class_probs'] = torch.cat(all_class_probs, dim=1)
        state['all_coverages'] = torch.cat(all_coverages[:-1], dim=2)
        state['all_attentions'] = torch.cat(all_attentions, dim=2)
        self.p_gen = torch.cat(all_pgens, dim=1).mean().item()
        state.pop('class_probs', None)
        state.pop('coverage', None)
        state.pop('attention', None)
        return state

    def _compute_loss(self,
                      target_tokens: Dict[str, torch.Tensor],
                      target_ids: torch.Tensor,
                      state: Dict[str, torch.Tensor]):
        # (B, L, V)
        all_class_probs = state['all_class_probs']
        attentions = state['all_attentions']
        target_mask = util.get_text_field_mask(target_tokens)[:, :-1]
        target = target_ids[:, 1:]
        assert target_mask.size(1) == target.size(1)
        # (B, L, 1)
        length = all_class_probs.size(1)
        step_losses = all_class_probs.new_zeros((all_class_probs.size(0),))
        target_mask_t = target_mask.transpose(0, 1).contiguous()
        coverages = state['all_coverages']
        coverage_losses = all_class_probs.new_zeros((all_class_probs.size(0),))

        for i in range(length):
            gold_probs = torch.gather(all_class_probs[:, i, :], 1, target[:, i].unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + 1e-7)
            if self.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attentions[:, :, i], coverages[:, :, i]), 1)
                step_loss = step_loss + self.coverage_lambda * step_coverage_loss
                step_coverage_loss = step_coverage_loss * target_mask_t[i]
                coverage_losses += step_coverage_loss
            step_loss = step_loss * target_mask_t[i]
            step_losses += step_loss

        if self.is_coverage:
            batch_coverage_loss = coverage_losses / \
                                  util.get_lengths_from_binary_sequence_mask(target_mask)
            total_coverage_loss = torch.mean(batch_coverage_loss)
            self.coverage_loss = total_coverage_loss.item()
        batch_avg_loss = step_losses / util.get_lengths_from_binary_sequence_mask(target_mask)
        total_loss = torch.mean(batch_avg_loss)
        return total_loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            'coverage_loss': self.coverage_loss,
            'p_gen': self.p_gen
        }
        if reset:
            self.coverage_loss = 0.0
            self.p_gen = 0.0
        return metrics_to_return
