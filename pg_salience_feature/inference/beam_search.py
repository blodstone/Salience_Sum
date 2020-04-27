from typing import List, Callable, Tuple, Dict, Optional
import warnings

import torch
import numpy as np
from torch.distributions import Categorical

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from pg_salience_feature.inference import lexical_constraints
from pg_salience_feature.inference.common_beam_search import CommonBeamSearch
from pg_salience_feature.inference.lexical_constraints import ConstrainedHypothesis, RawConstraintList
from pg_salience_feature.module.lexical_constraint_util import UpdateScores, SortByIndex, NormalizeAndUpdateFinished, \
    SortStateByIndex, PruneHypotheses

StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[
    [torch.Tensor, StateType, bool], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name
TokenIds = List[int]
BeamHistory = Dict[str, List]


class NBestTranslations:
    __slots__ = ('target_ids_list',
                 'attention_matrices',
                 'scores')

    def __init__(self,
                 target_ids_list: List[TokenIds],
                 attention_matrices: List[np.ndarray],
                 scores: List[float]) -> None:
        self.target_ids_list = target_ids_list
        self.attention_matrices = attention_matrices

        self.scores = scores


class Translation:
    __slots__ = ('target_ids',
                 'attention_matrix',
                 'score',
                 'beam_histories',
                 'nbest_translations',
                 'estimated_reference_length')

    def __init__(self,
                 target_ids: TokenIds,
                 score: float,
                 beam_histories: List[BeamHistory] = None,
                 nbest_translations: NBestTranslations = None,
                 estimated_reference_length: Optional[float] = None) -> None:
        self.target_ids = target_ids
        self.score = score
        self.beam_histories = beam_histories if beam_histories is not None else []
        self.nbest_translations = nbest_translations
        self.estimated_reference_length = estimated_reference_length


@CommonBeamSearch.register('beam_search')
class BeamSearch(CommonBeamSearch):
    """
    Implements the beam_search search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam_search used.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, as it can introduce
        more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(self,
                 top_p: float = 1.0,
                 beam_size: int = 10,
                 hyp_sampling: str = 'greedy',
                 per_node_beam_size: int = None) -> None:
        super().__init__(beam_size)
        # Pre-initialized value, must call init_beam to set the value
        self.hyp_sampling = hyp_sampling
        self.top_p = top_p
        self.end_index = 0
        self.unk_index = 0
        self.max_steps = 100
        self.update_finished = None
        self.is_beam_configured = False

        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.update_scores = UpdateScores()
        self.sort_by_index = SortByIndex()
        self.sort_state_by_index = SortStateByIndex()
        self.prune_hyp = PruneHypotheses(threshold=20, beam_size=self.beam_size)

    def config_beam(self, end_index: int, unk_index: int, max_steps: int):
        self.end_index = end_index
        self.unk_index = unk_index
        self.max_steps = max_steps
        self.update_finished = NormalizeAndUpdateFinished(0, self.end_index)
        self.is_beam_configured = True

    def unravel_index(self, index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def sample_topk2(self, scores, target_dists, finished):
        # Map the negative logprobs to probabilities so as to have a distribution
        target_dists = torch.exp(-target_dists)
        # Sample from the target distributions over words, then get the corresponding values from the cumulative scores
        best_word_indices = target_dists.multinomial(1)
        # Zeroes for finished hypotheses.
        best_word_indices = torch.where(finished.type(torch.bool).unsqueeze(1), torch.zeros_like(best_word_indices), best_word_indices)
        values = scores.take(best_word_indices)
        best_hyp_indices = torch.arange(0, target_dists.shape[0], dtype=torch.int32, device=target_dists.device)

        return best_hyp_indices, best_word_indices.squeeze(), values

    def sample_topk(self,
                    scores: torch.Tensor,
                    offset: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Choose an extension of each hypothesis from its softmax distribution.

            :param scores: Vocabulary scores for the next beam_search step. (batch_size * beam_size, target_vocabulary_size)
            :param target_dists: The non-cumulative target distributions (ignored).
            :param finished: The list of finished hypotheses.
            :param best_hyp_indices: Best hypothesis indices constant.
            :return: The row indices, column indices, and values of the sampled words.
            """
        k = self.beam_size
        batch_size = int(offset.shape[-1] / k)
        folded_scores = scores.reshape((batch_size, -1))
        # Map the negative logprobs to probabilities so as to have a distribution
        folded_probs = torch.exp(-folded_scores)
        norm_folded_probs = (folded_probs + 1e-9) / (folded_probs + 1e-9).sum(dim=-1).unsqueeze(1)
        indices = norm_folded_probs.multinomial(k)
        values = folded_scores.gather(-1, indices)
        indices = indices.type(torch.int32).reshape((-1,))
        values = values.reshape((-1, 1))
        best_hyp_indices, best_word_indices = self.unravel_index(indices, shape=(batch_size * k, scores.shape[-1]))
        if batch_size > 1:
            # Offsetting the indices to match the shape of the scores matrix
            best_hyp_indices += offset
        return best_hyp_indices, best_word_indices, values

    def topk(self, scores: torch.Tensor,
             offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the lowest k elements per sentence from a `scores` matrix.
        At the first timestep, the shape of scores is (batch, target_vocabulary_size).
        At subsequent steps, the shape is (batch * k, target_vocabulary_size).

        :param scores: Vocabulary scores for the next beam_search step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array (shape: batch_size * k) containing offsets to add to the hypothesis indices in batch decoding.
        :param k: The number of smallest scores to return.
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """

        # Compute the batch size from the offsets and k. We don't know the batch size because it is
        # either 1 (at timestep 1) or k (at timesteps 2+).
        # (batch_size, beam_size * target_vocab_size)
        k = self.beam_size
        batch_size = int(offset.shape[-1] / k)
        folded_scores = scores.reshape((batch_size, -1))

        # pylint: disable=unbalanced-tuple-unpacking
        values, indices = torch.topk(folded_scores, dim=1, k=k, largest=False)
        indices = indices.type(torch.int32).reshape((-1,))
        best_hyp_indices, best_word_indices = self.unravel_index(indices, shape=(batch_size * k, scores.shape[-1]))

        if batch_size > 1:
            # Offsetting the indices to match the shape of the scores matrix
            best_hyp_indices += offset

        values = values.reshape((-1, 1))
        return best_hyp_indices, best_word_indices, values

    def init_batch(self,
                   raw_constraints: List[Optional[RawConstraintList]],
                   beam_size: int,
                   start_id: int,
                   eos_id: int) -> List[Optional[ConstrainedHypothesis]]:
        """
        :param raw_constraints: The list of raw constraints (list of list of IDs).
        :param beam_size: The beam_search size.
        :param start_id: The target-language vocabulary ID of the SOS symbol.
        :param eos_id: The target-language vocabulary ID of the EOS symbol.
        :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
        """
        constraints = [None] * (len(raw_constraints) * beam_size)  # type: List[Optional[ConstrainedHypothesis]]
        if any(raw_constraints):
            for i, raw_list in enumerate(raw_constraints):
                num_constraints = sum([len(phrase) for phrase in raw_list]) if raw_list is not None else 0
                if num_constraints > 0:
                    hyp = ConstrainedHypothesis(raw_list, eos_id)
                    idx = i * beam_size
                    constraints[idx:idx + beam_size] = [hyp.advance(start_id) for x in range(beam_size)]

        return constraints

    def _assemble_translation(self, sequence: np.ndarray,
                              length: np.ndarray,
                              seq_score: np.ndarray) -> Translation:
        """
        Takes a set of data pertaining to a single translated item, performs slightly different
        processing on each, and merges it into a Translation object.
        :param sequence: Array of word ids. Shape: (batch_size, bucket_key).
        :param length: The length of the translated segment.
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch_size * self.beam_size, max_output_length, encoded_source_length).
        :param seq_score: Array of length-normalized negative log-probs.
        :param estimated_reference_length: Estimated reference length (if any).
        :param beam_history: The optional beam_search histories for each sentence in the batch.
        :return: A Translation object.
        """
        length = int(length)
        sequence = sequence[:length].tolist()
        score = float(seq_score)
        return Translation(sequence, score,
                           nbest_translations=None)

    def get_best_word_indices_for_kth_hypotheses(self, ks: np.ndarray, all_hyp_indices: np.ndarray) -> np.ndarray:
        """
        Traverses the matrix of best hypotheses indices collected during beam_search search in reversed order by
        using the kth hypotheses index as a backpointer.
        Returns an array containing the indices into the best_word_indices collected during beam_search search to extract
        the kth hypotheses.

        :param ks: The kth-best hypotheses to extract. Supports multiple for batch_size > 1. Shape: (batch,).
        :param all_hyp_indices: All best hypotheses indices list collected in beam_search search. Shape: (batch * beam_search, steps).
        :return: Array of indices into the best_word_indices collected in beam_search search
            that extract the kth-best hypothesis. Shape: (batch,).
        """
        batch_size = ks.shape[0]
        num_steps = all_hyp_indices.shape[1]
        result = np.zeros((batch_size, num_steps - 1), dtype=all_hyp_indices.dtype)
        # first index into the history of the desired hypotheses.
        pointer = all_hyp_indices[ks, -1]
        # for each column/step follow the pointer, starting from the penultimate column/step
        num_steps = all_hyp_indices.shape[1]
        for step in range(num_steps - 2, -1, -1):
            result[:, step] = pointer
            pointer = all_hyp_indices[pointer, step]
        return result

    def search(self,
               start_predictions: torch.Tensor,
               state: StateType,
               step: StepFunctionType,
               raw_constraint_list=None) -> List:
        """
        Given a starting state and a step function, apply beam_search search to find the
        most likely target sequences.

        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam_search size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from ``search``
        and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``StateType``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        """
        if not self.is_beam_configured:
            raise ConfigurationError('Beam is not configured through init_beam yet.')
        batch_size = start_predictions.shape[0]
        start_index = start_predictions[0].item()
        device = start_predictions.device

        # Expanding to (batch*beam_search) for all states
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor. \
                unsqueeze(1). \
                expand(batch_size, self.beam_size, *last_dims). \
                reshape(batch_size * self.beam_size, *last_dims)

        # (batch_size, ) the offset denoting starting batch index, repeated for batch
        batch_indices = torch.arange(0, batch_size * self.beam_size, self.beam_size, dtype=torch.long, device=device)
        # (batch_size*beam_size, )
        best_word_indices = start_predictions.unsqueeze(1) \
            .expand((batch_size, self.beam_size)).reshape((batch_size * self.beam_size,))

        constraints = self.init_batch(raw_constraint_list, self.beam_size,
                                      start_index, self.end_index)

        # (batch_size*beam_search, ) the offset denoting starting batch index, repeated for batch*beam_search
        offset = torch.arange(0, batch_size * self.beam_size,
                              self.beam_size, dtype=torch.int32, device=device) \
            .repeat_interleave(self.beam_size)

        # For forcing the first step to share the same ancestor
        first_step_mask = torch.full((batch_size * self.beam_size, 1),
                                     fill_value=float('Inf'), device=device)
        first_step_mask[batch_indices] = 1.0

        max_output_lengths = torch.full((batch_size * self.beam_size,),
                                        fill_value=self.max_steps, device=device)
        best_hyp_indices_list = []
        best_word_indices_list = []

        lengths = torch.zeros((batch_size * self.beam_size, 1), device=device)
        finished = torch.zeros((batch_size * self.beam_size,), device=device, dtype=torch.int32)

        is_pad_dist_set = False
        scores_accumulated = torch.zeros((batch_size * self.beam_size, 1), device=device)

        inactive = torch.zeros((batch_size * self.beam_size), dtype=torch.int32, device=device)
        for t in range(1, self.max_steps):
            # shape: (batch_size, num_classes)
            target_dists, state = step(best_word_indices.type(torch.long), state, True)

            if not is_pad_dist_set:
                pad_dist = torch.full((batch_size * self.beam_size, target_dists.shape[-1] - 1),
                                      fill_value=float('Inf'), device=device)
                is_pad_dist_set = True

            scores = self.update_scores.forward(target_dists, finished,
                                                inactive, scores_accumulated, pad_dist, 1.0)

            if self.hyp_sampling == 'sample':
                best_hyp_indices, best_word_indices, scores_accumulated = \
                    self.sample_topk2(scores, target_dists, finished)
            elif self.hyp_sampling == 'greedy':
                if t == 1:
                    scores *= first_step_mask
                best_hyp_indices, best_word_indices, scores_accumulated = self.topk(scores, offset)
            scores = self.update_scores.forward(target_dists, finished,
                                                inactive, scores_accumulated, pad_dist, self.top_p)
            # scores = self.update_scores.forward(target_dists, finished,
            #                                     inactive, saved_scores_accumulated, pad_dist, self.top_p/2)
            if any(raw_constraint_list):
                best_hyp_indices, best_word_indices, scores_accumulated, constraints, inactive = \
                    lexical_constraints.topk(
                        t,
                        batch_size,
                        self.beam_size,
                        inactive,
                        scores,
                        constraints,
                        best_hyp_indices,
                        best_word_indices,
                        scores_accumulated)
            finished = self.sort_by_index.forward(best_hyp_indices, finished)[0]
            state = self.sort_state_by_index.forward(best_hyp_indices, state)
            finished, scores_accumulated, lengths = self.update_finished.forward(best_word_indices,
                                                                                 max_output_lengths,
                                                                                 finished,
                                                                                 scores_accumulated,
                                                                                 lengths)

            inactive, best_word_indices, scores_accumulated = self.prune_hyp.forward(best_word_indices,
                                                                                     scores_accumulated,
                                                                                     finished)
            # Replace unknown with attention
            attention = state['attention']
            best_word_indices = best_word_indices.reshape((batch_size, -1))
            _, top_idx = attention.reshape((batch_size, -1, attention.shape[1], attention.shape[2])).topk(1, dim=2)
            for i in range(batch_size):
                for j in range(self.beam_size):
                    if best_word_indices[i][j] == self.unk_index:
                        best_word_indices[i][j] = top_idx.squeeze()[i][j]
            best_word_indices = best_word_indices.reshape((-1, ))

            best_hyp_indices_list.append(best_hyp_indices)
            best_word_indices_list.append(best_word_indices)
            if finished.sum().item() == batch_size * self.beam_size:
                break
        folded_accumulated_scores = scores_accumulated.reshape((batch_size,
                                                                self.beam_size * scores_accumulated.shape[-1]))
        indices = torch.argsort(folded_accumulated_scores, dim=1).type(torch.int32).reshape((-1,))
        best_hyp_indices, _ = self.unravel_index(indices, scores_accumulated.shape)
        best_hyp_indices = best_hyp_indices + offset
        best_hyp_indices_list.append(best_hyp_indices)
        lengths = lengths[best_hyp_indices.type(torch.long)]
        scores_accumulated = torch.take(scores_accumulated, best_hyp_indices.type(torch.long))
        if len(constraints) > 0:
            constraints = [constraints[x] for x in best_hyp_indices.cpu().numpy()]

        best_hyp_indices = torch.stack(best_hyp_indices_list, dim=1).cpu().numpy()
        best_word_indices = torch.stack(best_word_indices_list, dim=1).cpu().numpy()
        batch_size = best_hyp_indices.shape[0] // self.beam_size

        nbest_translations = []  # type: List[List[Translation]]
        for n in range(0, self.beam_size):
            # Initialize the best_ids to the first item in each batch, plus current nbest index
            best_ids = np.arange(n, batch_size * self.beam_size, self.beam_size, dtype='int32')

            # only check for constraints for 1-best translation for each sequence in batch
            if n == 0 and any(constraints):
                # For constrained decoding, select from items that have met all constraints (might not be finished)
                unmet = np.array([c.num_needed() if c is not None else 0 for c in constraints])
                filtered = np.where(unmet == 0, scores_accumulated.cpu().numpy().flatten(), np.inf)
                filtered = filtered.reshape((batch_size, self.beam_size))
                best_ids += np.argmin(filtered, axis=1).astype('int32')

            # Obtain sequences for all best hypotheses in the batch
            indices = self.get_best_word_indices_for_kth_hypotheses(best_ids, best_hyp_indices)  # type: np.ndarray
            # pylint: disable=unsubscriptable-object
            nbest_translations.append(
                [self._assemble_translation(*x) for x in zip(best_word_indices[indices, np.arange(indices.shape[1])],
                                                             lengths[best_ids],
                                                             scores_accumulated[best_ids], )])
        n_system_idx = []
        for best_translations in nbest_translations:
            system_idx = []
            for batch_idx, translation in enumerate(best_translations):
                predict_idx = []
                for idx in translation.target_ids:
                    if type(idx) != int:
                        idx = idx.item()
                    predict_idx.append(idx)
                system_idx.append(predict_idx)
            n_system_idx.append(system_idx)
        return n_system_idx


