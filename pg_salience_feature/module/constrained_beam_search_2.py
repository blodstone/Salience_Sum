import copy
import math

from collections import OrderedDict
from typing import List, Callable, Tuple, Dict
import warnings

import torch

from allennlp.common.checks import ConfigurationError

StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[
    [torch.Tensor, StateType, bool], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


def count_score(tracker, score):
    return math.floor(sum([score[item] for item, value in tracker.items() if value]))


def allocate_beam(scores, topk_index, bin_size, beam_size, bid):
    bin_idx = 0
    allocate_bin = 0
    allocate_index = []
    active_beam = 1
    for item, score in scores:
        if active_beam <= beam_size and score == allocate_bin:
            allocate_index.append(item)
            active_beam += 1
        if bin_idx + 1 == bin_size:
            bin_idx = 0
            allocate_bin += 1
        bin_idx += 1
    i = 0
    while active_beam <= beam_size:
        item = topk_index[i]
        index = hash((bid, item))
        if index not in allocate_index:
            allocate_index.append(index)
            active_beam += 1
        i += 1
    return allocate_index


def init_global_tracker(batch_size, constraints):
    # multi_parents = []
    trackers = {}
    scores_hash = {}
    scores = []
    num_constraints = {}
    for i, batch in enumerate(constraints):
        multi = dict()
        tracker = {}
        a_score = {}
        num_constraint = 0
        for c in batch:
            if isinstance(c, list):
                tracker.update({e: False for e in c})
                a_score.update({e: 1 / len(c) for e in c})
                num_constraint += 1
                for j in range(len(c))[1:]:
                    multi[c[j]] = c[j - 1]
            else:
                if c not in tracker:
                    a_score.update({c: 1})
                    tracker.update({c: False})
                    num_constraint += 1
        num_constraints[hash(tuple(tracker.keys()))] = num_constraint
        scores.append(a_score)
        scores_hash[hash(tuple(tracker.keys()))] = a_score
        trackers[hash(i)] = tracker
        # multi_parents.append(multi)
    hash_to_idx = {hash(i): i for i in range(batch_size)}
    return hash_to_idx, num_constraints, scores, scores_hash, trackers


class ConstrainedBeamSearch:
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam used.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, as it can introduce
        more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType,
               constraints) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
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
        batch_size = start_predictions.size()[0]

        # Boilerplate for constraint codes
        hash_to_idx, num_constraints, scores, scores_hash, trackers = init_global_tracker(batch_size, constraints)
        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []
        hashes: List[torch.Tensor] = []
        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(start_predictions, start_state, True)

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(f"Target vocab size ({num_classes:d}) too small "
                                     f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                                     f"Please decrease beam_size or per_node_beam_size.")

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        # start_top_log_probabilities, start_predicted_classes = \
        #         start_class_log_probabilities.topk(self.beam_size)
        start_top_log_probabilities = []
        start_predicted_classes = []
        start_predicted_hashes = []
        candidates = {}

        predictions_idx_to_hash = {}
        iter_items = list(hash_to_idx.items())
        for bid, batch in iter_items:
            beam = start_class_log_probabilities[batch]
            bin_size = math.floor(self.beam_size / num_constraints[hash(tuple(trackers[bid].keys()))]) or 1
            # (1) The best k tokens for a single batch
            topk_log_prob, topk_index = beam.topk(self.beam_size)

            # (2) Unmet constraint
            select_index = torch.tensor([c for c in trackers[bid].keys() if not trackers[bid][c]],
                                        dtype=torch.long, device=start_class_log_probabilities.device)
            select_index = torch.cat([select_index, topk_index.squeeze()]).unique()
            topk_index = topk_index.tolist()
            # (3) Expand and update trackers
            new_tracker_candidates = {}
            new_tracker_scores = {}
            for i, idx in enumerate(select_index.tolist()):
                hash_to_idx[hash((bid, idx))] = idx
                new_tracker_candidates[hash((bid, idx))] = copy.deepcopy(trackers[bid])
                if idx in trackers[bid]:
                    new_tracker_candidates[hash((bid, idx))][idx] = True
                    new_tracker_scores[hash((bid, idx))] = \
                        count_score(new_tracker_candidates[hash((bid, idx))], scores[batch])
            new_tracker_scores = sorted(new_tracker_scores.items(), key=lambda item: item[1])

            # (4) Obtain selected hash index
            start_hash_index = allocate_beam(new_tracker_scores, topk_index, bin_size, self.beam_size, bid)
            # (5) Remove hash index that is not in used
            candidates.update({a_hash: new_tracker_candidates[a_hash] for a_hash in start_hash_index})
            predictions_idx_to_hash.update(
                {idx: hash for hash, idx in hash_to_idx.items() if hash in start_hash_index})
            start_predicted_class = torch.tensor(
                [hash_to_idx[a_hash] for a_hash in start_hash_index],
                dtype=torch.long, device=start_class_log_probabilities.device)
            start_predicted_hash = torch.tensor(
                [a_hash for a_hash in start_hash_index],
                dtype=torch.long, device=start_class_log_probabilities.device)
            start_top_log_probability = beam.gather(0, start_predicted_class)
            start_predicted_classes.append(start_predicted_class)
            start_predicted_hashes.append(start_predicted_hash)
            start_top_log_probabilities.append(start_top_log_probability)
        start_predicted_classes = torch.stack(start_predicted_classes, dim=0)
        start_predicted_hashes = torch.stack(start_predicted_hashes, dim=0)
        start_top_log_probabilities = torch.stack(start_top_log_probabilities, dim=0)
        trackers = candidates
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)
        hashes.append(start_predicted_hashes)
        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor. \
                unsqueeze(1). \
                expand(batch_size, self.beam_size, *last_dims). \
                reshape(batch_size * self.beam_size, *last_dims)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)
            last_hashes = hashes[-1].reshape(batch_size * self.beam_size)
            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state, False)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size,
                num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.per_node_beam_size)
            next_top_log_probabilities = []
            next_predicted_classes = []
            next_predicted_hashes = []
            for bid, beam, predicted_class in zip(
                    last_hashes.split(1, 0),
                    cleaned_log_probabilities.split(1, 0),
                    predicted_classes.split(1, 0)
            ):
                bid = bid.item()
                bin_size = math.floor(self.beam_size / num_constraints[hash(tuple(trackers[bid].keys()))]) or 1
                select_index = torch.tensor([c for c in trackers[bid].keys() if not trackers[bid][c]], dtype=torch.long,
                                            device=start_class_log_probabilities.device)
                select_index = torch.cat([select_index, predicted_class.squeeze()]).unique()
                new_tracker_candidates = {}
                new_tracker_scores = {}
                for i, idx in enumerate(select_index.tolist()):
                    hash_to_idx[hash((bid, idx))] = idx
                    new_tracker_candidates[hash((bid, idx))] = copy.deepcopy(trackers[bid])
                    if idx in trackers[bid]:
                        new_tracker_candidates[hash((bid, idx))][idx] = True
                        new_tracker_scores[hash((bid, idx))] = \
                            count_score(new_tracker_candidates[hash((bid, idx))],
                                        scores_hash[
                                            hash(tuple(new_tracker_candidates[hash((bid, idx))].keys()))])
                new_tracker_scores = sorted(new_tracker_scores.items(), key=lambda item: item[1])
                start_hash_index = allocate_beam(new_tracker_scores, predicted_class.squeeze().tolist(), bin_size,
                                                 self.beam_size, bid)
                candidates.update({a_hash: new_tracker_candidates[a_hash] for a_hash in start_hash_index})
                predictions_idx_to_hash.update(
                    {idx: hash for hash, idx in hash_to_idx.items() if hash in start_hash_index})
                next_predicted_class = torch.tensor(
                    [hash_to_idx[a_hash] for a_hash in start_hash_index],
                    dtype=torch.long, device=start_class_log_probabilities.device)
                next_predicted_hash = torch.tensor(
                    [a_hash for a_hash in start_hash_index],
                    dtype=torch.long, device=start_class_log_probabilities.device)
                next_top_log_probability = beam.squeeze().gather(0, next_predicted_class)
                next_predicted_classes.append(next_predicted_class)
                next_predicted_hashes.append(next_predicted_hash)
                next_top_log_probabilities.append(next_top_log_probability)
            predicted_classes = torch.stack(next_predicted_classes, dim=0)
            predicted_hashes = torch.stack(next_predicted_hashes, dim=0)
            top_log_probabilities = torch.stack(next_top_log_probabilities, dim=0)
            trackers = candidates
            # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities. \
                unsqueeze(2). \
                expand(batch_size, self.beam_size, self.per_node_beam_size). \
                reshape(batch_size * self.beam_size, self.per_node_beam_size)

            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities. \
                reshape(batch_size, self.beam_size * self.per_node_beam_size)

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes. \
                reshape(batch_size, self.beam_size * self.per_node_beam_size)
            reshaped_predicted_hashes = predicted_hashes. \
                reshape(batch_size, self.beam_size * self.per_node_beam_size)
            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)
            restricted_predicted_hashes = reshaped_predicted_hashes.gather(1, restricted_beam_indices)
            predictions.append(restricted_predicted_classes)
            hashes.append(restricted_predicted_hashes)
            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices / self.per_node_beam_size

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer. \
                    view(batch_size, self.beam_size, *([1] * len(last_dims))). \
                    expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor. \
                    reshape(batch_size, self.beam_size, *last_dims). \
                    gather(1, expanded_backpointer). \
                    reshape(batch_size * self.beam_size, *last_dims)

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities
