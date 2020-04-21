from typing import Union, Optional

import torch
from torch.nn import Module


class UpdateScores(Module):

    def __init__(self):
        super(UpdateScores, self).__init__()

    def forward(self, target_dists, finished, inactive, scores_accumulated, pad_dist):
        scores = target_dists + scores_accumulated
        scores = torch.where(
            (finished.unsqueeze(1) | inactive.unsqueeze(1)).type(torch.ByteTensor),
            torch.cat((scores_accumulated, pad_dist), dim=1),
            scores)
        return scores


class SortStateByIndex(Module):
    """
    A HybridBlock that sorts args by the given indices.
    """

    def forward(self, indices, state):
        for key, state_tensor in state.items():
            state[key] = state[key][indices.type(torch.long)]
        return state


class SortByIndex(Module):
    """
    A HybridBlock that sorts args by the given indices.
    """

    def forward(self, indices, *args):
        return [torch.take(arg, indices.type(torch.long)) for arg in args]


class LengthPenalty(Module):
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs) -> None:
        super(LengthPenalty, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def forward(self, lengths, is_none=False):
        if self.alpha == 0.0:
            if is_none:
                return 1.0
            else:
                return torch.ones_like(lengths)
        else:
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator

    def get(self, lengths: Union[torch.Tensor, int, float]) -> Union[torch.Tensor, float]:
        """
        Calculate the length penalty for the given vector of lengths.

        :param lengths: A scalar or a matrix of sentence lengths of dimensionality (batch_size, 1).
        :return: The length penalty. A scalar or a matrix (batch_size, 1) depending on the input.
        """
        return self.forward(lengths, True)


class NormalizeAndUpdateFinished(Module):
    """
    A HybridBlock for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self, pad_id: int,
                 eos_id: int,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.length_penalty = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)

    def forward(self, best_word_indices, max_output_lengths,
                       finished, scores_accumulated, lengths):
        all_finished = ((best_word_indices == self.pad_id) | (best_word_indices == self.eos_id))
        newly_finished = all_finished.type(torch.int32) ^ finished
        scores_accumulated = torch.where(newly_finished.unsqueeze(1).type(torch.ByteTensor), scores_accumulated / self.length_penalty(lengths), scores_accumulated)

        # Update lengths of all items, except those that were already finished. This updates
        # the lengths for inactive items, too, but that doesn't matter since they are ignored anyway.
        lengths = lengths + (1 - finished.unsqueeze(1)).type(torch.float32)

        # Now, recompute finished. Hypotheses are finished if they are
        # - extended with <pad>, or
        # - extended with <eos>, or
        # - at their maximum length.
        finished = ((best_word_indices == self.pad_id) | (best_word_indices == self.eos_id)) | (lengths.reshape((-1,)).type(torch.int32) >= max_output_lengths)
        return finished.type(torch.int32), scores_accumulated, lengths
