import copy
from operator import attrgetter
from typing import List, Set, Tuple

# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
import torch
import numpy as np

RawConstraintList = List[List[int]]

def get_bank_sizes(num_constraints: int,
                   beam_size: int,
                   candidate_counts: List[int]) -> List[int]:
    """
    Evenly distributes the beam_search across the banks, where each bank is a portion of the beam_search devoted
    to hypotheses having met the same number of constraints, 0..num_constraints.
    After the assignment, banks with more slots than candidates are adjusted.

    :param num_constraints: The number of constraints.
    :param beam_size: The beam_search size.
    :param candidate_counts: The empirical counts of number of candidates in each bank.
    :return: A distribution over banks.
    """

    num_banks = num_constraints + 1
    bank_size = beam_size // num_banks
    remainder = beam_size - bank_size * num_banks

    # Distribute any remainder to the end
    assigned = [bank_size for x in range(num_banks)]
    assigned[-1] += remainder

    # Now, moving right to left, push extra allocation to earlier buckets.
    # This encodes a bias for higher buckets, but if no candidates are found, space
    # will be made in lower buckets. This may not be the best strategy, but it is important
    # that you start pushing from the bucket that is assigned the remainder, for cases where
    # num_constraints >= beam_size.
    for i in reversed(range(num_banks)):
        overfill = assigned[i] - candidate_counts[i]
        if overfill > 0:
            assigned[i] -= overfill
            assigned[(i - 1) % num_banks] += overfill

    return assigned

class ConstrainedHypothesis:
    """
    Represents a set of words and phrases that must appear in the output.
    A constraint is of two types: sequence or non-sequence.
    A non-sequence constraint is a single word and can therefore be followed by anything,
    whereas a sequence constraint must be followed by a particular word (the next word in the sequence).
    This class also records which constraints have been met.

    A list of raw constraints is maintained internally as two parallel arrays. The following raw constraint
    represents two phrases that must appear in the output: 14 and 19 35 14.

        raw constraint: [[14], [19, 35, 14]]

    This is represented internally as:

        constraints: [14 19 35 14]
        is_sequence: [False True True False]

    That is, the constraints are simply concatenated, and we maintain a parallel array indicating whether each
    token ID must be followed by the next token ID. The same token ID can be present any number of times.

    :param constraint_list: A list of zero or raw constraints (each represented as a list of integers).
    :param eos_id: The end-of-sentence ID.
    """

    def __init__(self,
                 constraint_list: RawConstraintList,
                 eos_id: int) -> None:

        # `constraints` records the words of the constraints, as a list (duplicates allowed).
        # `is_sequence` is a parallel array that records, for each corresponding constraint,
        #    whether the current word is the non-final word of a phrasal constraint.
        self.constraints = []  # type: List[int]
        self.is_sequence = []  # type: List[bool]
        for phrase in constraint_list:
            self.constraints += phrase
            self.is_sequence += [True] * len(phrase)
            self.is_sequence[-1] = False

        self.eos_id = eos_id

        # no constraints have been met
        self.met = [False for x in self.constraints]
        self.last_met = -1

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.constraints)

    def __str__(self) -> str:
        s = []
        for i, word_id in enumerate(self.constraints):
            s.append(str(word_id) if self.met[i] is False else 'X')
            if self.is_sequence[i]:
                s.append('->')
        return ' '.join(s)

    def size(self) -> int:
        """
        :return: the number of constraints
        """
        return len(self.constraints)

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        return sum(self.met)

    def num_needed(self) -> int:
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def allowed(self) -> Set[int]:
        """
        Returns the set of constrained words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is the list of all unmet constraints.
        If all constraints are met, an empty set is returned.

        :return: The ID of the next required word, or -1 if any word can follow
        """
        items = set()  # type: Set[int]
        # Add extensions of a started-but-incomplete sequential constraint
        if self.last_met != -1 and self.is_sequence[self.last_met] == 1:
            word_id = self.constraints[self.last_met + 1]
            if word_id != self.eos_id or self.num_needed() == 1:
                items.add(word_id)

        # Add all constraints that aren't non-initial sequences
        else:
            for i, word_id in enumerate(self.constraints):
                if not self.met[i] and (i == 0 or not self.is_sequence[i - 1]):
                    # The self.num_needed() is checked in case the 'eos' is part of the constraint
                    if word_id != self.eos_id or self.num_needed() == 1:
                        items.add(word_id)

        return items

    def finished(self) -> bool:
        """
        Return true if all the constraints have been met.

        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def is_valid(self, wordid) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return True
        # return self.finished() or wordid != self.eos_id or (self.num_needed() == 1 and self.eos_id in self.allowed())

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        """
        Updates the constraints object based on advancing on word_id.
        There is a complication, in that we may have started but not
        yet completed a multi-word constraint.  We need to allow constraints
        to be added as unconstrained words, so if the next word is
        invalid, we must "back out" of the current (incomplete) phrase,
        re-setting all of its words as unmet.

        :param word_id: The word ID to advance on.
        :return: A deep copy of the object, advanced on word_id.
        """

        obj = copy.deepcopy(self)

        # First, check if we're updating a sequential constraint.
        if obj.last_met != -1 and obj.is_sequence[obj.last_met] == 1:
            if word_id == obj.constraints[obj.last_met + 1]:
                # Here, the word matches what we expect next in the constraint, so we update everything
                obj.met[obj.last_met + 1] = True
                obj.last_met += 1
            else:
                # Here, the word is not the expected next word of the constraint, so we back out of the constraint.
                index = obj.last_met
                while obj.is_sequence[index]:
                    obj.met[index] = False
                    index -= 1
                obj.last_met = -1

        # If not, check whether we're meeting a single-word constraint
        else:
            # Build a list from all constraints of tuples of the
            # form (constraint, whether it's a non-initial sequential, whether it's been met)
            # is.sequence is shifted to the right one place because we want to check the previous token
            constraint_tuples = list(zip(obj.constraints, [False] + obj.is_sequence[:-1], obj.met))
            # We are searching for an unmet constraint (word_id) that is not the middle of a phrase and is not met
            query = (word_id, False, False)
            try:
                pos = constraint_tuples.index(query)
                obj.met[pos] = True
                obj.last_met = pos
            except ValueError:
                # query not found; identical but duplicated object will be returned
                pass

        return obj


class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam_search in topk().

    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: ConstrainedHypothesis) -> None:
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


def topk(timestep: int,
         batch_size: int,
         beam_size: int,
         inactive: torch.Tensor,
         scores: torch.Tensor,
         hypotheses: List[ConstrainedHypothesis],
         best_ids: torch.Tensor,
         best_word_ids: torch.Tensor,
         seq_scores: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, List[ConstrainedHypothesis], torch.Tensor]:
    """
    Builds a new topk list such that the beam_search contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam_search for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (batch_size if t==1 else beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects.
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param seq_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        if hypotheses[rows.start] is not None and hypotheses[rows.start].size() > 0:
            best_ids[rows], best_word_ids[rows], seq_scores[rows], \
            hypotheses[rows], inactive[rows] = _sequential_topk(timestep,
                                                                beam_size,
                                                                inactive[rows],
                                                                scores[rows],
                                                                hypotheses[rows],
                                                                best_ids[rows] - rows.start,
                                                                best_word_ids[rows],
                                                                seq_scores[rows])

            # offsetting since the returned smallest_k() indices were slice-relative
            best_ids[rows] += rows.start
        else:
            # If there are no constraints for this sentence in the batch, everything stays
            # the same, except we need to mark all hypotheses as active
            inactive[rows] = 0

    return best_ids, best_word_ids, seq_scores, hypotheses, inactive


def _sequential_topk(timestep: int,
                     beam_size: int,
                     inactive: torch.Tensor,
                     scores: torch.Tensor,
                     hypotheses: List[ConstrainedHypothesis],
                     best_ids: torch.Tensor,
                     best_word_ids: torch.Tensor,
                     sequence_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                             List[ConstrainedHypothesis], torch.Tensor]:
    """
    Builds a new topk list such that the beam_search contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param beam_size: The length of the beam_search for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects.
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    num_constraints = hypotheses[0].size()

    candidates = set()
    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row = int(row.item())
        col = int(col.item())
        if hypotheses[row] is not None and hypotheses[row].is_valid(col):
            seq_score = float(seq_score.item())
            new_item = hypotheses[row].advance(col)
            cand = ConstrainedCandidate(row, col, seq_score, new_item)
            candidates.add(cand)

    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = torch.argmin(scores, dim=1)
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.allowed()

        # (3) add the single-best item after this (if it's valid)
        col = int(best_next[row].item())
        if hyp.is_valid(col):
            nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            new_item = hyp.advance(col)
            score = scores[row, col].item()
            if score != float('Inf'):
                cand = ConstrainedCandidate(row, col, score, new_item)
                candidates.add(cand)

    # Sort the candidates. After allocating the beam_search across the banks, we will pick the top items
    # for each bank from this list
    sorted_candidates = sorted(candidates, key=attrgetter('score'))

    # The number of hypotheses in each bank
    counts = [0 for _ in range(num_constraints + 1)]
    for cand in sorted_candidates:
        counts[cand.hypothesis.num_met()] += 1

    # Adjust allocated bank sizes if there are too few candidates in any of them
    bank_sizes = get_bank_sizes(num_constraints, beam_size, counts)

    # Sort the candidates into the allocated banks
    pruned_candidates = []  # type: List[ConstrainedCandidate]
    left_over_candidates = []
    for i, cand in enumerate(sorted_candidates):
        bank = cand.hypothesis.num_met()

        if bank_sizes[bank] > 0:
            pruned_candidates.append(cand)
            bank_sizes[bank] -= 1
        else:
            left_over_candidates.append(cand)
    if len(left_over_candidates) > 0:
        needed_cand = int(beam_size * 0.1) - len(
            [cand for cand in pruned_candidates if cand.score <= left_over_candidates[0].score])
        if needed_cand <= len(left_over_candidates):
            reserve_top_k = needed_cand
        else:
            reserve_top_k = len(left_over_candidates)
        i = 0
        while reserve_top_k > 0:
            pruned_candidates[-reserve_top_k] = left_over_candidates[i]
            reserve_top_k -= 1
            i += 1
    pruned_candidates = sorted(pruned_candidates, key=attrgetter('score'))
    num_pruned_candidates = len(pruned_candidates)

    inactive[:num_pruned_candidates] = 0

    # Pad the beam_search so array assignment still works
    if num_pruned_candidates < beam_size:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (beam_size - num_pruned_candidates)
    return (torch.tensor([x.row for x in pruned_candidates]),
            torch.tensor([x.col for x in pruned_candidates]),
            torch.tensor([[x.score] for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            inactive)
