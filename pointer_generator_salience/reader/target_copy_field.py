from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Field, DataArray, Token, Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN

from pointer_generator_salience.reader.copy_field import CopyField


class TargetCopyField(Field[torch.Tensor]):

    def __init__(self,
                 source_tokens: List[Token],
                 target_tokens: List[Token],
                 max_tokens: int) -> None:
        self._source_tokens = source_tokens
        self._target_tokens = target_tokens
        self._max_tokens = max_tokens
        self._out: List[int] = []

    def index(self, vocab: Vocabulary):
        source_ids, _ = CopyField.generate_ids_out(vocab, self._source_tokens)
        for token in self._target_tokens:
            text = token.text.lower()
            if text in source_ids:
                self._out.append(source_ids[text])
            else:
                self._out.append(vocab.get_token_index(text))

    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": self._max_tokens}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        desired_length = padding_lengths["num_tokens"]
        padded_tokens = pad_sequence_to_length(self._out, desired_length)
        tensor = torch.LongTensor(padded_tokens)
        return tensor

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        return torch.stack(tensor_list)

    def empty_field(self) -> 'Field':
        return TargetCopyField([])
