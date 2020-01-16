from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Field, DataArray, Token, Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN


class CopyField(Field[torch.Tensor]):

    def __init__(self,
                 source_tokens: List[Token]) -> None:
        self._source_tokens = source_tokens
        self._out: List[int] = []

    def index(self, vocab: Vocabulary):
        vocab_size = vocab.get_vocab_size()
        ids = {}
        for token in self._source_tokens:
            text = token.text.lower()
            text_ids = vocab.get_token_index(text)
            if text_ids == vocab.get_token_index(DEFAULT_OOV_TOKEN):
                self._out.append(ids.setdefault(text, len(ids) + vocab_size))
            else:
                self._out.append(text_ids)

    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": len(self._source_tokens)}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        desired_length = padding_lengths["num_tokens"]
        padded_tokens = pad_sequence_to_length(self._out, desired_length)
        tensor = torch.LongTensor(padded_tokens)
        max_oov = tensor.max()
        return {
            'ids': tensor,
            'max_oov': max_oov
        }

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        ids = []
        max_oovs = []
        for t in tensor_list:
            ids.append(t['ids'])
            max_oovs.append(t['max_oov'])
        return {
            'ids': torch.stack(ids),
            'max_oov': torch.stack(max_oovs)
        }

    def empty_field(self) -> 'Field':
        return CopyField([])
