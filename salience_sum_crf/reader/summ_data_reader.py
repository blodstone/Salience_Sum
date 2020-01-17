from typing import Optional, Iterable, List

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import softmax
from allennlp.data import DatasetReader, Tokenizer, Instance, Token
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("crfsummdatareader")
class SummDataReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 interpolation: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._interpolation = interpolation

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as file:
            for line in file:
                src_tagged_seq, tgt_seq = line.split('\t')
                src_seq, salience_seq = zip(*[group.split(u'￨') for group in src_tagged_seq.split()])
                assert len(src_seq) == len(salience_seq)
                yield self.text_to_instance(
                    [Token(token) for token in src_seq],
                    tgt_seq,
                    [value for value in salience_seq])

    def smooth_and_norm(self, value):
        value_dict = {i: x for i, x in enumerate(value) if x != 0}
        try:
            cs = CubicSpline(list(value_dict.keys()), list(value_dict.values()))
        except:
            return value
        c = [int(cs(i)) if i in value_dict.keys() else cs(i) * cs(i) for i in range(len(value))]
        return c

    def text_to_instance(self, src_seq: List[Token], tgt_seq: str, salience_seq: List[str]) -> Instance:
        indexer = SingleIdTokenIndexer(lowercase_tokens=True)
        tokenized_src = src_seq[:self._source_max_tokens]
        tag_indexer = SingleIdTokenIndexer(namespace='labels')
        sequence = TextField(tokenized_src, {'labels': tag_indexer})
        tokenized_tgt = self._tokenizer.tokenize(tgt_seq)[:self._target_max_tokens]
        source_field = TextField(tokenized_src, {'tokens': indexer})
        target_field = TextField(tokenized_tgt, {'tokens': indexer})
        saliency_field = SequenceLabelField(salience_seq[:self._source_max_tokens], sequence, 'labels')
        return Instance({
            'tokens': source_field,
            'summs': target_field,
            'tags': saliency_field
        })
