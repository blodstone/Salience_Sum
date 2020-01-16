from typing import Optional, Iterable, List

import numpy as np
from allennlp.data import DatasetReader, Tokenizer, Instance, Token
from scipy.interpolate import CubicSpline
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register("salience_summdatareader")
class SummDataReader(DatasetReader):

    def __init__(self,
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 interpolation: bool = False,
                 use_salience: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._interpolation = interpolation
        self._use_salience = use_salience

    def process_line(self, line):
        src_tagged_seq, tgt_seq = line.split('\t')
        src_seq, salience_seq = zip(*[group.split('|%|') for group in src_tagged_seq.split()])
        assert len(src_seq) == len(salience_seq)
        if self._use_salience:
            return [Token(token) for token in src_seq], [Token(token) for token in tgt_seq.split()],\
                   [int(value) for value in salience_seq]
        else:
            return [Token(token) for token in src_seq], [Token(token) for token in tgt_seq.split()]

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as file:
            for line in file:
                yield self.text_to_instance(*self.process_line(line))

    def smooth_and_norm(self, value):
        value_dict = {i: x for i, x in enumerate(value) if x != 0}
        try:
            cs = CubicSpline(list(value_dict.keys()), list(value_dict.values()))
        except:
            return value
        c = [float(cs(i)) if i in value_dict.keys() else cs(i) * cs(i) for i in range(len(value))]
        return c

    def text_to_instance(self, src_seq: List[Token], tgt_seq: List[Token],
                         salience_seq: List[int] = None) -> Instance:
        indexer = SingleIdTokenIndexer(lowercase_tokens=True,
                                       start_tokens=['<s>'], end_tokens=['</s>'])
        tokenized_src = src_seq[:self._source_max_tokens]
        tokenized_tgt = tgt_seq[:self._target_max_tokens]
        source_field = TextField(tokenized_src, {'tokens': indexer})
        target_field = TextField(tokenized_tgt, {'tokens': indexer})
        if salience_seq:
            if self._interpolation:
                saliency_field = ArrayField(np.array(self.smooth_and_norm(salience_seq)[:self._source_max_tokens]))
            else:
                saliency_field = SequenceLabelField(
                    salience_seq[:self._source_max_tokens], source_field)
            return Instance({
                'source_tokens': source_field,
                'target_tokens': target_field,
                'salience_values': saliency_field
            })
        else:
            return Instance({
                'source_tokens': source_field,
                'target_tokens': target_field,
            })
