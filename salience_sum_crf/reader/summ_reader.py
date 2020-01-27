from typing import Optional, Iterable, List

import numpy as np
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField, ArrayField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from scipy.interpolate import CubicSpline

from pointer_generator_salience.reader.copy_field import CopyField
from pointer_generator_salience.reader.target_copy_field import TargetCopyField


@DatasetReader.register("summdatareader_salience")
class SummDataReader(DatasetReader):

    def __init__(self,
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 interpolation: bool = False,
                 predict: bool = True,
                 use_salience: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._predict = predict
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._interpolation = interpolation
        self._use_salience = use_salience

    def process_line(self, line):
        src_seq, tgt_seq = line.split('\t')
        return_res = []
        if self._use_salience:
            collection_seq = list(zip(*[group.split(u'￨') for group in src_seq.split()]))
            src_seq = collection_seq[0]
            salience_seqs = [[float(value) for value in seq] for seq in collection_seq[1:]]
            return_res.append([Token(token) for token in src_seq])
            if not self._predict:
                return_res.append([Token(token) for token in tgt_seq.split()])
            return_res.append(salience_seqs)
        else:
            return_res.append([Token(token) for token in src_seq.split()])
            if not self._predict:
                return_res.append([Token(token) for token in tgt_seq.split()])
        return return_res

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if line == '':
                    continue
                yield self.text_to_instance(*self.process_line(line))

    def smooth_and_norm(self, value):
        value_dict = {i: x for i, x in enumerate(value) if x != 0}
        try:
            cs = CubicSpline(list(value_dict.keys()), list(value_dict.values()))
        except:
            return value
        c = [float(cs(i)) if i in value_dict.keys() else cs(i) * cs(i) for i in range(len(value))]
        return c

    @staticmethod
    def text_to_ids(tokens):
        ids = {}
        out = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)+1))
        return out

    def text_to_instance(self, src_seq: List[Token], tgt_seq: List[Token] = None,
                         salience_seq: Iterable[float] = None) -> Instance:
        indexer = SingleIdTokenIndexer(lowercase_tokens=True)

        tokenized_src = src_seq[:self._source_max_tokens]

        source_field = TextField(tokenized_src, {'tokens': indexer})
        source_text_field = MetadataField(metadata=tokenized_src)
        source_ids_field = CopyField(tokenized_src)

            # ArrayField(np.array(self.text_to_ids(tokenized_src)))
        output_field = {
            'source_tokens': source_field,
            'source_text': source_text_field,
            'source_ids': source_ids_field,
        }
        if tgt_seq:
            tokenized_tgt = [Token(START_SYMBOL)] + \
                            tgt_seq[:self._target_max_tokens] + \
                            [Token(END_SYMBOL)]
            # Source and target are sharing vocabulary
            target_field = TextField(tokenized_tgt, {'tokens': indexer})
            target_ids_field = TargetCopyField(tokenized_src,
                                               tokenized_tgt,
                                               self._target_max_tokens)
            target_text_field = MetadataField(metadata=tokenized_tgt)
            output_field['target_tokens'] = target_field
            output_field['target_ids'] = target_ids_field
            output_field['target_text'] = target_text_field
        if salience_seq:
            if self._interpolation:
                salience_field = ArrayField(
                    np.array(self.smooth_and_norm(salience_seq)[:self._source_max_tokens]),
                    padding_value=-1
                )
            else:
                salience_seq = [seq for seq in zip(*salience_seq)][:self._source_max_tokens]
                salience_field = ArrayField(
                    np.array(salience_seq[:self._source_max_tokens]), padding_value=-1)
            output_field['salience_values'] = salience_field
        return Instance(output_field)
