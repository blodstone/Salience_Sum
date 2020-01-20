from pathlib import Path
from typing import List, Dict, Any, Union, MutableMapping

from spacy.tokens import Doc

Salience = List[List[float]]
Text = List[Union[List[str], Doc]]


class SalienceSet(Dict[str, Salience]):

    def __init__(self):
        super().__init__()
        self.salience_set = dict()

    def __setitem__(self, name: str, salience: Salience):
        self.salience_set[name] = salience

    def __getitem__(self, name: str) -> Salience:
        return self.salience_set[name]

    def __len__(self) -> int:
        return len(self.salience_set)

    def __iter__(self):
        return iter(self.salience_set)

    @staticmethod
    def init_salience_set(doc_size: List[int]) -> List[List[float]]:
        salience = []
        for length in doc_size:
            salience.append([0.0 for _ in range(length)])
        return salience


class Instance:

    def __init__(self, doc: Text, raw: Text, summ: Text, nlp_doc: Text = None
                 ):
        self.summ = summ
        self.doc = doc
        self.raw = raw
        self.index_map = self.build_map()
        self.nlp_doc = nlp_doc
        self.salience_set = SalienceSet()
        self.doc_size = [len(line) for line in doc]

    def add_salience(self, name: str, salience: Salience):
        assert len(self.doc) == len(salience)
        for doc_line, sal_line in zip(self.doc, salience):
            assert len(doc_line) == len(sal_line)
        self.salience_set[name] = salience

    def build_map(self):
        index_map = dict()
        idx = 0
        for i, line in enumerate(self.doc):
            for j, word in enumerate(line):
                for _ in word:
                    index_map[idx] = (i, j)
                    idx += 1
                index_map[idx] = (-1, -1)
                idx += 1
            idx -= 1
        del index_map[idx]
        assert len(index_map) == len(''.join([' '.join([word for word in line]) for line in self.doc]))
        return index_map


class Dataset(MutableMapping[str, Instance]):

    def __delitem__(self, v: str) -> None:
        pass

    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset = dict()

    def __setitem__(self, doc_id: str, salience_instance: Instance):
        self.dataset[doc_id] = salience_instance

    def __getitem__(self, doc_id: str) -> Instance:
        return self.dataset[doc_id]

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def write_to_file(self, output_path: Path, output_format: str, extra_name: str = ''):
        output_docs = []
        output_summs = []
        for _, instance in self.dataset.items():
            summ_groups = []
            salience_sets = []
            for summ_group, salience_set in instance.salience_set.salience_set.items():
                summ_groups.append(summ_group)
                salience_sets.append(salience_set)
            saliences = list(zip(*salience_sets))
            output_line = []
            for line, salience_values in zip(instance.doc, saliences):
                for token in zip(line, *salience_values):
                    output_token = u'￨'.join([str(t) for t in token])
                    output_line.append(output_token)
            if output_format == 'allennlp':
                output_docs.append(' '.join(output_line) + '\t' + ' '.join(instance.summ))
            else:
                output_docs.append(' '.join(output_line))
            output_summs.append(' '.join([' '.join(tokens) for tokens in instance.summ]))
        output_docs = '\n'.join(output_docs)
        output_summs = '\n'.join(output_summs)
        if output_format == 'allennlp':
            tsv_file = output_path / f'{self.dataset_name}{extra_name}.tsv'
            tsv_file.write_text(output_docs)
        else:
            src_file = output_path / f'{self.dataset_name}{extra_name}.src.txt'
            src_file.write_text(output_docs)
            tgt_file = output_path / f'{self.dataset_name}{extra_name}.tgt.txt'
            tgt_file.write_text(output_summs)
        info_file = output_path / 'summ_groups.txt'
        info_file.write_text(u'￨'.join(summ_groups))
