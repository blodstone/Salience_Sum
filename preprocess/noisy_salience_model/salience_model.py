import os
from pathlib import Path
from typing import List, Dict, Any, Union, MutableMapping, Tuple

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

    def __init__(self, doc: Text, raw: Text, summ: Text):
        self.summ = summ
        self.doc = doc
        self.raw = raw
        self.index_map = self.build_map()
        self.salience_set = SalienceSet()
        self.doc_size = [len(line) for line in doc]

    def add_salience(self, name: str, salience: Salience):
        assert len(self.doc) == len(salience)
        for doc_line, sal_line in zip(self.doc, salience):
            assert len(doc_line) == len(sal_line)
        self.salience_set[name] = salience

    def build_map(self) -> Dict[int, Tuple[int, int]]:
        index_map = dict()
        idx = 0
        for i, line in enumerate(self.doc):
            for j, word in enumerate(line):
                index_map[idx] = (i, j)
                idx += 1
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

    def write_to_file(self, output_path: Path, extra_name: str):
        output_path.mkdir(parents=True, exist_ok=True)
        tsv_file = output_path / f'{self.dataset_name}.{extra_name}.tsv'
        src_file = output_path / f'{self.dataset_name}.{extra_name}.src.txt'
        tgt_file = output_path / f'{self.dataset_name}.{extra_name}.tgt.txt'
        if tsv_file.exists():
            print(f'{str(tsv_file)} exists. Deleting file.')
            os.remove(str(tsv_file))
        if src_file.exists():
            print(f'{str(src_file)} exists. Deleting file.')
            os.remove(str(src_file))
        if tgt_file.exists():
            print(f'{str(tgt_file)} exists. Deleting file.')
            os.remove(str(tgt_file))
        j = 1
        for doc_id, instance in self.dataset.items():
            salience_sets = []
            for summ_group, salience_set in instance.salience_set.salience_set.items():
                salience_sets.append(salience_set)
            saliences = list(zip(*salience_sets))
            output_line = []
            for line, salience_values in zip(instance.doc, saliences):
                for token in zip(line, *salience_values):
                    output_token = u'￨'.join([str(t) for t in token])
                    output_line.append(output_token)
            summ = [token for line in instance.summ for token in line]
            tsv_file.open('a').write(doc_id + '\t' + ' '.join(output_line) + '\t' + ' '.join(summ) + '\n')
            src_file.open('a').write(doc_id + '\t' + ' '.join(output_line) + '\n')
            tgt_file.open('a').write(doc_id + '\t' + ' '.join(summ) + '\n')
            print(f'Write to file ({j}): {doc_id}')
            j += 1
