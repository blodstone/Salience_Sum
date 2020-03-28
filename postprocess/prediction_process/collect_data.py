
import argparse
import pickle
from multiprocessing.pool import Pool
from pathlib import Path

from rouge_score import rouge_scorer
from typing import Dict


class Data:

    def __init__(self, doc, ref, cand, salience):
        self.doc = doc
        self.ref = ref
        self.cand = cand
        self.scores = None
        self.salience = salience

    def calc_rouge_score(self, scorer):
        scores = scorer.score(' '.join(self.ref), ' '.join(self.cand))
        self.scores = {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure,
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure,
            },
            'rougeLsum': {
                'precision': scores['rougeLsum'].precision,
                'recall': scores['rougeLsum'].recall,
                'fmeasure': scores['rougeLsum'].fmeasure,
            }
        }


def rouge(name, data, output_folder):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    print(f'Calculating ROUGE score for {name} file.')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    for a_data in data:
        a_data.calc_rouge_score(scorer)
    pickle.dump(data, (output_folder / f'{name}.pickle').open('wb'))


def main():
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    src_f = Path(args.src)
    print('Reading source lines.')
    with src_f.open(encoding='utf-8') as src:
        src_lines = [line for line in src.readlines() if line.strip() != '']

    for f in input_folder.iterdir():
        print(f'Reading {f.stem}')
        data = []
        with f.open(encoding='utf-8') as cand:
            candidate_lines = [line for line in cand.readlines() if line.strip() != '']
            assert len(candidate_lines) == len(src_lines)
            for cand_line, src_line in zip(candidate_lines, src_lines):
                if cand_line.strip() == '' and src_line.strip() == '':
                    continue
                else:
                    cand_line = cand_line.strip()
                    src_line = src_line.strip()
                src_seq, tgt_seq = src_line.split('\t')
                collection = list(zip(*[group.split(u'￨') for group in src_seq.split()]))
                salience = collection[1:]
                src_seq = collection[0]
                data.append(Data(src_seq, tgt_seq.split(), cand_line.split(), salience))
        rouge(f.stem, data, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_folder', help='folder containing candidate files.', default='')
    parser.add_argument('-src', help='source tsv file with salience')
    parser.add_argument('-output_folder', help='folder for outputing the pickle files.', default='')
    args = parser.parse_args()
    main()

