import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Tuple

import spacy
from spacy.tokens import Doc

from noisy_salience_model import AKE, NER, tfidf, summ
from noisy_salience_model.salience_model import Dataset, Instance, SalienceSet


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load("en_core_web_sm", disable=["textcat", 'parser', 'entity_linker', 'tagger'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def gen_salience_instance(document_path: Path, summary_path: Path, set_name: str, index: dict) -> Tuple[str, Instance]:
    for doc_f in document_path.iterdir():
        doc_id = doc_f.stem
        if index.get(doc_id, None) != set_name:
            continue
        summ_f = summary_path / f'{doc_id}.fs'
        doc_content = [[word.lower() for word in line.strip().split()] for line in doc_f.open().readlines()]
        raw_content = [[word for word in line.strip().split()] for line in doc_f.open().readlines()]
        summ_content = [[word.lower() for word in line.strip().split()] for line in summ_f.open().readlines()]
        salience = Instance(doc_content, raw_content, summ_content)
        yield doc_id, salience


def process(dataset, max_words):
    for doc_id, instance in dataset.dataset.items():
        salience_set = instance.salience_set
        new_salience = SalienceSet.init_salience_set(instance.doc_size)
        total_aggregates = []
        for name, salience in salience_set.salience_set.items():
            if name in ['tfidf', 'NER']:
                continue
            salience_sets = [salience]
            ner_idx = -1
            tfidf_idx = -1
            if 'NER' in salience_set.salience_set.keys():
                salience_sets.append(salience_set['NER'])
                ner_idx = len(salience_sets) - 1
            if 'tfidf' in salience_set.salience_set.keys():
                salience_sets.append(salience_set['tfidf'])
                tfidf_idx = len(salience_sets) - 1
            saliences = list(zip(*salience_sets))
            aggregates = []
            for i, salience_values in enumerate(saliences):
                for j, token in enumerate(zip(*salience_values)):
                    aggregate = token[0]
                    if ner_idx != -1:
                        aggregate += token[ner_idx]
                    if tfidf_idx != -1:
                        aggregate *= token[tfidf_idx]
                    aggregates.append((aggregate, i, j))
            aggregates = sorted(aggregates, key=lambda x: x[0], reverse=True)[:max_words]
            total_aggregates.append(aggregates)
        for aggregates in total_aggregates:
            for aggregate in aggregates:
                _, i, j = aggregate
                new_salience[i][j] += 1
        instance.salience_set['filter_aggregate'] = new_salience
    return dataset


def main():
    set_names = args.set
    docs_path = Path(args.docs_pku)
    golds_path = Path(args.golds_pku)
    output_path = Path(args.output)
    summs_path = Path(args.summs_pku)
    max_words = args.max_words
    modes = args.modes
    index = {doc_id: dataset for dataset, doc_ids in json.load(open(args.index)).items() for doc_id in doc_ids}
    summ_groups = []
    # The folder name has to match these names
    if args.submodular:
        summ_groups.append('submodular')
    if args.textrank:
        summ_groups.append('textrank')
    if args.centroid:
        summ_groups.append('centroid')
    if args.lexpagerank:
        summ_groups.append('lexpagerank')
    if args.AKE:
        summ_groups.append('AKE')
    if args.NER:
        summ_groups.append('NER')

    for dataset_name in set_names:
        dataset = Dataset(dataset_name)
        for doc_id, salience_instance in gen_salience_instance(
                docs_path, golds_path, dataset_name, index):
            salience = None
            for summ_group in summ_groups:
                if summ_group in ['submodular', 'textrank', 'centroid', 'lexpagerank']:
                    summ_path = summs_path / summ_group / f'{doc_id}.restbody'
                    if summ_path.exists():
                        salience = summ.process(salience_instance, summ_path)
                    else:
                        break
                if summ_group == 'AKE':
                    salience = AKE.process(salience_instance, args.window)
                if summ_group == 'NER':
                    salience = NER.process(salience_instance, nlp)
                if salience:
                    salience_instance.salience_set[summ_group] = salience
            print(f'Processed ({len(dataset)}): {doc_id}')
            dataset[doc_id] = salience_instance
        if args.tfidf:
            doc_word_count = dict()
            if args.doc_word_count and Path(args.doc_word_count).exists():
                doc_word_count = pickle.load(Path(args.doc_word_count).open('rb'))
            elif 'train' in set_names:
                i = 1
                for doc_id, salience_instance in gen_salience_instance(
                        docs_path, golds_path, 'train', index):
                    print(f'Building word count ({i}): {doc_id}')
                    i += 1
                    doc_word_count[doc_id] = Counter()
                    for line in salience_instance.doc:
                        doc_word_count[doc_id].update(line)
                pickle.dump(doc_word_count, (output_path / 'doc_word_count.pickle').open('wb'))
                print(f'Saving document word count.')
            else:
                raise Exception
            dataset = tfidf.process(dataset, doc_word_count)
        if 'filter' in modes:
            dataset = process(dataset, max_words)
        dataset.write_to_file(output_path, args.extra_name)


if __name__ == '__main__':
    # -set train val test -docs_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs -golds_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc-tokenized-segmented-final/firstsentence -summs_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs_small -output /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen --submodular --centroid --textrank --lexpagerank --tfidf -index /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc-tokenized-segmented-final/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json -train_size 5
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', nargs='+', help='Input list of set names (the folder name has to match the same name).')
    parser.add_argument('-index', help='Index for matching files.')
    parser.add_argument('-docs_pku', help='The path containing the document for each set.')
    parser.add_argument('-golds_pku', help='The path containing the gold summary for each set.')
    # We have a fixed set of summ folder name: submodular, centroid and textrank
    parser.add_argument('-summs_pku', help='Folder consisting of generated pkusumsum output for each set.')
    parser.add_argument('-output', help='Folder for generating the output.')
    parser.add_argument('--submodular', help='Submodular.', action='store_true')
    parser.add_argument('--centroid', help='Centroid.', action='store_true')
    parser.add_argument('--textrank', help='Textrank.', action='store_true')
    parser.add_argument('--lexpagerank', help='Lexpagerank.', action='store_true')
    parser.add_argument('--tfidf', help='TFIDF', action='store_true')
    parser.add_argument('--AKE', help='Automated Keyword Extraction (AKE) using textrank.',
                        action='store_true')
    parser.add_argument('-window', help='Window for Textrank, needed by AKE.',
                        default=5, type=int)
    parser.add_argument('-doc_word_count', help='doc_word_count file for tfidf.')
    parser.add_argument('--NER', help='Named Entity Recognition.', action='store_true')
    parser.add_argument('-max_words', help='Maximum words.', default=35, type=int)
    parser.add_argument('-extra_name', help='Additional name for the output file path.', default='tagged')
    parser.add_argument('-modes', nargs='+', help='Filter the salience to max words for each summ groups',
                        default=['all'])
    args = parser.parse_args()
    main()
