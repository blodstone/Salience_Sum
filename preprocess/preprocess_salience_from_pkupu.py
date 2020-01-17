import argparse
import json
import os
from pathlib import Path

import spacy
from suffixtree import *
from typing import List, Tuple

from noisy_salience_model import AKE
from noisy_salience_model import NER
from difflib import SequenceMatcher

nlp = spacy.load("en_core_web_sm", disable=["textcat", 'parser', 'tagger', 'entity_linker'])


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def gen_doc_sum(document_path, summary_path, set_name):
    document_path = Path(document_path, set_name)
    summary_path = Path(summary_path, set_name)
    for doc_f in document_path.iterdir():
        summ_f = summary_path / doc_f.name
        yield doc_f.read_text().strip(), doc_f, summ_f.read_text().strip(), summ_f


def find_overlap(summ: List[str], doc: List[str]) -> Tuple[int, int]:
    l_summ = len(summ)
    first_summ_token = [i for i, token in enumerate(doc) if token == summ[0]]
    for i in first_summ_token:
        t_doc = ' '.join(doc[i:i+l_summ])
        if similar(t_doc, ' '.join(summ)) > 0.9:
            return i, i+l_summ-1
    return 0, 0


def run(summ_pair, summ_groups, summs_path, dataset, index, max_words):
    doc, doc_f, gold, gold_f = summ_pair
    # Every token start with zero salience
    result_labels = [0 for _ in doc.split()]
    tokens = [word.strip().lower() for word in doc.split()]
    # Then iterate each summary group to add salience points
    for summ_group in summ_groups:
        if summ_group in ['submodular', 'textrank', 'centroid']:
            summ_content = open(os.path.join(summs_path, dataset, summ_group, doc_f.name)).readlines()
            start_idx, end_idx = find_overlap(summ_content[0].split(), doc.split())
            result_labels = [val+1 if start_idx <= i <= end_idx else val for i, val in enumerate(result_labels)]
        elif summ_group == 'AKE':
            window = args.window
            results = AKE.run(max_words, window, doc)
            assert len(result_labels) == len(results)
            result_labels = [result_labels[i] + results[i] for i, v in enumerate(results)]
        elif summ_group == 'NER':
            results = NER.run(max_words, doc, nlp)
            assert len(result_labels) == len(results)
            result_labels = [result_labels[i] + results[i] for i, v in enumerate(results)]
    new_docs = []
    for token, value in zip(tokens, result_labels):
        new_docs.append(u'{}￨{}'.format(token, value))
    print(f'{gold_f.name} and {doc_f.name}')
    return '{}\t{}\n'.format(' '.join(new_docs), gold)


def main():
    set_name = args.set
    docs_path = args.docs_pku
    golds_path = args.golds_pku
    output_path = args.output
    summs_path = args.summs_pku
    max_words = args.max_words
    index = json.load(open(args.index))
    summ_groups = []
    # The folder name has to match these names
    if args.submodular:
        summ_groups.append('submodular')
    if args.textrank:
        summ_groups.append('textrank')
    if args.centroid:
        summ_groups.append('centroid')
    if args.AKE:
        summ_groups.append('AKE')
    if args.NER:
        summ_groups.append('NER')

    for dataset in set_name:
        print('Processing Dataset {}'.format(dataset))
        doc_summ_pair = gen_doc_sum(docs_path, golds_path, dataset)
        new_lines = [run(summ_pair, summ_groups, summs_path, dataset, index, max_words) for summ_pair in doc_summ_pair]
        write_file = open(os.path.join(output_path, dataset + '.tsv.tagged'), 'w')
        write_file.writelines(new_lines)
        write_file.close()


if __name__ == '__main__':
    # -set train val -docs_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs -golds_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/gold -summs_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs -output /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen --submodular --centroid --textrank
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', nargs='+', help='Input list of set names (the folder name has to match the same name).')
    parser.add_argument('-index', help='Index for matching files.')
    parser.add_argument('-docs_pku', help='The path containing the document for each set.')
    parser.add_argument('-golds_pku', help='The path containing the gold summary for each set.')
    # We have a fixed set of summ folder name: submodular, centroid and textrank
    parser.add_argument('-summs_pku', help='Folder consisting of generated pkusumsum output for each set.')
    parser.add_argument('-output', help='Folder for generating the output.')
    # parser.add_argument('-src', help='Source document (pickle format).')
    parser.add_argument('--submodular', help='Submodular.',
                        action='store_true')
    parser.add_argument('-submodular_tgt', help='Preprocessed submodular summ in advance.')
    parser.add_argument('--centroid', help='Centroid.',
                        action='store_true')
    parser.add_argument('-centroid_tgt', help='Preprocessed centroid summ in advance.')
    parser.add_argument('--textrank', help='Textrank.',
                        action='store_true')
    parser.add_argument('-textrank_tgt', help='Preprocessed textrank summ in advance.')
    parser.add_argument('--compression', help='Compression.',
                        action='store_true')
    parser.add_argument('--AKE', help='Automated Keyword Extraction (AKE) using textrank.',
                        action='store_true')
    parser.add_argument('-window', help='Window for Textrank, needed by AKE.',
                        default=5, type=int)
    parser.add_argument('--NER', help='Named Entity Recognition.',
                        action='store_true')
    parser.add_argument('--gold', help='Gold annotations.', action='store_true')
    parser.add_argument('-highlight', help='Path to pandas highlight.')
    parser.add_argument('-doc_id', help='Path to doc_id.')
    parser.add_argument('-max_words', help='Maximum words.', default=35, type=int)
    args = parser.parse_args()
    main()
