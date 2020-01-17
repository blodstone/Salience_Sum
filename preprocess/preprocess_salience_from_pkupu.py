import argparse
import os
import pickle
from collections import defaultdict

import spacy
import pandas as pd

from noisy_salience_model import AKE
from noisy_salience_model import NER

nlp = spacy.load("en_core_web_sm", disable=["textcat", 'parser', 'tagger', 'entity_linker'])


def gen_doc_sum(document_path, summary_path, list_name):
    # document_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc-tokenized-segmented-final/restbody'
    # summary_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc-tokenized-segmented-final/firstsentence'
    for name in list_name:
        doc_f = name
        summ_f = '{}.fs'.format(name.split('.')[0])
        with open(os.path.join(document_path, doc_f)) as doc:
            with open(os.path.join(summary_path, summ_f)) as summ:
                yield doc.readlines(), doc_f, summ.readlines(), summ_f


def retrieve(doc_path, summ_path):
    contents = []
    doc_files = [(name, int(name.split('.')[0])) for name in os.listdir(doc_path)]
    doc_files.sort(key=lambda x: x[1])
    summ_files = [(name, int(name.split('.')[0])) for name in os.listdir(summ_path)]
    summ_files.sort(key=lambda x: x[1])
    for doc, summ in zip(doc_files, summ_files):
        doc_file = doc[0]
        summ_file = summ[0]
        if doc_file != summ_file:
            print('Error')
            break
        else:
            doc = open(os.path.join(doc_path, doc_file)).readlines()[0]
            summ_content = list(open(os.path.join(summ_path, summ_file)).readlines())
            if len(summ_content) == 0:
                print('Empty summary on file: {} of {}'.format(summ_file, summ_path))
                summ = ''
            else:
                summ = summ_content[0]
            content = '{}\t{}'.format(doc, summ)
        contents.append(content)
    return contents


def process_gold_result(gold_result):
    df_scores = []
    for score in gold_result:
        df_score = pd.DataFrame(score)
        df_score = df_score / df_score.sum()
        df_scores.append(df_score)
    return df_scores


def process_results(results):
    df_docs = defaultdict(list)
    df_scores = defaultdict(list)
    for model, docs in results.items():
        for idx, doc in enumerate(docs):
            df_docs[idx].append(doc)
    for doc_idx, scores in df_docs.items():
        # Normalize the scores
        df_score = pd.DataFrame(scores).sum(axis=0)
        df_score = df_score / df_score.sum()
        df_scores[doc_idx] = df_score
    return df_scores


def run(summ_pair, summ_groups, summs_path, dataset, index, max_words):
    doc, doc_f, gold, gold_f = summ_pair
    # Every token start with zero salience
    result_labels = [0 for sent in doc for _ in sent.split()]
    gold = [word.strip().lower() for word in gold[0].split()]
    tokens = [word.strip().lower() for sent in doc for word in sent.split()]
    # Then iterate each summary group to add salience points
    for summ_group in summ_groups:
        if summ_group in ['submodular', 'textrank', 'centroid']:
            summ_content = list(open(os.path.join(
                summs_path, dataset, summ_group, index[dataset][doc_f])).readlines())
            # Some system produce no summary, because the document has only one sentence
            if len(summ_content) != 0:
                for summ_sent in summ_content:
                    for i, doc_sent in enumerate(doc):
                        if doc_sent.replace(' ', '').replace('\n', '').lower() \
                                == summ_sent.replace(' ', '').replace('\n', ''):
                            idx = 0
                            for j, sent in enumerate(doc):
                                for _ in sent.split():
                                    if i == j:
                                        result_labels[idx] += 1
                                    idx += 1
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
        new_docs.append('{}u"￨"{}'.format(token, value))
    print(f'{gold_f} and {doc_f}')
    return '{}\t{}\n'.format(' '.join(new_docs), gold[0])


def main():
    set_name = args.set
    docs_path = args.docs_pku
    golds_path = args.golds_pku
    output_path = args.output
    summs_path = args.summs_pku
    max_words = args.max_words
    index = {}
    for name in set_name:
        index[name] = pickle.load(open(os.path.join(args.index, '{}_final_idx'.format(name)), 'rb'))
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
        doc_summ_pair = gen_doc_sum(docs_path, golds_path, list(index[dataset].keys()))
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
