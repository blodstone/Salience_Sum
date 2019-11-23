import argparse
import os
import pickle
from collections import defaultdict
from itertools import product

import pandas as pd
import spacy
from allennlp.common import Tqdm

from noisy_salience_model import AKE, NER, pkusumsum, gold

nlp = spacy.load("en_core_web_lg", disable=["textcat", "parser"])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


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


def main(args):
    set_name = args.set
    docs_path = args.docs_pku
    golds_path = args.golds_pku
    output_path = args.output
    summs_path = args.summs_pku
    max_words = args.max_words
    summ_groups = []
    summaries = {}
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
        # Retrieve each document set and sort them
        doc_path = os.path.join(docs_path, dataset)
        doc_files = [(name, int(name.split('.')[0])) for name in os.listdir(doc_path)]
        doc_files.sort(key=lambda x: x[1])

        # Retrieve each gold set and sort them also
        gold_path = os.path.join(golds_path, dataset)
        gold_files = [(name, int(name.split('.')[0])) for name in os.listdir(gold_path)]
        gold_files.sort(key=lambda x: x[1])
        new_lines = []
        for doc_set, gold_set in Tqdm.tqdm(zip(doc_files, gold_files)):
            doc = open(os.path.join(doc_path, doc_set[0])).readlines()[0]
            gold = open(os.path.join(gold_path, gold_set[0])).readlines()[0]
            nlp_doc = nlp(doc)
            # Every token start with zero salience
            result_labels = [0 for _ in nlp_doc]
            # Then iterate each summary group to add salience points
            for summ_group in summ_groups:
                if summ_group in ['submodular', 'textrank', 'centroid']:
                    summ_content = list(open(os.path.join(summs_path, dataset, summ_group, doc_set[0])).readlines())
                    # Some system produce no summary, because the document has only one sentence
                    if len(summ_content) != 0:
                        summ = summ_content[0]
                        nlp_summ = nlp(summ.strip())
                        for doc_sent in nlp_doc.sents:
                            for summ_sent in nlp_summ.sents:
                                if doc_sent.similarity(summ_sent) > 0.9:
                                    for t in doc_sent:
                                        result_labels[t.i] += 1
                elif summ_group == 'AKE':
                    window = args.window
                    results = AKE.run(max_words, window, nlp_doc)
                    result_labels = [result_labels[i] + results[i] for i, v in enumerate(results)]
                elif summ_group == 'NER':
                    results = NER.run(max_words, nlp_doc)
                    result_labels = [result_labels[i] + results[i] for i, v in enumerate(results)]
            new_docs = []
            for token, value in zip(nlp_doc, result_labels):
                new_docs.append('{}|%|{}'.format(token.text, value))
            new_lines.append('{}\t{}\n'.format(' '.join(new_docs), gold))
        write_file = open(os.path.join(output_path, dataset + '.tsv.tagged'), 'w')
        write_file.writelines(new_lines)
        write_file.close()
    # src_path = args.src
    # if args.AKE:
    #     window = args.window
    #     results['AKE'] = AKE.run(max_words, window, src_path)
    # if args.NER:
    #     results['NER'] = NER.run(max_words, src_path)
    #
    # if args.submodular:
    #     tgt_path = args.centroid_tgt
    #     results['centroid'] = pkusumsum.run(src_path, tgt_path)
    # if args.submodular:
    #     tgt_path = args.textrank_tgt
    #     results['textrank'] = pkusumsum.run(src_path, tgt_path)
    # if args.gold:
    #     highlight_path = args.highlight
    #     doc_id_path = args.doc_id
    #     gold_result = gold.run(src_path, highlight_path, doc_id_path)
    #
    # df_scores = process_results(results)
    # df_gold_scores = process_gold_result(gold_result)
    # output_file = open('sample_data/df_scores.pickle', 'wb')
    # pickle.dump(df_scores, output_file)
    # output_file.close()
    # output_file = open('sample_data/df_gold_scores.pickle', 'wb')
    # pickle.dump(df_gold_scores, output_file)
    # output_file.close()


if __name__ == '__main__':
    # doc_folder = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs'
    # docs_name = ['train', 'val']
    # summs_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs/'
    # summs_group = [['submodular', 'textrank', 'centroid'], ['submodular_val', 'textrank_val', 'centroid_val']]
    # output_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen'
    # -set train val -docs_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/docs -golds_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/gold -summs_pku /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/PKUSUMSUM/summs -output /home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen --submodular --centroid --textrank
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', nargs='+', help='Input list of set names (the folder name has to match the same name).')
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
    main(args)
