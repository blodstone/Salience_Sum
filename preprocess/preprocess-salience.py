import argparse
import pickle
from collections import defaultdict
import pandas as pd

from noisy_salience_model import AKE, NER, pkusumsum, gold


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
    max_words = args.max_words
    results = {}
    gold_result = []
    src_path = args.src
    if args.AKE:
        window = args.window
        results['AKE'] = AKE.run(max_words, window, src_path)
    if args.NER:
        results['NER'] = NER.run(max_words, src_path)
    if args.submodular:
        tgt_path = args.submodular_tgt
        results['submodular'] = pkusumsum.run(src_path, tgt_path)
    if args.submodular:
        tgt_path = args.centroid_tgt
        results['centroid'] = pkusumsum.run(src_path, tgt_path)
    if args.submodular:
        tgt_path = args.textrank_tgt
        results['textrank'] = pkusumsum.run(src_path, tgt_path)
    if args.gold:
        highlight_path = args.highlight
        doc_id_path = args.doc_id
        gold_result = gold.run(src_path, highlight_path, doc_id_path)

    df_scores = process_results(results)
    df_gold_scores = process_gold_result(gold_result)
    output_file = open('sample_data/df_scores.pickle', 'wb')
    pickle.dump(df_scores, output_file)
    output_file.close()
    output_file = open('sample_data/df_gold_scores.pickle', 'wb')
    pickle.dump(df_gold_scores, output_file)
    output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Source document (pickle format).')
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
