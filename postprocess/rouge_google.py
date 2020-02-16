# -*- encoding: utf-8 -*-
import argparse
import os
import time
import shutil
import sys
import codecs
from pathlib import Path

from rouge_score import rouge_scorer


def get_mean(all_scores, rouge_type):
    scores = [scores[rouge_type] for scores in all_scores]
    mean_recall = sum([score.recall for score in scores]) / len(scores)
    mean_precision = sum([score.precision for score in scores]) / len(scores)
    mean_fmeasure = sum([score.fmeasure for score in scores]) / len(scores)
    return mean_precision, mean_recall, mean_fmeasure


def send_to_file(result):
    to_file = f'Precision,{result[0]}\nRecall,{result[1]}\nF1,{result[2]}'
    output = Path(args.o)



def rouge(cand, ref):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    candidates = [line.strip() for line in cand if line.strip() != '']
    references = [line.strip().split('\t')[1] for line in ref if line.strip() != '']
    assert len(candidates) == len(references), f'{len(candidates)}, {len(references)}'
    all_scores = []
    for c, r in zip(candidates, references):
        scores = scorer.score(r, c)
        all_scores.append(scores)
    results_1 = get_mean(all_scores, 'rouge1')
    results_2 = get_mean(all_scores, 'rouge2')
    results_l = get_mean(all_scores, 'rougeL')
    return f'{results_1}, {results_2}, {results_l}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='folder containing candidate files.')
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file (TXT format)')
    parser.add_argument('-r', type=str, default="reference.tsv",
                        help='reference file (TSV format)')
    parser.add_argument('-o', type=str, help='output path.')
    parser.add_argument('-n', type=str, help='output filename.')
    args = parser.parse_args()
    # if args.c.upper() == "STDIN":
    #     candidates = sys.stdin
    # else:
    #     candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")
    input_folder = Path(args.f)
    results = []
    for i in input_folder.iterdir():
        candidates = i.open().readlines()
        results.append(rouge(candidates, references))
    output_path = Path(args.o)
    (output_path / args.n).open('w').write('\n'.join(results))
