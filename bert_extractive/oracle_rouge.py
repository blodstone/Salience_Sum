from pathlib import Path

import statistics
from rouge_score import rouge_scorer
import re
import spacy
from spacy.lang.en import English
import torch
import argparse
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch.nn as nn

from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


# Only works in GPU because BERT in spacy doesn't support CPU
is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
cos = nn.CosineSimilarity(dim=1)
# Initialize the spacy
segmenter = English()
sentencizer = segmenter.create_pipe("sentencizer")
segmenter.add_pipe(sentencizer)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def read_text(lines):
    salience_summs = []
    for i, line in enumerate(lines):
        print(f'Process {i}')
        if line.strip() == '':
            continue
        oracle_summ = process_doc(line)
        salience_summs.append(oracle_summ)
    return salience_summs


def process_doc(line):
    src_seq, tgt_seq = line.split('\t')
    if args.dataset == 'CNN':
        # Remove tag
        p = re.compile('(?<=<t>)(.*?)(?=</t>)')
        all = [sent.strip() for sent in p.findall(tgt_seq)]
        tgt_seq = ' '.join(all)
    doc = segmenter(src_seq)
    doc_sims = []
    for sent in doc.sents:
        score = scorer.score(tgt_seq, sent.text)
        doc_sims.append(score['rougeL'].fmeasure)
        # We re-parse
    oracle_summ = []

    sent_idx = [(i, j, len(list(doc.sents)[i]), list(doc.sents)[i].text) for i, j in enumerate(doc_sims)]
    sent_idx.sort(key=lambda x: x[1], reverse=True)
    len_sum = 0
    max_len = 100
    for i, _, l, text in sent_idx:
        if len_sum + l <= max_len:
            oracle_summ.append(text)
            len_sum += l
            if args.dataset == 'BBC':
                break
        else:
            break
    return ' '.join(oracle_summ)


def main():
    input_path = Path(args.input)
    output_path = Path(args.output)
    lines = input_path.open('r').readlines()
    oracle_summs = read_text(lines)
    (output_path / f'oracle-rouge-{args.dataset}.txt').write_text('\n'.join(oracle_summs))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-input', help='The input text file (tsv).')
    parse.add_argument('-output', help='The output path.')
    parse.add_argument('-dataset', help='BBC or CNN/DM')
    args = parse.parse_args()
    main()
