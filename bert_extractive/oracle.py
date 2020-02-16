from pathlib import Path

import statistics
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

nlp = spacy.load("en_trf_bertbaseuncased_lg", disable=["textcat", 'parser', 'entity_linker', 'tagger'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


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
    collection_seq = list(zip(*[group.split(u'￨') for group in src_seq.split()]))
    src_seq = ' '.join(collection_seq[0])
    doc = segmenter(src_seq)
    tgt = nlp(tgt_seq)
    # Assuming 1 sentence gold summary for each document (BBC only).
    # For CNN/DM, it will need to be rewrote
    if is_using_gpu:
        t_tensor = from_dlpack(tgt.tensor.toDlpack())
    else:
        t_tensor = torch.tensor(tgt.tensor)
    doc_sims = []
    for sent in doc.sents:
        # We re-parse each sentence to limit the context to the sentence.
        src = nlp(sent.text)
        if is_using_gpu:
            src_tensor = from_dlpack(src.tensor.toDlpack())
        else:
            src_tensor = torch.tensor(src.tensor)
        result_tensor = torch.zeros(src_tensor.size(0), t_tensor.size(0))
        for i, s in enumerate(src_tensor.split(1, 0)):
            for j, t in enumerate(t_tensor.split(1, 0)):
                result_tensor[i, j] = s.squeeze().dot(t.squeeze())
        recall = result_tensor.max(dim=0)[0].sum() / t_tensor.size(0)
        precision = result_tensor.max(dim=1)[0].sum() / src_tensor.size(0)
        f1 = 2 * (recall * precision) / (recall + precision)
        doc_sims.append(f1.item())
    sent_idx = [(i, j, len(list(doc.sents)[i])) for i, j in enumerate(doc_sims)]
    sent_idx.sort(key=lambda x: x[1], reverse=True)
    oracle_summ = ''
    for i, _, l in sent_idx:
        if l <= 100:
            oracle_summ = list(doc.sents)[i].text
            break
    return oracle_summ


def main():
    input_path = Path(args.input)
    output_path = Path(args.output)
    lines = input_path.open('r').readlines()
    oracle_summs = read_text(lines)
    (output_path / 'oracle.txt').write_text('\n'.join(oracle_summs))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-input', help='The input text file (tsv).')
    parse.add_argument('-output', help='The output path.')
    args = parse.parse_args()
    main()