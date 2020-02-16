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
segmenter.tokenizer = WhitespaceTokenizer(segmenter.vocab)

nlp = spacy.load("en_trf_bertbaseuncased_lg", disable=["textcat", 'parser', 'entity_linker', 'tagger', 'ner'])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def read_text(lines):
    oracle_summs = []
    for i, line in enumerate(lines):
        print(f'Process {i}')
        if line.strip() == '':
            continue
        oracle_summ = process_doc(line)
        oracle_summs.append(oracle_summ)
    return oracle_summs


def process_doc(line):
    src_seq, tgt_seq = line.split('\t')
    collection_seq = list(zip(*[group.split(u'￨') for group in src_seq.split()]))
    src_seq = ' '.join(collection_seq[0])
    salience_seqs = [[float(value) for value in seq] for seq in collection_seq[1:]]
    doc = segmenter(src_seq)
    sent_nlps = []
    t_tensors = []
    for sent in doc.sents:
        sent_nlp = nlp(sent.text)
        sent_nlps.append(sent_nlp)
        if is_using_gpu:
            sent_nlp_tensor = from_dlpack(sent_nlp.tensor.toDlpack())
        else:
            sent_nlp_tensor = torch.tensor(sent_nlp.tensor)
        t_tensor = torch.zeros(len(salience_seqs), sent_nlp_tensor.size(1))
        for idx, salience_model in enumerate(salience_seqs):
            model_tensor = torch.zeros(1, sent_nlp_tensor.size(1))
            for i, token in enumerate(sent):
                if salience_model[token.i] == 1.0:
                    model_tensor += sent_nlp_tensor[i, :] * salience_model[token.i]
            model_tensor = model_tensor / sum(salience_model)
            t_tensor[idx, :] = model_tensor
        t_tensors.append(t_tensor)
    doc_sims = []
    for idx, sent_nlp in enumerate(sent_nlps):
        t_tensor = t_tensors[idx]
        if is_using_gpu:
            src_tensor = from_dlpack(sent_nlp.tensor.toDlpack())
        else:
            src_tensor = torch.tensor(sent_nlp.tensor)
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
    summ = ''
    for i, _, l in sent_idx:
        if l <= 100:
            summ = list(doc.sents)[i].text
            break
    return summ


def main():
    input_path = Path(args.input)
    output_path = Path(args.output)
    lines = input_path.open('r').readlines()
    summs = read_text(lines)
    (output_path / 'salience_out.txt').write_text('\n'.join(summs))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-input', help='The input text file (tsv).')
    parse.add_argument('-output', help='The output path.')
    args = parse.parse_args()
    main()
