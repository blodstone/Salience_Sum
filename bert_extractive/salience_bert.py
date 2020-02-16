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
    sum_t_tensor = torch.zeros(1, 768)
    j = 0
    sent_nlps = []
    for sent in doc.sents:
        sent_nlp = nlp(sent.text)
        sent_nlps.append(sent_nlp)
        if is_using_gpu:
            sent_nlp_tensor = from_dlpack(sent_nlp.tensor.toDlpack())
        else:
            sent_nlp_tensor = torch.tensor(sent_nlp.tensor)
        for salience_model in enumerate(salience_seqs):
            token_j = j
            for i, token in enumerate(sent_nlp):
                if not token.is_stop:
                    sum_t_tensor += sent_nlp_tensor[i, :] * salience_model[j]
                    token_j += 1
        j = token_j
    doc_sims = []
    for sent_nlp in sent_nlps:
        if is_using_gpu:
            sent_nlp_tensor = from_dlpack(sent_nlp.tensor.toDlpack())
        else:
            sent_nlp_tensor = torch.tensor(sent_nlp.tensor)
        sent_sims = []
        for t in sent_nlp_tensor.split(1, 0):
            similarity = t.squeeze().dot(sum_t_tensor.squeeze())
            sent_sims.append(similarity.item())
        doc_sims.append(sent_sims)
    means = [statistics.mean(sims) for sims in doc_sims]
    sent_idx = [(i, j, len(list(doc.sents)[i])) for i, j in enumerate(means)]
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
