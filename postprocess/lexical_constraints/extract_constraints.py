import argparse
import spacy
from spacy.tokens import Doc
from pathlib import Path

from typing import List


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def subset_sum(a: List, k: float):
    k = int(k)
    if len(a) == 0:
        return False

    n = len(a)
    m = {}
    for i in range(n + 1):
        m[i] = {}
        for k in range(k + 1):
            m[i][k] = False

    for i in range(1, n + 1):
        for s in range(k + 1):
            if s - a[i - 1][1] >= 0:
                m[i][s] = m[i - 1][s] or a[i - 1][1] == s or m[i - 1][s - a[i - 1][1]]
            else:
                m[i][s] = m[i - 1][s] or a[i - 1][1] == s

    searched_set = []
    i = k
    j = len(a)
    while j > 0:
        if i == 0:
            break
        if a[j-1] != i:
            while m[j-1][i]:
                j -= 1
        searched_set.append(a[j-1])
        i = i - a[j-1][1]
        j -= 1
    return searched_set


def knapSack(W, wt, val, n):
    K = [[0 for w in range(W + 1)]
         for i in range(n + 1)]

    # Build table K[][] in bottom
    # up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]
                              + K[i - 1][w - wt[i - 1]],
                              K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

                # stores the result of Knapsack
    res = K[n][W]

    w = W
    items = []
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # either the result comes from the
        # top (K[i-1][w]) or from (val[i-1]
        # + K[i-1] [w-wt[i-1]]) as in Knapsack
        # table. If it comes from the latter
        # one/ it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:

            # This item is included.
            items.append(wt[i - 1])

            # Since this weight is included
            # its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]
    return items


def main():
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    k = args.k
    tsv = Path(args.tsv)
    updated_lines = []
    for line in tsv.open('r'):
        line = line.strip()
        if line == '':
            continue
        try:
            orig_src_seq, tgt_seq = line.split('\t')
            collection_seq = list(zip(*[group.split(u'￨') for group in orig_src_seq.split()]))
            src_seq = collection_seq[0]

            salience_seqs = [[float(value) if float(value) <= 1.0 else 1.0 for value in seq] for seq in
                             collection_seq[1:]]
            salience_seqs = [sum(saliences) for saliences in zip(*salience_seqs)]
            # (1) Extract noun chunks and calculate the average score of the chunks,
            # also remove stop words from noun chunks
            result = []
            used_idx = []
            doc = nlp(" ".join(src_seq))
            chunks = doc.noun_chunks
            for chunk in chunks:
                phrase = []
                score = 0
                for token in chunk:
                    if not token.is_stop:
                        phrase.append(token.i)
                        used_idx.append(token.i)
                        score += salience_seqs[token.i]
                if len(phrase) != 0:
                    score /= len(phrase)
                    result.append((phrase, score))
            # (2) Extract single word with stop words and noun chunks removed from text
            for token in doc:
                if not token.is_stop and not token.i in used_idx:
                    result.append(([token.i], salience_seqs[token.i]))
            # (3) Sort and extract top k
            result.sort(key=lambda x: x[1], reverse=True)
            final_set = result[:k]
            # (4) Format for writing back
            formatted_result = []
            for phrase, value in final_set:
                words = u' '.join([str(idx) for idx in phrase])
                formatted_result.append(words)
            updated_line = f'{orig_src_seq}\t{tgt_seq}\t{u"￨".join(formatted_result)}'
            updated_lines.append(updated_line)
        except ValueError:
            continue
    (tsv.parent / f'{tsv.name.split(".")[0]}.constraint.tsv').open('w')\
        .write('\n'.join(updated_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsv', help='TSV file.')
    parser.add_argument('-k', help='Top phrases to use.', type=int)
    args = parser.parse_args()
    main()
