import argparse
import spacy
from spacy.tokens import Doc
from pathlib import Path

from typing import List

from tqdm import tqdm


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
        if a[j - 1] != i:
            while m[j - 1][i]:
                j -= 1
        searched_set.append(a[j - 1])
        i = i - a[j - 1][1]
        j -= 1
    return searched_set


def extract_phrases(doc_src, doc_tgt, max_len, salience_seqs, k, is_oracle):
    doc_chunks = doc_src.noun_chunks
    tgt_chunks = doc_tgt.noun_chunks
    added_words_src = []
    added_words_tgt = []
    result = []
    used_idx_src = []
    used_idx_tgt = []
    # (1) Extract noun chunks and calculate the average score of the chunks,
    # also remove stop words from noun chunks
    for chunk in doc_chunks:
        phrase_idxs = []
        score = 0
        phrase_words = []
        for token in chunk:
            if not token.is_stop and token.i < max_len:
                phrase_words.append(token.text)
                phrase_idxs.append(token.i)
                used_idx_src.append(token.i)
                score += salience_seqs[token.i]
        if len(phrase_idxs) != 0 and phrase_words not in added_words_src:
            score /= len(phrase_idxs)
            result.append((phrase_idxs, phrase_words, score))
            added_words_src.append(phrase_words)
    # (2) Extract single word with stop words and noun chunks removed from text
    for token in doc_src:
        if not token.is_stop and (token.i not in used_idx_src) and (token.i < max_len):
            if [token.text] not in added_words_src:
                result.append(([token.i], [token.text], salience_seqs[token.i]))
                added_words_src.append([token.text])

    if is_oracle:
        for chunk in tgt_chunks:
            phrase_idxs = []
            phrase_words = []
            for token in chunk:
                if not token.is_stop and token.i < max_len:
                    phrase_words.append(token.text)
                    phrase_idxs.append(token.i)
                    used_idx_tgt.append(token.i)
            if len(phrase_idxs) != 0 and phrase_words not in added_words_tgt:
                added_words_tgt.append(phrase_words)
        for token in doc_tgt:
            if not token.is_stop and not token.i in used_idx_tgt and token.i < max_len:
                if [token.text] not in added_words_tgt:
                    added_words_tgt.append([token.text])
        new_result = []
        for r in result:
            if r[1] in added_words_tgt:
                new_result.append(r)
        result = new_result
    if len(result) == 0:
        return []
    # (3) Sort and extract top k
    result.sort(key=lambda x: x[1], reverse=True)
    if is_oracle:
        return result
    else:
        return result[:k]


def main():
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    k = args.k
    is_oracle = args.oracle
    output_name = args.output_name
    max_len = args.max_len
    if max_len == -1:
        max_len = float('inf')
    tsv = Path(args.tsv)
    updated_lines = []
    count_empty_final_set = 0
    for line in tqdm(tsv.open('r').readlines()):
        line = line.strip()
        if line == '':
            continue
        orig_src_seq, tgt_seq = line.split('\t')
        collection_seq = list(zip(*[group.split(u'￨') for group in orig_src_seq.split()]))
        src_seq = collection_seq[0]

        salience_seqs = [[float(value) if float(value) <= 1.0 else 1.0 for value in seq] for seq in
                         collection_seq[1:]]
        salience_seqs = [sum(saliences) for saliences in zip(*salience_seqs)]

        doc_src = nlp(" ".join(src_seq))
        doc_tgt = nlp(tgt_seq)

        final_set = extract_phrases(doc_src, doc_tgt, max_len, salience_seqs, k, is_oracle)

        if len(final_set) == 0:
            print(count_empty_final_set)
            count_empty_final_set += 1
        # (4) Format for writing back
        formatted_result = []
        for phrase, _, value in final_set:
            words = u' '.join([str(idx) for idx in phrase])
            formatted_result.append(words)
        updated_line = f'{orig_src_seq}\t{tgt_seq}\t{u"￨".join(formatted_result)}'
        updated_lines.append(updated_line)
    (tsv.parent / f'{tsv.name.split(".")[0]}.{output_name}.tsv').open('w') \
        .write('\n'.join(updated_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsv', help='TSV file.')
    parser.add_argument('-k', help='Top phrases to use.', type=int)
    parser.add_argument('-max_len', help='Maximum length of text to consider.', type=int, default=-1)
    parser.add_argument('--oracle', help='Set to oracle', action='store_true')
    parser.add_argument('-output_name', help='The output name. The prefix is set to the input name.')
    args = parser.parse_args()
    main()
