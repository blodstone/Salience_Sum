import argparse
from pathlib import Path


def main():
    tsv = Path(args.tsv)
    updated_lines = []
    i = 1
    for line in tsv.open('r'):
        line = line.strip()
        if line == '':
            continue
        print(i)
        i += 1
        orig_src_seq, tgt_seq, const_seq = line.split('\t')
        collection_seq = list(zip(*[group.split(u'￨') for group in orig_src_seq.split()]))
        src_seq = collection_seq[0]

        salience_seqs = [[float(value) if float(value) <= 1.0 else 1.0 for value in seq] for seq in
                         collection_seq[1:]]
        salience_seqs = [sum(saliences) for saliences in zip(*salience_seqs)]
        const_seq = [constraint.split() for constraint in const_seq.split(u'￨')]
        for phrase in const_seq:
            vocab_indexes = []
            for word_idx in phrase:
                text = src_seq[int(word_idx)]
    print('all clear')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsv', help='TSV file.')
    args = parser.parse_args()
    main()
