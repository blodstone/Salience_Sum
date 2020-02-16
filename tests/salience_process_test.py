import argparse
from pathlib import Path


def main():
    input_path = Path(args.input)
    lines = input_path.open('r').readlines()
    for i, line in enumerate(lines):
        if line.strip() == '':
            continue
        src_seq, tgt_seq = line.split('\t')
        collection_seq = list(zip(*[group.split(u'￨') for group in src_seq.split()]))
        src_seq = ' '.join(collection_seq[0])
        salience_seqs = [[float(value) for value in seq] for seq in collection_seq[1:]]
        assert len(tgt_seq) != 0
        for model in salience_seqs:
            assert len(model) == len(collection_seq[0]), f'{len(model)}, {len(collection_seq[0])}'
        print(f'Process {i}: passed')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-input', help='The input text file (tsv).')
    args = parse.parse_args()
    main()
