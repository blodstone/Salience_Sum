#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from tqdm import tqdm


def main():
    input_file = Path(args.input).open().readlines()
    output_file = (Path(args.output) / f'{Path(args.input).stem}.san').open('w')
    sanitized_file = []
    for line in tqdm(input_file):
        if line.strip() == '':
            continue
        src_seq, tgt_seq = line.split('\t')
        collection_seq = list(zip(*[group.split(u'￨') for group in src_seq.split()]))
        salience_seqs = [[float(value) if float(value) <= 1.0 else 1.0 for value in seq] for seq in collection_seq[1:]]
        src_seq = collection_seq[0]
        saliences = list(zip(*salience_seqs))
        output_line = []
        for line, salience_values in zip(src_seq, saliences):
            output_token = u'￨'.join([str(t) for t in list([line]) + list(salience_values)])
            output_line.append(output_token)
        sanitized_file.append(' '.join(output_line) + '\t' + tgt_seq)
    output_file.write('\n'.join(sanitized_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='The input that need to be sanitized (it has to be concat format).')
    parser.add_argument('-output', help='The output of the sanitized.')
    args = parser.parse_args()
    main()