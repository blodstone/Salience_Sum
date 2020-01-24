#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path


def write_to_file(src_seq, salience_seq, tgt_seq, tsv_file, src_file, tgt_file):
    output_line = []
    for token, salience_value in zip(src_seq, salience_seq):
        output_token = f'{token}￨{salience_value}'
        output_line.append(output_token)
    tsv_file.open('a').write(' '.join(output_line) + '\t' + ' '.join(tgt_seq) + '\n')
    src_file.open('a').write(' '.join(output_line) + '\n')
    tgt_file.open('a').write(' '.join(tgt_seq) + '\n')


def salience_sum(salience_seqs):
    salience_seqs = [[float(value) for value in salience_seq] for salience_seq in salience_seqs]
    salience_seq = [str(sum(joined_salience)) for joined_salience in zip(*salience_seqs)]
    return salience_seq


def salience_concat(salience_seqs):
    salience_seqs = [[str(float(value)) for value in salience_seq] for salience_seq in salience_seqs]
    salience_seq = [u'￨'.join(joined_salience) for joined_salience in zip(*salience_seqs)]
    return salience_seq


def main():
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    for set_name in args.set:
        output_folder.mkdir(parents=True, exist_ok=True)
        input_folders = [(input_folder / folder_name) for folder_name in args.folders]
        tsv_txt_files = [(input_folder / f'{set_name}.raw.tsv').open().readlines() for input_folder in input_folders]
        if 'sum' in args.modes:
            mode = 'sum'
            src_file, tgt_file, tsv_file = create_output_files(mode, output_folder, set_name)
            process(tsv_txt_files, mode, src_file, tgt_file, tsv_file)
        if 'concat' in args.modes:
            mode = 'concat'
            src_file, tgt_file, tsv_file = create_output_files(mode, output_folder, set_name)
            process(tsv_txt_files, mode, src_file, tgt_file, tsv_file)


def process(tsv_txt_files, mode, src_file, tgt_file, tsv_file):
    for lines in zip(*tsv_txt_files):
        doc_ids = [line.split('\t')[0] for line in lines]
        assert len(list(set(doc_ids))) == 1
        tgt_seq = lines[0].split('\t')[2]
        src_seq = list(zip(*[group.split(u'￨') for group in lines[0].split('\t')[1].split()]))[0]
        salience_seqs = [list(zip(*[group.split(u'￨') for group in line.split('\t')[1].split()]))[1]
                         for line in lines]
        salience_seq = None
        if mode == 'sum':
            salience_seq = salience_sum(salience_seqs)
        elif mode == 'concat':
            salience_seq = salience_concat(salience_seqs)
        write_to_file(src_seq, salience_seq, tgt_seq, tsv_file, src_file, tgt_file)


def create_output_files(mode, output_folder, set_name):
    tsv_file = output_folder / f'{set_name}.{mode}.tsv'
    src_file = output_folder / f'{set_name}.{mode}.src.txt'
    tgt_file = output_folder / f'{set_name}.{mode}.tgt.txt'
    if tsv_file.exists():
        print(f'{str(tsv_file)} exists. Deleting file.')
        os.remove(str(tsv_file))
    if src_file.exists():
        print(f'{str(src_file)} exists. Deleting file.')
        os.remove(str(src_file))
    if tgt_file.exists():
        print(f'{str(tgt_file)} exists. Deleting file.')
        os.remove(str(tgt_file))
    return src_file, tgt_file, tsv_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_folder', help='Input path containing folders.')
    parser.add_argument('-folders', nargs='+', help="Folder to be combined.")
    parser.add_argument('-set', nargs='+', help='Set name.')
    parser.add_argument('-output_folder', help='Output path.')
    parser.add_argument('-modes', nargs='+', help='The combination mode [concat | sum].')
    args = parser.parse_args()
    main()
