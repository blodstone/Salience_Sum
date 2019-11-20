"""
Take a src and tgt files (standard input in opennmt and fairseq) and join them into a single tsv
"""
import os

from allennlp.common import Tqdm


def gen_line(path):
    file = open(path)
    for line in file:
        yield line
    file.close()


if __name__ == '__main__':
    src_path = '../data/bbc/val.txt.src'
    tgt_path = '../data/bbc/val.txt.tgt'
    src = gen_line(src_path)
    tgt = gen_line(tgt_path)
    new_lines = []
    i = 0
    for src_line, tgt_line in Tqdm.tqdm(zip(src, tgt)):
        new_line = src_line.strip() + '\t' + tgt_line.strip() + '\n'
        new_lines.append(new_line)
    file = open(os.path.join('../data/bbc/', os.path.basename(src_path).split('.')[0]+'.tsv'), 'w')
    file.writelines(new_lines)
    file.close()
