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
    src_folder_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_highres/Documents'
    tgt_folder_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_highres/Summaries'
    src_folders = sorted(os.listdir(src_folder_path))
    tgt_folders = sorted(os.listdir(tgt_folder_path))
    contents = []
    for src_filename, tgt_filename in zip(src_folders, tgt_folders):
        assert src_filename.split('.')[0] == tgt_filename.split('.')[0]
        with open(os.path.join(src_folder_path, src_filename)) as src_file:
            with open(os.path.join(tgt_folder_path, tgt_filename)) as tgt_file:
                contents.append('{}\t{}\n'.format(
                    src_file.readlines()[0].strip(), tgt_file.readlines()[0].strip()))
    file = open(os.path.join('../data/bbc_highres/', 'test.tsv'), 'w')
    file.writelines(contents)
    file.close()
