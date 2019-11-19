import argparse
import pickle
import os

import spacy


def preprocess(filename, name):
    src_data = open(filename).readlines()
    output_file = open(os.path.join(args.output, name+'.pickle'), 'wb')
    docs = list(nlp.pipe(src_data))
    pickle.dump(docs, output_file)
    output_file.close()


def main():
    # preprocess(args.src, 'train_src')
    preprocess(args.tgt, 'textrank_train_tgt')


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Source document.')
    parser.add_argument('-tgt', help='Target document.')
    parser.add_argument('-val', help='Validation document.')
    parser.add_argument('-output', help='Output folder.')
    args = parser.parse_args()
    main()
