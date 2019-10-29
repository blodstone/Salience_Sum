import argparse
import pickle
import os

import spacy


def main():
    src_data = open(args.src).readlines()
    output_file = open(os.path.join(args.output, 'train_src.pickle'), 'wb')
    docs = list(nlp.pipe(src_data))
    pickle.dump(docs, output_file)


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Source document.')
    parser.add_argument('-tgt', help='Target document.')
    parser.add_argument('-val', help='Validation document.')
    parser.add_argument('-output', help='Output folder.')
    args = parser.parse_args()
    main()
