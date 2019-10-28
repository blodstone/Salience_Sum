import argparse
from noisy_salience_model import AKE


def main(args):
    max_words = args.max_words
    src = open(args.src).readlines()
    if args.AKE:
        AKE.run(max_words, src)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='Source document.')
    parser.add_argument('submodular', help='Submodular.')
    parser.add_argument('compression', help='Compression.')
    parser.add_argument('AKE', help='Automated Keyword Extraction using textrank.')
    parser.add_argument('NER', help='Named Entity Recognition.')
    parser.add_argument('max_words', help='Maximum words.')
    args = parser.parse_args()
    main(args)