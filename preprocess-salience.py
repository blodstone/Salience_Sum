import argparse
from noisy_salience_model import AKE, NER


def main(args):
    max_words = args.max_words
    results = {}
    src_path = args.src
    if args.AKE:
        window = args.window
        results['AKE'] = AKE.run(max_words, window, src_path)
    if args.NER:
        results['NER'] = NER.run(max_words, src_path)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Source document (pickle format).')
    parser.add_argument('--submodular', help='Submodular.',
                        action='store_true')
    parser.add_argument('--compression', help='Compression.',
                        action='store_true')
    parser.add_argument('--AKE', help='Automated Keyword Extraction (AKE) using textrank.',
                        action='store_true')
    parser.add_argument('-window', help='Window for Textrank, needed by AKE.',
                        default=5, type=int)
    parser.add_argument('--NER', help='Named Entity Recognition.',
                        action='store_true')
    parser.add_argument('-max_words', help='Maximum words.', default=35, type=int)
    args = parser.parse_args()
    main(args)
