import argparse
import pickle
import os

import spacy
from allennlp.common import Tqdm


def preprocess(filename, name):
    output_file = open(os.path.join(args.output, name + '.pickle'), 'wb')

    with open(filename) as file:
        docs = []
        summs = []
        for line in Tqdm.tqdm(file):
            doc, summ = line.strip().split('\t')
            docs.append(nlp(doc))
            summs.append(nlp(summ))
        docs_spacy = docs
        summs_spacy = summs
        dataset = {
            'docs': docs_spacy,
            'summs': summs_spacy
        }
        pickle.dump(dataset, output_file)
        output_file.close()


def main():
    preprocess(args.tsv, args.name)


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsv', help='TSV document.')
    parser.add_argument('-output', help='Output folder.')
    parser.add_argument('-name', help='Output name.')
    args = parser.parse_args()
    main()

