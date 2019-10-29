'''
Named Entity Recognition
'''
import pickle
from collections import Counter

import spacy


def run(max_words, src_path):
    nlp = spacy.load('en_core_web_sm')
    docs = pickle.load(open(src_path, 'rb'))
    result_labels = []
    for doc in docs:
        ent_counter = Counter([ent.lower_ for ent in doc.ents]).most_common(n=max_words)
        ents = [ent[0] for ent in ent_counter]
        result_label = []
        for word in doc:
            if word.lower_ in ents:
                result_label.append(1)
            else:
                result_label.append(0)
        result_labels.append(result_label)
    return result_labels


if __name__ == '__main__':
    src_path = '../sample_data/train_src.pickle'
    result_labels = run(2, src_path)
    print(result_labels)
