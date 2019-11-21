'''
Named Entity Recognition
'''
import pickle
from collections import Counter

import spacy


def run(max_words, doc):
    ent_counter = Counter([ent.lower_ for ent in doc.ents]).most_common(n=max_words)
    ents = [ent[0] for ent in ent_counter]
    result_label = []
    for word in doc:
        if word.lower_ in ents:
            result_label.append(1)
        else:
            result_label.append(0)
    return result_label
