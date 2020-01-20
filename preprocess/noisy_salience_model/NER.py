'''
Named Entity Recognition
'''
from collections import Counter
from typing import List, Any

from noisy_salience_model.salience_model import Salience, SalienceSet, Instance


def process(salience_instance: Instance, nlp: Any) -> Salience:
    salience = SalienceSet.init_salience_set(salience_instance.doc_size)
    for i, line in enumerate(salience_instance.raw):
        nlp_doc = nlp(' '.join(line))
        for j, token in enumerate(nlp_doc):
            salience[i][j] = 0 if token.ent_type_ == '' else 1
    return salience

if __name__ == '__main__':
    pass
