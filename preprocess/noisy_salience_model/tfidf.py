import math
from collections import Counter

from noisy_salience_model.salience_model import Dataset, SalienceSet


def process(dataset: Dataset) -> Dataset:
    tfidf = dict()
    tf_count = Counter()
    idf_count = dict()
    for doc_id, instance in dataset.items():
        idf_count[doc_id] = Counter()
        for line in instance.doc:
            tf_count.update(line)
            idf_count[doc_id].update(line)
    for word, tf in tf_count.items():
        idf = math.log(
            len(idf_count) /
            (1 + sum([1 for _, counter in idf_count.items() if counter[word] > 0]))) + 1
        if tf != 0:
            tf = math.log(1 + tf)
        tfidf[word] = tf * idf
    for doc_id, instance in dataset.dataset.items():
        salience = SalienceSet.init_salience_set(instance.doc_size)
        for i, line in enumerate(instance.doc):
            for j, word in enumerate(line):
                salience[i][j] = tfidf[word]
        dataset[doc_id].salience_set['tfidf'] = salience
    return dataset
