import math
from collections import Counter

from noisy_salience_model.salience_model import Dataset, SalienceSet


def process(dataset: Dataset) -> Dataset:
    idf = dict()
    tf_count_all = Counter()
    tf_count = dict()
    for doc_id, instance in dataset.items():
        tf_count[doc_id] = Counter()
        for line in instance.doc:
            tf_count_all.update(line)
            tf_count[doc_id].update(line)
    for word, tf in tf_count_all.items():
        idf[word] = math.log(
            len(tf_count) /
            (1 + sum([1 for _, counter in tf_count.items() if counter[word] > 0])), 2) + 1
    for doc_id, instance in dataset.dataset.items():
        salience = SalienceSet.init_salience_set(instance.doc_size)
        for i, line in enumerate(instance.doc):
            for j, word in enumerate(line):
                salience[i][j] = idf[word] * math.log(tf_count[doc_id][word], 2)
        dataset[doc_id].salience_set['tfidf'] = salience
    return dataset
