import math
from collections import Counter

from noisy_salience_model.salience_model import Dataset, SalienceSet


def process(dataset: Dataset, doc_word_count: Counter) -> Dataset:
    if not dataset.dataset_name == 'train':
        for doc_id, instance in dataset.dataset.items():
            if doc_id in doc_word_count.keys():
                continue
            else:
                doc_word_count[doc_id] = Counter()
                for line in instance.doc:
                    doc_word_count[doc_id].update(line)
    i = 1
    save_word_doc_number = dict()
    for doc_id, instance in dataset.dataset.items():
        print(f'Processed tfidf ({i}): {doc_id}')
        i += 1
        salience = SalienceSet.init_salience_set(instance.doc_size)
        for i, line in enumerate(instance.doc):
            for j, word in enumerate(line):
                doc_size = len(doc_word_count.keys())
                word_doc_number = save_word_doc_number.get(
                    word, sum([1 for _, counter in doc_word_count.items() if counter[word] > 0]))
                save_word_doc_number[word] = word_doc_number
                idf = math.log(doc_size / (1 + word_doc_number)) + 1
                salience[i][j] = idf * math.log(doc_word_count[doc_id][word] + 1)
                assert salience[i][j] != 0.0
        dataset[doc_id].salience_set['tfidf'] = salience
    return dataset
