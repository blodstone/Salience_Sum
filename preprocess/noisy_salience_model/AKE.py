'''
Automated Keyword Retrieval using Textrank
'''
from typing import List

import networkx as nx
import nltk
from collections import namedtuple

import spacy

from noisy_salience_model.salience_model import Instance, Salience, SalienceSet, Text

Labeled_Word = namedtuple('Labeled_Word', ['word', 'label', 'idx', 'sent_idx', 'score'])
desired_pos = ['JJ', 'JJR', 'NN', 'NNS', 'NNP', 'NNPS', ]


def create_labeled_doc(doc: Text) -> list:
    '''
    Creates initial labeled documents and numbers the index only for the
    desired tags
    :param doc: Document splitted from spacy
    :return: Labeled document
    '''
    labeled_doc = []
    for sent_idx, sent in enumerate(doc):
        set_tagged = nltk.pos_tag(sent)
        for idx, pair in enumerate(set_tagged):
            word, pos = pair
            if pos in desired_pos:
                labeled_doc.append(Labeled_Word(word.lower(), 1, idx, sent_idx, 0))
            else:
                labeled_doc.append(Labeled_Word(word.lower(), 0, idx, sent_idx, 0))
    return labeled_doc


def process(salience_instance: Instance, window: int) -> Salience:
    # Split and create initial label
    labeled_doc = create_labeled_doc(salience_instance.doc)
    # Build graph
    graph = nx.Graph()
    # Add pair to graph from labeled doc that has no -1 index
    for i, labeled_word in enumerate(labeled_doc[:len(labeled_doc)]):
        if labeled_word.label == 0:
            continue
        j = i + 1
        window_taken = 0
        # Only add word that is within the window
        while j < len(labeled_doc) and window_taken < window \
                and labeled_doc[i].sent_idx == labeled_doc[j].sent_idx:
            if labeled_doc[j].label == 0:
                j += 1
                continue
            try:
                graph[labeled_doc[i].word][labeled_doc[j].word]['weight'] += 1.0
            except KeyError:
                graph.add_edge(labeled_doc[i].word, labeled_doc[j].word, weight=1.0)
            j += 1
            window_taken += 1
    # g = Network(height=800, width=800)
    # g.from_nx(graph)
    # g.show('graph.html')
    # Retrieve top N keywords
    ranks = nx.pagerank(graph)
    top_n_ranks = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:100])
    salience = SalienceSet.init_salience_set(salience_instance.doc_size)
    # Score keywords
    for i, _ in enumerate(labeled_doc):
        if labeled_doc[i].word in top_n_ranks.keys():
            salience[labeled_doc[i].sent_idx][labeled_doc[i].idx] = 1
        else:
            salience[labeled_doc[i].sent_idx][labeled_doc[i].idx] = 0
    return salience
