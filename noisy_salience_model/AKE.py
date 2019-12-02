'''
Automated Keyword Retrieval using Textrank
'''
import nltk
from collections import namedtuple

import spacy
import networkx as nx


Labeled_Word = namedtuple('Labeled_Word', ['word', 'label', 'idx', 'sent_idx', 'score'])
desired_pos = ['JJ', 'JJR', 'NN', 'NNS', 'NNP', 'NNPS', ]


def create_labeled_doc(doc) -> list:
    '''
    Creates initial labeled documents and numbers the index only for the
    desired tags
    :param doc: Document splitted from spacy
    :return: Labeled document
    '''
    result = []
    idx = sent_idx = 0
    for sent in doc:
        sent_idx += 1
        set_tagged = nltk.pos_tag(sent.split())
        for pair in set_tagged:
            word, pos = pair
            if pos in desired_pos:
                result.append(Labeled_Word(word.lower(), 1, idx, sent_idx, 0))
            else:
                result.append(Labeled_Word(word.lower(), 0, idx, sent_idx, 0))
            idx += 1
    return result


def run(max_words, window, doc):
    # Split and create initial label
    labeled_doc = create_labeled_doc(doc)
    # Build graph
    graph = nx.Graph()
    i = 0
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
    top_n_ranks = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:max_words])
    # Score keywords
    for i, _ in enumerate(labeled_doc):
        if labeled_doc[i].word in top_n_ranks.keys():
            labeled_doc[i] = Labeled_Word(labeled_doc[i].word, 1,
                                          labeled_doc[i].idx, labeled_doc[i].sent_idx,
                                          ranks[labeled_doc[i].word])
        else:
            labeled_doc[i] = Labeled_Word(labeled_doc[i].word, 0,
                                          labeled_doc[i].idx, labeled_doc[i].sent_idx,
                                          -1)
    return [labeled_word.label for labeled_word in labeled_doc]
