from collections import namedtuple

import spacy
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

Labeled_Word = namedtuple('Labeled_Word', ['word', 'label', 'idx', 'sent_idx', 'score'])
desired_pos = ['NOUN', 'ADJ']


def create_labeled_doc(doc) -> list:
    '''
    Creates initial labeled documents and numbers the index only for the
    desired tags
    :param doc: Document splitted from spacy
    :return: Labeled document
    '''
    result = []
    idx = sent_idx = 0
    for sent in doc.sents:
        sent_idx += 1
        for word in sent:
            if word.pos_ in desired_pos:
                result.append(Labeled_Word(word.lower_, 1, idx, sent_idx, 0))
            else:
                result.append(Labeled_Word(word.lower_, 0, idx, sent_idx, 0))
            idx += 1
    return result


def run(max_words, window, src):
    nlp = spacy.load('en_core_web_sm')
    # Split and create initial label
    labeled_docs = [create_labeled_doc(doc) for doc in list(nlp.pipe(src))]
    for labeled_doc in labeled_docs:
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
        g = Network(height=800, width=800)
        g.from_nx(graph)
        g.show('graph.html')
        # nx.draw_networkx(graph, pos=nx.spring_layout(graph))
        # Retrieve top N keywords
        ranks = nx.pagerank(graph)
        top_n_ranks = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:max_words])
        # Score keywords
        i = 0
        while i < len(labeled_doc):
            if labeled_doc[i].word in top_n_ranks.keys():
                labeled_doc[i] = Labeled_Word(labeled_doc[i].word, 1,
                                              labeled_doc[i].idx, labeled_doc[i].sent_idx,
                                              ranks[labeled_doc[i].word])
            else:
                labeled_doc[i] = Labeled_Word(labeled_doc[i].word, 0,
                                              labeled_doc[i].idx, labeled_doc[i].sent_idx,
                                              -1)
            i += 1
    return labeled_docs


if __name__ == '__main__':
    src = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
    max_words = 22
    window = 2
    labeled_docs = run(max_words, window, src.split('\n'))
    for doc in labeled_docs:
        i = 1
        print('Words that are labeled as 1.')
        for word in doc:
            if word.label == 1:
                print('{}:{}'.format(i, word))
                i += 1
        i = 1
        print('Words that are labeled as 0.')
        for word in doc:
            if word.label == 0:
                print('{}:{}'.format(i, word))
                i += 1