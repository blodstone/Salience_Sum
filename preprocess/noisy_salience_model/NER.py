'''
Named Entity Recognition
'''
from collections import Counter


def run(max_words, doc, nlp):
    nlp_doc = list(nlp.pipe(doc))
    result_label = []
    for i, adoc in enumerate(nlp_doc):
        label = [0]
        for ent in adoc.ents:
            if ent.start_char == 0:
                label = [1]
                break
        for j, c in enumerate(doc[i]):
            if c == ' ' or c == u'\xa0':
                alabel = 0
                for ent in adoc.ents:
                    if ent.start_char <= j+1 <= ent.end_char:
                        alabel = 1
                        break
                label.append(alabel)
        assert len(label) == len(doc[i].split())
        result_label.extend(label)
    # ent_counter = Counter([ent.lower_ for doc in nlp_doc for ent in doc.ents]).most_common(n=max_words)
    # ents = [ent[0] for ent in ent_counter]
    # result_label = []
    # for sent in doc:
    #     for word in sent.split():
    #         if word in ents:
    #             result_label.append(1)
    #         else:
    #             result_label.append(0)
    return result_label

if __name__ == '__main__':
    pass