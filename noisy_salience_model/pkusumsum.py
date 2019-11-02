import os
import pickle
import spacy


def run(src_path, tgt_path):
    docs = pickle.load(open(src_path, 'rb'))
    summs = pickle.load(open(tgt_path, 'rb'))
    result_labels = []
    for i in range(len(docs)):
        result_label = []
        doc = docs[i]
        summ = summs[i]
        summ_sents = list(summ.sents)
        for doc_sent in doc.sents:
            found = False
            for summ_sent in summ_sents:
                if doc_sent.similarity(summ_sent) > 0.9:
                    found = True
            if found:
                result_label.extend([1 for _ in doc_sent])
            else:
                result_label.extend([0 for _ in doc_sent])
        result_labels.append(result_label)
    return result_labels


if __name__ == '__main__':
    src_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/sample_data/train_src.pickle'
    tgt_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/sample_data/ext_train_tgt.pickle'
    run(src_path, tgt_path)
