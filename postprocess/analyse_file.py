from pathlib import Path
from collections import Counter
gold = Path('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen/gold.src.txt')
silver = Path('/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_allen/test.src.txt')

gold_result = dict()
for line in gold.open().readlines():
    src_seq, salience_seq = zip(*[group.split(u'￨') for group in line.split()])
    gold_result[''.join(src_seq)] = (src_seq, salience_seq)

silver_result = dict()
for line in silver.open().readlines():
    src_seq, submodular, textrank, centroid, lexpagerank, AKE, NER, tfidf = zip(*[group.split(u'￨') for group in line.split()])
    salience_seq = [float(a) + float(b) + float(c) + float(d) for a,b,c,d,e,f,g in zip(submodular, textrank, centroid, lexpagerank, AKE, NER, tfidf) ]
    silver_result[''.join(src_seq)] = (src_seq, salience_seq)

total_recall = 0
total_precision = 0
total_acc = 0
for idx, key in enumerate(sorted(silver_result.keys())):
    gold_keys = list(sorted(gold_result.keys()))
    gold_key = gold_keys[idx]
    gold_seq, gold_salience_seq = gold_result[gold_key]
    silver_seq, silver_salience_seq = silver_result[key]
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    all_pos = 0
    all_neg = 0
    for g, s, g_t, s_t in zip(gold_salience_seq, silver_salience_seq, gold_seq, silver_seq):
        if g_t in ['``', '"', ')', '(', '€', '`', '£', '3']:
            g_t = s_t
        assert g_t == s_t, f'{g_t}, {s_t}'
        g = float(g)
        s = float(s)
        if g > 0:
            all_pos += 1
        else:
            all_neg += 1
        if g > 0 and s > 0:
           true_pos += 1
        elif g > 0 and s == 0:
            false_neg += 1
        elif g == 0 and s == 0:
            true_neg += 1
        elif g == 0 and s > 0:
            false_pos += 1
        else:
            print('error')
    recall = true_pos / all_pos
    precision = true_neg / all_neg
    acc = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg)
    total_recall += recall
    total_precision += precision
    total_acc += acc
result_recall = total_recall / len(silver_result.keys())
result_precision = total_precision / len(silver_result.keys())
result_acc = total_acc / len(silver_result.keys())

print(result_recall)
print(result_precision)
print(result_acc)
