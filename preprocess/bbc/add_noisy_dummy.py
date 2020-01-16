"""
This is just for adding noisy dummy data
"""
import numpy as np

def gen_lines(path):
    with open(path) as file:
        new_lines = []
        i = 0
        for line in file:
            print(i)
            i += 1
            src, tgt = line.strip().split('\t')
            words = src.split()
            alphas = np.random.beta(a=2, b=5, size=len(words))
            probs = np.random.dirichlet(alphas, 1)
            comb = []
            for word, prob in zip(words, np.squeeze(probs, axis=0).tolist()):
                comb.append('{}|#|{:.6f}'.format(word, prob))
            new_lines.append(' '.join(comb) + '\t' + tgt + '\n')
        return new_lines

train_path = '../../data/dev_bbc/train.dev.tsv'
val_path = '../../data/dev_bbc/val.dev.tsv'
tagged_train_path = '../../data/dev_bbc/train.dev.tsv.tagged'
tagged_val_path = '../../data/dev_bbc/val.dev.tsv.tagged'
tagged_train_file = open(tagged_train_path, 'w')
tagged_val_file = open(tagged_val_path, 'w')
tagged_train_file.writelines(gen_lines(train_path))
tagged_val_file.writelines(gen_lines(val_path))
tagged_val_file.close()
tagged_train_file.close()
